"""Симулятор биржи для paper trading и бэктеста.

Учитывает то, о чём обычно забывают любительские бэктесты:
  * задержку сети: ордер появляется в стакане и снимается не мгновенно;
  * post-only: ордер, пересекающий спред, отклоняется;
  * очередь в стакане: лимитный ордер считается исполненным, только если
    рынок прошёл сквозь его цену, а не просто коснулся её;
  * объём сделки: исполняется не больше, чем реально проторговалось;
  * комиссии мейкера и тейкера.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import ExchangeConfig
from ..events import BookEvent, Event, FillEvent, OrderStatus, OrderStatusEvent, Side, TradeEvent
from ..oms import Order
from ..strategy import Instrument
from .base import Gateway


@dataclass
class _Resting:
    order: Order
    remaining: float


class PaperGateway(Gateway):
    def __init__(self, cfg: ExchangeConfig, latency_ms: float = 80.0, fill_on_touch: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.latency = latency_ms / 1000
        self.fill_on_touch = fill_on_touch
        self.now = 0.0
        self.book: BookEvent | None = None
        self._pending: list[tuple[float, str, Order]] = []
        self._resting: dict[str, _Resting] = {}
        self._cancelled_early: set[str] = set()

    def instrument(self) -> Instrument:
        return Instrument(self.cfg.tick_size, self.cfg.lot_size)

    async def place(self, order: Order) -> None:
        self._pending.append((self.now + self.latency, "new", order))

    async def cancel(self, order: Order) -> None:
        self._pending.append((self.now + self.latency, "cancel", order))

    async def cancel_all(self) -> None:
        for r in list(self._resting.values()):
            await self.cancel(r.order)
        for _, kind, order in list(self._pending):
            if kind == "new":
                await self.cancel(order)

    # --- симуляция ---
    def on_market_event(self, ev: Event) -> None:
        if isinstance(ev, BookEvent):
            self.now = ev.ts
            self.book = ev
        elif isinstance(ev, TradeEvent):
            self.now = ev.ts
        else:
            return
        self._process_pending()
        if isinstance(ev, TradeEvent):
            self._match_trade(ev)
        else:
            self._match_book(ev)

    def _process_pending(self) -> None:
        due = [p for p in self._pending if p[0] <= self.now]
        if not due:
            return
        self._pending = [p for p in self._pending if p[0] > self.now]
        for _, kind, order in due:
            if kind == "new":
                self._activate(order)
            else:
                self._cancel_now(order)

    def _status(self, order: Order, status: OrderStatus, reason: str = "") -> None:
        self.emit(OrderStatusEvent(self.now, order.client_id, status, order.client_id, reason))

    def _activate(self, order: Order) -> None:
        if order.client_id in self._cancelled_early:
            self._cancelled_early.discard(order.client_id)
            self._status(order, OrderStatus.CANCELED)
            return
        book = self.book
        if book is None or not book.bids or not book.asks:
            self._status(order, OrderStatus.REJECTED, "нет стакана")
            return
        if order.market:
            self._fill_market(order, book)
            return
        crosses = (order.side is Side.BUY and order.price >= book.asks[0][0]) or \
                  (order.side is Side.SELL and order.price <= book.bids[0][0])
        if crosses:
            self._status(order, OrderStatus.REJECTED, "post-only: ордер пересёк бы спред")
            return
        self._resting[order.client_id] = _Resting(order, order.size)
        self._status(order, OrderStatus.OPEN)

    def _cancel_now(self, order: Order) -> None:
        if self._resting.pop(order.client_id, None) is not None:
            self._status(order, OrderStatus.CANCELED)
        elif any(o.client_id == order.client_id for _, k, o in self._pending if k == "new"):
            # Отмена дошла раньше самого ордера (обогнала его в сети).
            self._cancelled_early.add(order.client_id)

    def _fill(self, order: Order, price: float, size: float, maker: bool) -> None:
        fee_bps = self.cfg.maker_fee_bps if maker else self.cfg.taker_fee_bps
        self.emit(FillEvent(self.now, order.client_id, order.side, price, size,
                            price * size * fee_bps / 1e4, maker))

    def _fill_market(self, order: Order, book: BookEvent) -> None:
        levels = book.asks if order.side is Side.BUY else book.bids
        left = order.size
        for price, size in levels:
            take = min(left, size)
            if take > 0:
                self._fill(order, price, take, maker=False)
                left -= take
            if left <= 1e-12:
                break
        if left > 1e-12:
            self._status(order, OrderStatus.CANCELED, "недостаточно ликвидности")

    def _match_trade(self, t: TradeEvent) -> None:
        available = t.size
        for cid, r in list(self._resting.items()):
            o = r.order
            if o.side is Side.BUY:
                hit = t.price < o.price or (self.fill_on_touch and t.price == o.price)
            else:
                hit = t.price > o.price or (self.fill_on_touch and t.price == o.price)
            if not hit or available <= 0:
                continue
            qty = min(r.remaining, available)
            available -= qty
            self._fill(o, o.price, qty, maker=True)
            r.remaining -= qty
            if r.remaining <= 1e-12:
                del self._resting[cid]

    def _match_book(self, b: BookEvent) -> None:
        if not b.bids or not b.asks:
            return
        for cid, r in list(self._resting.items()):
            o = r.order
            through = (o.side is Side.BUY and b.asks[0][0] <= o.price) or \
                      (o.side is Side.SELL and b.bids[0][0] >= o.price)
            if through:
                self._fill(o, o.price, r.remaining, maker=True)
                del self._resting[cid]
