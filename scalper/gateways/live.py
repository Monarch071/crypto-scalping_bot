"""Шлюз к реальной бирже через ccxt (REST для ордеров, WebSocket для статусов).

ВНИМАНИЕ: с реальной биржей этот код ещё не проверялся. Перед использованием
гоняйте его на testnet (exchange.testnet = true).
"""
from __future__ import annotations

import asyncio
import logging
import time

from ..config import ExchangeConfig
from ..events import FillEvent, OrderStatus, OrderStatusEvent, Side
from ..oms import Order
from ..strategy import Instrument
from .base import Gateway

log = logging.getLogger(__name__)

_STATUS = {
    "open": OrderStatus.OPEN,
    "closed": OrderStatus.FILLED,
    "canceled": OrderStatus.CANCELED,
    "cancelled": OrderStatus.CANCELED,
    "expired": OrderStatus.CANCELED,
    "rejected": OrderStatus.REJECTED,
}


class LiveGateway(Gateway):
    def __init__(self, exchange, cfg: ExchangeConfig) -> None:
        super().__init__()
        self.ex = exchange  # общий экземпляр ccxt.pro с маркет-датой
        self.cfg = cfg
        self.symbol = cfg.symbol
        self._instrument: Instrument | None = None
        self._by_cid: dict[str, Order] = {}
        self._by_xid: dict[str, Order] = {}
        self._filled: dict[str, float] = {}
        self._cancel_requested: set[str] = set()
        self._tasks: list[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        markets = await self.ex.load_markets()
        m = markets[self.symbol]
        tick, lot = m["precision"]["price"], m["precision"]["amount"]
        if self.ex.precisionMode != 4:  # 4 = ccxt.TICK_SIZE; иначе это число знаков
            tick, lot = 10 ** -tick, 10 ** -lot
        self._instrument = Instrument(float(tick), float(lot))
        maker = m.get("maker")
        if maker is not None:
            self.cfg.maker_fee_bps = float(maker) * 1e4
        log.info("Рынок %s: tick=%s lot=%s maker=%.2fbps", self.symbol, tick, lot, self.cfg.maker_fee_bps)
        # Начинаем с чистого листа: снимаем всё, что осталось от прошлого запуска.
        await self.cancel_all()
        self._running = True
        self._tasks.append(asyncio.create_task(self._watch_orders()))

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        await self.cancel_all()

    def instrument(self) -> Instrument:
        assert self._instrument is not None, "start() ещё не вызывался"
        return self._instrument

    async def fetch_position(self) -> float:
        """Текущая позиция по деривативу (для сверки при старте)."""
        try:
            positions = await self.ex.fetch_positions([self.symbol])
        except Exception as e:  # noqa: BLE001
            log.warning("Не удалось получить позицию: %s", e)
            return 0.0
        total = 0.0
        for p in positions:
            contracts = float(p.get("contracts") or 0)
            total += contracts if p.get("side") == "long" else -contracts
        return total

    async def place(self, order: Order) -> None:
        self._by_cid[order.client_id] = order
        params = {"clientOrderId": order.client_id}
        if order.reduce_only:
            params["reduceOnly"] = True
        try:
            if order.market:
                res = await self.ex.create_order(self.symbol, "market", order.side.value,
                                                 order.size, None, params)
            else:
                params["postOnly"] = True
                res = await self.ex.create_order(self.symbol, "limit", order.side.value,
                                                 order.size, order.price, params)
        except Exception as e:  # noqa: BLE001
            self.emit(OrderStatusEvent(time.time(), order.client_id, OrderStatus.REJECTED, reason=str(e)))
            return
        xid = str(res.get("id")) if res.get("id") else None
        if xid:
            self._by_xid[xid] = order
        self.emit(OrderStatusEvent(time.time(), order.client_id, OrderStatus.OPEN, xid))
        self._apply(res)
        if order.client_id in self._cancel_requested:
            await self.cancel(order)

    async def cancel(self, order: Order) -> None:
        if order.exchange_id is None:
            # Ордер ещё не подтверждён биржей: снимем, как только придёт id.
            self._cancel_requested.add(order.client_id)
            return
        try:
            await self.ex.cancel_order(order.exchange_id, self.symbol)
        except Exception as e:  # noqa: BLE001
            # Чаще всего ордер уже исполнен или снят; статус придёт через watch_orders.
            log.info("Отмена %s не прошла: %s", order.client_id, e)

    async def cancel_all(self) -> None:
        try:
            await self.ex.cancel_all_orders(self.symbol)
        except Exception as e:  # noqa: BLE001
            log.error("cancel_all_orders не прошёл: %s", e)

    async def _watch_orders(self) -> None:
        backoff = 1.0
        while self._running:
            try:
                for o in await self.ex.watch_orders(self.symbol):
                    self._apply(o)
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                log.warning("watch_orders: %s, переподключение через %.0fс", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    def _apply(self, o: dict) -> None:
        """Превращает ответ ccxt в события. Исполнения считаем по приросту filled."""
        order = self._by_cid.get(o.get("clientOrderId") or "") or self._by_xid.get(str(o.get("id")))
        if order is None:
            return
        ts = (o.get("timestamp") or time.time() * 1000) / 1000
        filled = float(o.get("filled") or 0)
        prev = self._filled.get(order.client_id, 0.0)
        if filled > prev + 1e-12:
            delta = filled - prev
            self._filled[order.client_id] = filled
            price = float(o.get("average") or o.get("price") or order.price)
            maker = not order.market
            fee_bps = self.cfg.maker_fee_bps if maker else self.cfg.taker_fee_bps
            self.emit(FillEvent(ts, order.client_id, Side(order.side), price, delta,
                                price * delta * fee_bps / 1e4, maker))
        status = _STATUS.get(o.get("status") or "")
        if status is not None and status is not OrderStatus.FILLED:
            self.emit(OrderStatusEvent(ts, order.client_id, status, str(o.get("id"))))
