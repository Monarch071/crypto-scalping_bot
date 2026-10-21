"""Торговый движок: связывает маркет-дату, стратегию, риск, OMS и шлюз.

Один и тот же движок работает в бэктесте (время берётся из событий)
и в реальном времени (время берётся из часов).
"""
from __future__ import annotations

import asyncio
import logging
import time

from .config import Config
from .events import BookEvent, Event, FillEvent, OrderStatus, OrderStatusEvent, Side, TradeEvent
from .gateways.base import Gateway
from .marketdata import Recorder
from .oms import Order, OrderManager
from .orderbook import OrderBook
from .portfolio import Portfolio
from .risk import RiskManager
from .strategy import MarketMaker
from .telegram import Notifier

log = logging.getLogger(__name__)


class Engine:
    def __init__(self, cfg: Config, gateway: Gateway, notifier: Notifier | None = None,
                 recorder: Recorder | None = None, realtime: bool = False) -> None:
        self.cfg = cfg
        self.gateway = gateway
        self.notifier = notifier or Notifier()
        self.recorder = recorder
        self.realtime = realtime
        self.book = OrderBook()
        self.oms = OrderManager()
        self.portfolio = Portfolio()
        self.risk = RiskManager(cfg.risk)
        self.strategy: MarketMaker | None = None
        self.queue: asyncio.Queue[Event] = asyncio.Queue()
        self.gateway.emit = self.emit
        self._sim_now = 0.0
        self._last_quote_ts = -1e18
        self._kill_notified = False
        self._stale_handled = False
        self._tasks: set[asyncio.Task] = set()
        self._last_reject_log = 0.0

    # --- жизненный цикл ---
    async def start(self) -> None:
        await self.gateway.start()
        self.strategy = MarketMaker(self.cfg.strategy, self.gateway.instrument(), self.cfg.risk.max_position)

    def now(self) -> float:
        return time.time() if self.realtime else self._sim_now

    def emit(self, ev: Event) -> None:
        self.queue.put_nowait(ev)

    async def drain(self) -> None:
        """Обработать все накопившиеся события (используется в бэктесте)."""
        while not self.queue.empty():
            await self.handle(self.queue.get_nowait())

    async def run_forever(self) -> None:
        while True:
            await self.handle(await self.queue.get())

    def _submit(self, coro) -> None:
        # В реальном времени вызовы REST не должны блокировать обработку маркет-даты.
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _call(self, coro) -> None:
        if self.realtime:
            self._submit(coro)
        else:
            await coro

    # --- обработка событий ---
    async def handle(self, ev: Event) -> None:
        if isinstance(ev, BookEvent):
            self._sim_now = max(self._sim_now, ev.ts)
            if self.recorder:
                self.recorder.write(ev)
            self.book.update(ev.bids, ev.asks, ev.ts)
            self.risk.on_market_data(self.now())
            self.gateway.on_market_event(ev)
            if self.book.ready:
                self.strategy.on_book(self.book)
            await self.on_tick()
        elif isinstance(ev, TradeEvent):
            self._sim_now = max(self._sim_now, ev.ts)
            if self.recorder:
                self.recorder.write(ev)
            self.gateway.on_market_event(ev)
            self.strategy.on_trade(ev)
        elif isinstance(ev, OrderStatusEvent):
            self.oms.on_status(ev)
        elif isinstance(ev, FillEvent):
            if self.oms.on_fill(ev) is None:
                return
            self.portfolio.on_fill(ev)
            log.info("FILL %s %.6f @ %.2f fee=%.4f pos=%.6f", ev.side.value, ev.size, ev.price,
                     ev.fee, self.portfolio.position)

    async def on_tick(self) -> None:
        now = self.now()
        if not self.book.ready:
            return
        self.risk.check_pnl(self.portfolio.total_pnl(self.book.mid), now)
        if self.risk.killed:
            await self._while_killed()
            return
        if self.risk.data_is_stale(now):
            if not self._stale_handled:
                log.warning("Нет свежих данных, снимаю котировки")
                await self._cancel_all_working()
                self._stale_handled = True
            return
        self._stale_handled = False
        if now - self._last_quote_ts < self.cfg.strategy.min_requote_interval_s:
            return
        self._last_quote_ts = now
        await self._requote(now)

    async def _requote(self, now: float) -> None:
        quotes = {q.side: q for q in self.strategy.quotes(self.book, self.portfolio.position)}
        threshold = self.book.mid * self.cfg.strategy.requote_threshold_bps / 1e4
        for side in (Side.BUY, Side.SELL):
            working = self.oms.working(side)
            # Ждём подтверждения от биржи, чтобы не оказаться с двумя ордерами на одной стороне.
            if any(o.status in (OrderStatus.PENDING_NEW, OrderStatus.PENDING_CANCEL) for o in working):
                continue
            q = quotes.get(side)
            if working:
                keep = working[0]
                for extra in working[1:]:
                    await self._cancel(extra)
                if q is None or abs(keep.price - q.price) >= threshold:
                    await self._cancel(keep)
                continue
            if q is None:
                continue
            reason = self.risk.check_order(side, q.price, q.size, self.portfolio.position,
                                           self.oms.working(), self.book.mid, now)
            if reason:
                if now - self._last_reject_log > 10:
                    log.info("Риск отклонил %s %.6f @ %.2f: %s", side.value, q.size, q.price, reason)
                    self._last_reject_log = now
                continue
            order = self.oms.create(side, q.price, q.size, now)
            await self._call(self.gateway.place(order))
        self.oms.prune()

    async def _cancel(self, order: Order) -> None:
        if order.status is OrderStatus.PENDING_CANCEL:
            return
        order.status = OrderStatus.PENDING_CANCEL
        await self._call(self.gateway.cancel(order))

    async def _cancel_all_working(self) -> None:
        for o in self.oms.working():
            if not o.market:
                await self._cancel(o)

    async def _while_killed(self) -> None:
        if not self._kill_notified:
            self._kill_notified = True
            await self._cancel_all_working()
            await self._call(self.gateway.cancel_all())
            await self.notifier.send(f"⛔ Торговля остановлена: {self.risk.kill_reason}\n{self.status()}")
            return
        await self._cancel_all_working()
        pos = self.portfolio.position
        lot = self.strategy.instrument.lot_size
        if self.cfg.risk.flatten_on_kill and abs(pos) >= lot / 2 and not self.oms.working():
            side = Side.SELL if pos > 0 else Side.BUY
            size = self.strategy.instrument.round_size(abs(pos)) or lot
            log.warning("Закрываю позицию %.6f рыночным ордером", pos)
            order = self.oms.create(side, self.book.mid, size, self.now(), reduce_only=True, market=True)
            await self._call(self.gateway.place(order))

    # --- управление ---
    async def kill(self, reason: str = "ручная остановка") -> str:
        self.risk.kill(reason)
        await self.on_tick()
        return "Останавливаю торговлю"

    async def resume(self) -> str:
        self.risk.reset()
        self._kill_notified = False
        return "Торговля возобновлена"

    def status(self) -> str:
        p = self.portfolio
        mark = self.book.mid if self.book.ready else 0.0
        state = f"KILLED ({self.risk.kill_reason})" if self.risk.killed else \
            (self.strategy.state if self.strategy else "init")
        return (f"Состояние: {state}\n"
                f"Позиция: {p.position:.6f} @ {p.avg_price:.2f}\n"
                f"PnL: {p.total_pnl(mark):.4f} (реализ. {p.realized:.4f}, комиссии {p.fees:.4f})\n"
                f"Сделок: {p.fills}, оборот {p.volume:.2f}\n"
                f"Рабочих ордеров: {len(self.oms.working())}")
