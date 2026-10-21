"""Источники рыночных данных: биржа (WebSocket), запись на диск, повтор и синтетика."""
from __future__ import annotations

import asyncio
import gzip
import json
import logging
import math
import random
from typing import Callable, Iterator

from .events import BookEvent, Event, Side, TradeEvent

log = logging.getLogger(__name__)


class Recorder:
    """Пишет маркет-дату в jsonl.gz. Из этих файлов потом делается бэктест."""

    def __init__(self, path: str) -> None:
        self._f = gzip.open(path, "at", encoding="utf-8")
        self._n = 0

    def write(self, ev: Event) -> None:
        if isinstance(ev, BookEvent):
            rec = {"t": "b", "ts": ev.ts, "b": ev.bids, "a": ev.asks}
        elif isinstance(ev, TradeEvent):
            rec = {"t": "t", "ts": ev.ts, "p": ev.price, "s": ev.size,
                   "d": ev.side.value if ev.side else None}
        else:
            return
        self._f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self._n += 1
        if self._n % 1000 == 0:
            self._f.flush()

    def close(self) -> None:
        self._f.close()


def replay(path: str) -> Iterator[Event]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["t"] == "b":
                yield BookEvent(r["ts"], [tuple(x) for x in r["b"]], [tuple(x) for x in r["a"]])
            else:
                yield TradeEvent(r["ts"], r["p"], r["s"], Side(r["d"]) if r["d"] else None)


def synthetic(n_seconds: int = 3600, seed: int = 1, start_price: float = 60000.0,
              tick: float = 0.1, sigma_bps_per_s: float = 1.0, start_ts: float = 1.7e9,
              jump_prob: float = 0.003, jump_bps: float = 4.0) -> Iterator[Event]:
    """Синтетический рынок: случайное блуждание со скачками, 10 обновлений стакана и ~3 сделки в секунду.

    Нужен только для проверки, что система работает. Прибыль на синтетике
    ничего не говорит о прибыли на настоящем рынке.
    """
    rng = random.Random(seed)
    mid = start_price
    dt = 0.1
    ts = start_ts
    step = sigma_bps_per_s / 1e4 * math.sqrt(dt)
    for _ in range(int(n_seconds / dt)):
        ts += dt
        mid *= math.exp(rng.gauss(0, step))
        if rng.random() < jump_prob:
            mid *= math.exp(rng.gauss(0, jump_bps / 1e4))
        bb = math.floor(mid / tick) * tick
        ba = bb + tick
        bids = [(round(bb - i * tick, 8), round(rng.uniform(0.05, 2.0), 3)) for i in range(10)]
        asks = [(round(ba + i * tick, 8), round(rng.uniform(0.05, 2.0), 3)) for i in range(10)]
        yield BookEvent(ts, bids, asks)
        if rng.random() < 0.3:
            side = Side.BUY if rng.random() < 0.5 else Side.SELL
            # Иногда крупная заявка «пробивает» несколько уровней.
            depth = int(rng.expovariate(0.7))
            price = asks[min(depth, 9)][0] if side is Side.BUY else bids[min(depth, 9)][0]
            yield TradeEvent(ts + 0.001, price, round(rng.expovariate(2.0), 3) + 0.001, side)


class CcxtFeed:
    """Стакан и сделки с биржи через WebSocket (ccxt.pro), с автопереподключением."""

    def __init__(self, exchange, symbol: str, depth: int, sink: Callable[[Event], None]) -> None:
        self.ex = exchange
        self.symbol = symbol
        self.depth = depth
        self.sink = sink
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        self._tasks = [asyncio.create_task(self._loop(self._books, "orderbook")),
                       asyncio.create_task(self._loop(self._trades, "trades"))]

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()

    async def _loop(self, fn, name: str) -> None:
        backoff = 1.0
        while True:
            try:
                await fn()
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                log.warning("Поток %s упал: %s, переподключение через %.0fс", name, e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def _books(self) -> None:
        while True:
            ob = await self.ex.watch_order_book(self.symbol, self.depth)
            ts = (ob.get("timestamp") or self.ex.milliseconds()) / 1000
            self.sink(BookEvent(ts,
                                [(float(p), float(s)) for p, s, *_ in ob["bids"][: self.depth]],
                                [(float(p), float(s)) for p, s, *_ in ob["asks"][: self.depth]]))

    async def _trades(self) -> None:
        while True:
            for t in await self.ex.watch_trades(self.symbol):
                side = Side(t["side"]) if t.get("side") in ("buy", "sell") else None
                self.sink(TradeEvent(t["timestamp"] / 1000, float(t["price"]), float(t["amount"]), side))
