"""Бэктест: события из файла (или синтетики) прогоняются через тот же движок, что и в live."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .config import Config
from .engine import Engine
from .events import BookEvent, Event
from .gateways.paper import PaperGateway


@dataclass
class Report:
    duration_h: float
    fills: int
    volume: float
    pnl: float
    realized: float
    fees: float
    final_position: float
    max_abs_position: float
    max_drawdown: float
    sharpe_hourly: float
    killed: str

    def __str__(self) -> str:
        return (
            f"Длительность:           {self.duration_h:.2f} ч\n"
            f"Исполнений:             {self.fills}\n"
            f"Оборот:                 {self.volume:,.2f}\n"
            f"Чистый PnL:             {self.pnl:.4f}\n"
            f"  реализованный:        {self.realized:.4f}\n"
            f"  комиссии:             {self.fees:.4f}\n"
            f"Итоговая позиция:       {self.final_position:.6f}\n"
            f"Макс. позиция:          {self.max_abs_position:.6f}\n"
            f"Макс. просадка:         {self.max_drawdown:.4f}\n"
            f"Шарп (по часам):        {self.sharpe_hourly:.2f}\n"
            f"Kill switch:            {self.killed or 'нет'}"
        )


async def run_backtest(cfg: Config, events: Iterable[Event]) -> Report:
    gw = PaperGateway(cfg.exchange, cfg.paper.latency_ms)
    engine = Engine(cfg, gw, realtime=False)
    await engine.start()

    first_ts = last_ts = None
    peak, max_dd, max_pos = 0.0, 0.0, 0.0
    hourly: list[float] = []
    next_hour = None
    for ev in events:
        engine.emit(ev)
        await engine.drain()
        if not isinstance(ev, BookEvent) or not engine.book.ready:
            continue
        first_ts = ev.ts if first_ts is None else first_ts
        last_ts = ev.ts
        pnl = engine.portfolio.total_pnl(engine.book.mid)
        peak = max(peak, pnl)
        max_dd = max(max_dd, peak - pnl)
        max_pos = max(max_pos, abs(engine.portfolio.position))
        if next_hour is None:
            next_hour = ev.ts + 3600
        if ev.ts >= next_hour:
            hourly.append(pnl)
            next_hour += 3600

    p = engine.portfolio
    mark = engine.book.mid if engine.book.ready else 0.0
    rets = [b - a for a, b in zip([0.0] + hourly, hourly)]
    sharpe = 0.0
    if len(rets) > 1:
        mean = sum(rets) / len(rets)
        sd = math.sqrt(sum((r - mean) ** 2 for r in rets) / (len(rets) - 1))
        sharpe = mean / sd if sd > 0 else 0.0
    return Report(
        duration_h=((last_ts or 0) - (first_ts or 0)) / 3600,
        fills=p.fills, volume=p.volume, pnl=p.total_pnl(mark), realized=p.realized,
        fees=p.fees, final_position=p.position, max_abs_position=max_pos,
        max_drawdown=max_dd, sharpe_hourly=sharpe, killed=engine.risk.kill_reason,
    )
