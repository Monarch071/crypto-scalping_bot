"""Прогон живого кода бота по истории час за часом («ускоренный paper»).

В отличие от rotator/backtest.py (быстрый векторный расчёт), здесь работает ровно тот же Trader,
что и на бирже: те же решения, план ордеров, полосы, бумажное исполнение с комиссией и проскальзыванием.
Если результаты прогона и бэктеста расходятся сильнее, чем объясняют издержки, — в коде бота ошибка.
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pandas as pd

from .config import RotationConfig
from .market import PaperAccount
from .trader import Trader

HOUR = 3600


class HistoricalMarket:
    """Рынок «как на бирже», но из истории. Свеча с меткой t закрывается в t + 1 ч."""

    def __init__(self, close: pd.DataFrame, qvol: pd.DataFrame) -> None:
        self.close = close
        self.vol24 = qvol.rolling(24).sum()
        self.t = close.index[0]

    async def universe(self, quote, n, exclude, btc):
        top = [s for s in self.vol24.loc[self.t].nlargest(n).index if s not in exclude]
        return top if btc in top else top + [btc]

    async def closes(self, symbols, hours, now=None):
        return self.close.loc[:self.t, [s for s in symbols if s in self.close]].iloc[-hours:]

    async def prices(self, symbols):
        row = self.close.loc[self.t]
        return {s: float(row[s]) for s in symbols if s in row and pd.notna(row[s])}


def replay(close: pd.DataFrame, qvol: pd.DataFrame, cfg: RotationConfig, start: str, end: str | None = None,
           slippage_bps: float = 5.0) -> tuple[pd.Series, Trader]:
    hours = close.loc[start:end].index
    md = HistoricalMarket(close, qvol)
    state = Path(tempfile.mkdtemp()) / "state.json"
    run_cfg = RotationConfig(**{**cfg.__dict__, "state_file": str(state), "mode": "paper"})
    trader = Trader(run_cfg, md, None)
    trader.account = PaperAccount(trader.state.setdefault("paper_balances", {cfg.quote: cfg.paper_start_usdt}),
                                  cfg.fee_bps, slippage_bps)
    equity = {}

    async def go():
        for t in hours:
            md.t = t
            await trader.step(t.timestamp() + HOUR + 60)  # через минуту после закрытия свечи t
            equity[t] = trader.state["equity"]

    asyncio.run(go())
    return pd.Series(equity), trader
