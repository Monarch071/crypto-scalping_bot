"""Прогон бота сигналов по истории час за часом — тот же код, что работает вживую.

Нужен, чтобы убедиться: бот находит те же входы и выходы, что и исследование
(research/confluence_final.py), и считает результат так же.
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pandas as pd

from .bot import SignalBot
from .config import SignalConfig

HOUR = 3600


class HistoricalCandles:
    def __init__(self, frames: dict[str, pd.DataFrame], universe_size: int) -> None:
        self.frames = frames
        self.n = universe_size
        qv = pd.DataFrame({s: (f.close * f.volume) for s, f in frames.items()})
        self.vol24 = qv.rolling(24).sum()
        self.t = None

    async def universe(self, quote, n, exclude, btc):
        row = self.vol24.loc[:self.t].iloc[-1].dropna()
        top = [s for s in row.nlargest(n).index if s not in exclude]
        return top if btc in top else top + [btc]

    async def candles(self, symbols, hours, now=None):
        return {s: self.frames[s].loc[:self.t].iloc[-hours:] for s in symbols
                if s in self.frames and len(self.frames[s].loc[:self.t])}

    async def prices(self, symbols):
        return {s: float(self.frames[s].close.loc[:self.t].iloc[-1]) for s in symbols if s in self.frames}


def load_frames(panel_path: str) -> dict[str, pd.DataFrame]:
    panel = pd.read_parquet(panel_path)
    panel["timestamp"] = pd.to_datetime(panel.timestamp, utc=True)
    return {s.replace("USDT", "/USDT"): d.set_index("timestamp").sort_index()[["open", "high", "low", "close", "volume"]]
            for s, d in panel.groupby("symbol")}


def replay(frames: dict[str, pd.DataFrame], cfg: SignalConfig, start: str, end: str) -> SignalBot:
    md = HistoricalCandles(frames, cfg.universe_size)
    run_cfg = SignalConfig(**{**cfg.__dict__, "state_file": str(Path(tempfile.mkdtemp()) / "s.json")})
    bot = SignalBot(run_cfg, md, None)
    hours = frames["BTC/USDT"].loc[start:end].index

    async def go():
        for t in hours:
            md.t = t
            await bot.step(t.timestamp() + HOUR + 30)

    asyncio.run(go())
    return bot
