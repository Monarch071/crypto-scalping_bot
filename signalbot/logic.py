"""Логика сигналов без сети: поиск входа, уровни стоп/тейк, размер позиции, итог сделки, статистика."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import SignalConfig
from .indicators import atr, votes

STRATEGY_COUNT = 15


@dataclass
class Signal:
    id: str
    symbol: str
    side: str            # long | short
    time: str            # ISO, закрытие часа сигнала
    entry: float
    sl: float
    tp: float
    expires: str         # ISO
    size: float          # количество монет при заданном риске
    risk_usdt: float
    agree: list[str]
    status: str = "open"  # open | tp | sl | time
    exit_price: float | None = None
    exit_time: str | None = None
    r: float | None = None
    net_bps: float | None = None

    @property
    def sign(self) -> int:
        return 1 if self.side == "long" else -1


def vote_table(df: pd.DataFrame, btc_close: pd.Series) -> pd.DataFrame:
    """Голоса на сетке часов BTC: у BTC история длиннее, и EMA за 30 дней считается без искажений."""
    grid = btc_close.index.union(df.index)
    return votes(df.reindex(grid), btc_close.reindex(grid).ffill()).reindex(df.index)


def find_signal(symbol: str, df: pd.DataFrame, btc_close: pd.Series, cfg: SignalConfig,
                allowed: bool = True) -> Signal | None:
    """Сигнал на последней закрытой свече: голосов «за» стало не меньше min_votes (а часом раньше было меньше)."""
    if len(df) < 300 or not allowed:
        return None
    V = vote_table(df, btc_close)
    a = atr(df.high, df.low, df.close).iloc[-1]
    if np.isnan(a) or a <= 0:
        return None
    last, prev = V.iloc[-1], V.iloc[-2]
    for side, s in (("long", 1), ("short", -1)):
        if side not in cfg.sides:
            continue
        now_n, prev_n = int((last == s).sum()), int((prev == s).sum())
        if now_n >= cfg.min_votes > prev_n:
            entry = float(df.close.iloc[-1])
            sl = entry - s * cfg.sl_atr * a
            tp = entry + s * cfg.tp_atr * a
            risk = cfg.deposit_usdt * cfg.risk_pct / 100
            t = df.index[-1]
            return Signal(id=f"{symbol}-{t:%Y%m%d%H}-{side}", symbol=symbol, side=side, time=t.isoformat(),
                          entry=entry, sl=float(sl), tp=float(tp),
                          expires=(t + pd.Timedelta(hours=cfg.max_hold_h)).isoformat(),
                          size=risk / abs(entry - sl), risk_usdt=risk,
                          agree=[n for n in V.columns if last[n] == s])
    return None


def check_exit(sig: Signal, high: float, low: float, close: float, now: pd.Timestamp) -> tuple[str, float] | None:
    """Сработал ли выход. Если в одном баре задеты и стоп, и тейк, считаем стоп (пессимистично)."""
    s = sig.sign
    if (low <= sig.sl) if s > 0 else (high >= sig.sl):
        return "sl", sig.sl
    if (high >= sig.tp) if s > 0 else (low <= sig.tp):
        return "tp", sig.tp
    if now >= pd.Timestamp(sig.expires):
        return "time", close
    return None


def close_signal(sig: Signal, kind: str, price: float, when: pd.Timestamp, cfg: SignalConfig) -> Signal:
    cost = cfg.entry_cost_bps + {"tp": cfg.tp_cost_bps, "sl": cfg.sl_cost_bps, "time": cfg.time_exit_cost_bps}[kind]
    gross = sig.sign * (price / sig.entry - 1) * 1e4
    sig.status, sig.exit_price, sig.exit_time = kind, price, when.isoformat()
    sig.net_bps = gross - cost
    sig.r = sig.net_bps / (abs(sig.entry - sig.sl) / sig.entry * 1e4)
    return sig


def stats(closed: list[Signal]) -> str:
    if not closed:
        return "Закрытых сигналов пока нет."
    r = np.array([s.r for s in closed])
    t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 and r.std() > 0 else 0.0
    kinds = pd.Series([s.status for s in closed]).value_counts().to_dict()
    eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(eq) - eq).max())
    return (f"Закрыто сигналов: {len(r)} (тейк {kinds.get('tp', 0)}, стоп {kinds.get('sl', 0)}, "
            f"по времени {kinds.get('time', 0)})\n"
            f"Прибыльных: {(r > 0).mean():.0%}\nСредний результат: {r.mean():+.2f}R "
            f"({np.mean([s.net_bps for s in closed]):+.1f} б.п.)\n"
            f"Сумма: {r.sum():+.1f}R — при риске 1% это {r.sum():+.1f}% депозита\n"
            f"Макс. просадка: {dd:.1f}R\nt-статистика: {t:+.1f} "
            f"({'уже значимо' if t > 3 else 'пока не доказано, нужно t > 3'})")


def fmt_price(x: float) -> str:
    return f"{x:.6g}"


def format_signal(sig: Signal, rank: int | None = None) -> str:
    arrow = "🟢 ЛОНГ (покупка)" if sig.side == "long" else "🔴 ШОРТ (продажа)"
    pct = lambda p: (p / sig.entry - 1) * 100  # noqa: E731
    return (f"{arrow} {sig.symbol}\n"
            f"Вход: ~{fmt_price(sig.entry)}\n"
            f"Стоп-лосс: {fmt_price(sig.sl)} ({pct(sig.sl):+.2f}%)\n"
            f"Тейк-профит: {fmt_price(sig.tp)} ({pct(sig.tp):+.2f}%)\n"
            f"Закрыть, если не сработало до: {pd.Timestamp(sig.expires):%d.%m %H:%M} UTC\n"
            f"Размер: {sig.size:.6g} {sig.symbol.split('/')[0]} (≈{sig.size * sig.entry:.2f} USDT, "
            f"риск {sig.risk_usdt:.2f} USDT)\n"
            f"За: {len(sig.agree)} из {STRATEGY_COUNT}"
            + (f", моментум отн. BTC: #{rank}" if rank else "") +
            "\nСтоп и тейк ставьте сразу при открытии позиции.")


def to_dict(sig: Signal) -> dict:
    return asdict(sig)


def from_dict(d: dict) -> Signal:
    return Signal(**d)
