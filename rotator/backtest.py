"""Бэктест стратегии ротации на тех же функциях, что и живой бот.

Вход: parquet с часовыми свечами многих монет (колонки symbol, timestamp, close, volume),
например из research/download_binance.py --interval 1h --panel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RotationConfig
from .strategy import BTC, btc_uptrend, eligible, momentum_vs_btc, target_weights

HPY = 365 * 24


def load_panel(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["symbol"] = df["symbol"].str.replace(r"^(.*)USDT$", r"\1/USDT", regex=True)
    close = df.pivot_table(index="timestamp", columns="symbol", values="close")
    qvol = df.assign(q=df.close * df.volume).pivot_table(index="timestamp", columns="symbol", values="q")
    idx = pd.date_range(close.index.min(), close.index.max(), freq="h")
    return close.reindex(idx).ffill(limit=3), qvol.reindex(idx).fillna(0.0)


def run(close: pd.DataFrame, qvol: pd.DataFrame, cfg: RotationConfig, cost_bps: float | None = None):
    """Возвращает почасовую доходность портфеля и веса."""
    cost = cfg.fee_bps if cost_bps is None else cost_bps
    scores = momentum_vs_btc(close, cfg.lookback_h)
    ok = eligible(close, cfg.min_history_h)
    up = btc_uptrend(close[BTC], cfg.trend_ema_h)
    vol24 = qvol.rolling(24).sum()
    times = close.index[close.index.hour % cfg.rebalance_h == 0]
    # Пустая цель (вне рынка) — это строка нулей, а не пропуск: иначе ffill держал бы старые позиции.
    w = pd.DataFrame(0.0, index=times, columns=close.columns)
    for t in times:
        top = vol24.loc[t].nlargest(cfg.universe_size).index
        in_universe = ok.loc[t] & ok.columns.isin(list(top) + [BTC])
        for sym, x in target_weights(scores.loc[t], in_universe, bool(up.loc[t]), cfg.top_k,
                                     frozenset(cfg.exclude)).items():
            w.at[t, sym] = x
    w = (w * cfg.capital_fraction).reindex(close.index).ffill().fillna(0.0)
    r = close.pct_change().fillna(0.0)
    turnover = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    ret = (w.shift(1).fillna(0.0) * r).sum(axis=1) - turnover.shift(1).fillna(0.0) * cost / 1e4
    return ret, w


def report(ret: pd.Series, w: pd.DataFrame) -> str:
    lines = ["| Период | Доходность | Шарп | Макс. просадка | Прибыльных месяцев | В рынке |", "|---|---|---|---|---|---|"]
    years = sorted(set(ret.index.year))
    for label, mask in [(str(y), ret.index.year == y) for y in years] + [("Всё время", slice(None))]:
        r = ret[mask]
        eq = (1 + r).cumprod()
        daily = (1 + r).groupby(r.index.normalize()).prod() - 1
        monthly = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
        yrs = len(r) / HPY
        total = (eq.iloc[-1] ** (1 / yrs) - 1) * 100 if label == "Всё время" else (eq.iloc[-1] - 1) * 100
        sharpe = daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else 0.0
        lines.append(f"| {label} | {total:+.1f}%{'/год' if label == 'Всё время' else ''} | {sharpe:.2f} | "
                     f"{(1 - eq / eq.cummax()).max() * 100:.0f}% | {(monthly > 0).mean() * 100:.0f}% | "
                     f"{(w[mask].sum(axis=1) > 0).mean() * 100:.0f}% |")
    return "\n".join(lines)
