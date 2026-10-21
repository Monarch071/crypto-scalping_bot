"""Стратегия ротации: держим монеты с самым сильным моментумом относительно BTC,
но только пока у самого BTC восходящий тренд.

Чистые функции без сети и состояния: одни и те же вызываются в бэктесте и в живом боте.
Формулы совпадают с research/xs_study.py (сигнал «Моментум отн. BTC», режим «long + тренд BTC»).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BTC = "BTC/USDT"
BETA_WINDOW_H = 24 * 30
BETA_MIN_H = 24 * 10


def momentum_vs_btc(close: pd.DataFrame, lookback_h: int, btc: str = BTC) -> pd.DataFrame:
    """Доходность монеты за lookback_h часов минус та часть, которую объясняет движение BTC (бета)."""
    lr = np.log(close).diff()
    b = lr[btc]
    beta = lr.rolling(BETA_WINDOW_H, min_periods=BETA_MIN_H).cov(b).div(
        b.rolling(BETA_WINDOW_H, min_periods=BETA_MIN_H).var(), axis=0)
    mom = np.log(close / close.shift(lookback_h))
    return mom - beta.mul(mom[btc], axis=0)


def btc_uptrend(btc_close: pd.Series, ema_h: int) -> pd.Series:
    return btc_close > btc_close.ewm(span=ema_h, adjust=False).mean()


def eligible(close: pd.DataFrame, min_history_h: int) -> pd.DataFrame:
    """Монета участвует в отборе, только когда у неё достаточно истории."""
    return close.notna() & (close.notna().cumsum() > min_history_h)


def target_weights(scores: pd.Series, ok: pd.Series, uptrend: bool, top_k: int,
                   exclude: frozenset[str] = frozenset()) -> dict[str, float]:
    """Равные доли в top_k монетах с лучшим счётом; пусто, если BTC не в тренде или монет мало."""
    if not uptrend:
        return {}
    s = scores[ok.reindex(scores.index, fill_value=False)].dropna()
    s = s[[sym for sym in s.index if sym not in exclude]]
    if len(s) < 2 * top_k:
        return {}
    best = s.sort_values(ascending=False, kind="stable").index[:top_k]
    return {sym: 1.0 / top_k for sym in best}
