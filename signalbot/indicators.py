"""15 классических стратегий-индикаторов. Каждая голосует за монету: +1 покупать, -1 продавать, 0 нейтрально.

Общий код для бота сигналов (signalbot) и исследования (research/confluence_study.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit

def ema(x, n):
    return x.ewm(span=n, adjust=False).mean()


def atr(h, l, c, n=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def sgn(cond_long, cond_short):
    return np.where(cond_long, 1, np.where(cond_short, -1, 0)).astype(np.int8)


def votes(df: pd.DataFrame, btc_close: pd.Series) -> pd.DataFrame:
    """Голоса 15 стратегий для одной монеты (часовые свечи)."""
    o, h, l, c, v = df.open, df.high, df.low, df.close, df.volume
    out = {}
    out["EMA 20/50"] = sgn(ema(c, 20) > ema(c, 50), ema(c, 20) < ema(c, 50))
    out["Цена vs EMA 200"] = sgn(c > ema(c, 200), c < ema(c, 200))
    macd = ema(c, 12) - ema(c, 26)
    hist = macd - ema(macd, 9)
    out["MACD"] = sgn(hist > 0, hist < 0)
    d = c.diff()
    rsi = 100 - 100 / (1 + d.clip(lower=0).ewm(alpha=1 / 14).mean() / (-d.clip(upper=0)).ewm(alpha=1 / 14).mean())
    out["RSI > 55 / < 45"] = sgn(rsi > 55, rsi < 45)
    hi48, lo48 = h.rolling(48).max(), l.rolling(48).min()
    pos = (c - lo48) / (hi48 - lo48)
    out["Канал Дончиана 48ч"] = sgn(pos > 0.8, pos < 0.2)
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    pb = (c - (mid - 2 * sd)) / (4 * sd)
    out["Боллинджер %B"] = sgn(pb > 0.8, pb < 0.2)
    up_m, dn_m = h.diff(), -l.diff()
    plus_dm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
    minus_dm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
    a14 = atr(h, l, c)
    pdi = 100 * pd.Series(plus_dm, index=c.index).ewm(alpha=1 / 14).mean() / a14
    mdi = 100 * pd.Series(minus_dm, index=c.index).ewm(alpha=1 / 14).mean() / a14
    adx = (100 * (pdi - mdi).abs() / (pdi + mdi)).ewm(alpha=1 / 14).mean()
    out["ADX > 25 + DI"] = sgn((adx > 25) & (pdi > mdi), (adx > 25) & (mdi > pdi))
    out["Supertrend 10/3"] = supertrend(h.to_numpy(), l.to_numpy(), c.to_numpy(), atr(h, l, c, 10).to_numpy(), 3.0)
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    top, bot = np.maximum(span_a, span_b), np.minimum(span_a, span_b)
    out["Ишимоку: облако"] = sgn(c > top, c < bot)
    ret24 = c / c.shift(24) - 1
    vol_up = v.rolling(24).mean() > v.rolling(24 * 30).mean()
    out["Объём + движение 24ч"] = sgn(vol_up & (ret24 > 0), vol_up & (ret24 < 0))
    rel = np.log(c / c.shift(168)) - np.log(btc_close / btc_close.shift(168))
    out["Сильнее/слабее BTC 7д"] = sgn(rel > 0, rel < 0)
    btc_up = btc_close > ema(btc_close, 720)
    out["Режим рынка (BTC vs EMA 30д)"] = sgn(btc_up, ~btc_up)
    k = 100 * (c - l.rolling(14).min()) / (h.rolling(14).max() - l.rolling(14).min())
    dk = k.rolling(3).mean()
    out["Стохастик"] = sgn((dk > 50) & (k > dk), (dk < 50) & (k < dk))
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    out["OBV растёт/падает"] = sgn(ema(obv, 12) > ema(obv, 48), ema(obv, 12) < ema(obv, 48))
    vwap = (c * v).rolling(24).sum() / v.rolling(24).sum()
    out["VWAP 24ч"] = sgn(c > vwap, c < vwap)
    return pd.DataFrame(out, index=c.index)


@njit(cache=True)
def supertrend(h, l, c, a, mult):
    n = len(c)
    out = np.zeros(n, dtype=np.int8)
    up_band, dn_band, trend = 0.0, 0.0, 1
    for i in range(n):
        if np.isnan(a[i]):
            continue
        mid = (h[i] + l[i]) / 2
        ub, lb = mid + mult * a[i], mid - mult * a[i]
        up_band = ub if (ub < up_band or c[i - 1] > up_band) else up_band
        dn_band = lb if (lb > dn_band or c[i - 1] < dn_band) else dn_band
        if trend == 1 and c[i] < dn_band:
            trend = -1
        elif trend == -1 and c[i] > up_band:
            trend = 1
        out[i] = trend
    return out
