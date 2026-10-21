"""Стоп по ATR против фиксированного стопа 3% / 5% / 10% для итоговой стратегии совпадения.

  python research/stop_comparison.py PANEL.parquet
"""
import sys, numpy as np, pandas as pd
from numba import njit
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import confluence_v2 as V2
from rotator.strategy import eligible, momentum_vs_btc
from signalbot.indicators import atr, votes

@njit
def sim(H, L, C, cnt, entries, stop_d, max_hold, exit_votes):
    n = len(entries); g = np.empty(n); k = np.empty(n, np.int8); xi = np.empty(n, np.int64); last = len(C) - 1
    for i in range(n):
        e = entries[i]; p0 = C[e]; stop = p0 - stop_d[i]; end = min(e + max_hold, last)
        kk, px, x = 0, C[end], end
        for j in range(e + 1, end + 1):
            if L[j] <= stop:
                kk, px, x = -1, stop, j; break
            if cnt[j] < exit_votes:
                kk, px, x = 2, C[j], j; break
        g[i] = (px / p0 - 1) * 1e4; k[i] = kk; xi[i] = x
    return g, k, xi

panel = pd.read_parquet(sys.argv[1]); panel["timestamp"] = pd.to_datetime(panel.timestamp, utc=True)
fr = {s.replace("USDT", "/USDT"): d.set_index("timestamp").sort_index() for s, d in panel.groupby("symbol")}
idx = pd.date_range(pd.Timestamp("2020-01-01", tz="UTC"), panel.timestamp.max(), freq="h")
close = pd.DataFrame({s: f.close for s, f in fr.items()}).reindex(idx).ffill(limit=3)
rank = momentum_vs_btc(close, 168).where(eligible(close, 1440)).rank(axis=1, ascending=False)
btc = close["BTC/USDT"]
data = {}
for sym, df in fr.items():
    df = df.reindex(idx).ffill(limit=3).dropna()
    Vt = votes(df, btc.reindex(df.index)).iloc[24 * 35:]; d = df.reindex(Vt.index)
    A = atr(df.high, df.low, df.close).reindex(Vt.index).to_numpy(); cnt = (Vt.to_numpy() > 0).sum(1)
    on = cnt >= 15; e = np.where(on & ~np.r_[False, on[:-1]] & ~np.isnan(A))[0]; e = e[e < len(cnt) - 2]
    e = e[rank[sym].reindex(Vt.index).to_numpy()[e] <= 3]
    data[sym] = (d.high.to_numpy(), d.low.to_numpy(), d.close.to_numpy(), cnt, e, A, Vt.index)
for name, fn in (("2 ATR (сейчас)", lambda C, A, e: 2 * A[e]), ("фикс. 3%", lambda C, A, e: 0.03 * C[e]),
                 ("фикс. 5%", lambda C, A, e: 0.05 * C[e]), ("фикс. 10%", lambda C, A, e: 0.10 * C[e])):
    parts = []
    for sym, (H, L, C, cnt, e, A, ix) in data.items():
        sd = fn(C, A, e)
        g, k, xi = sim(H, L, C, cnt, e, sd, 72, 10)
        cost = V2.ENTRY_BPS + np.where(k == -1, V2.STOP_BPS, 15.5)
        stop_pct = sd / C[e] * 100
        parts.append(pd.DataFrame({"sym": sym, "time": ix[e], "exit_time": ix[xi], "R": (g - cost) / (stop_pct * 100),
                                   "stop_pct": stop_pct, "k": k}))
    T = V2.one_per_coin(pd.concat(parts, ignore_index=True))
    y = T.groupby(T.time.dt.year).R.sum()
    eq = T.sort_values("exit_time").R.cumsum(); dd = (eq.cummax() - eq).max()
    print(f"{name:15s}: стоп ~{T.stop_pct.median():4.1f}% | покупка ~${100/T.stop_pct.median():4.0f} при риске $1 | "
          f"стопом закрыто {np.mean(T.k==-1):3.0%} | сделка {T.R.mean():+.3f}R | {y.mean():+5.1f}%/год | "
          f"просадка {dd:3.0f}% | лет в плюсе {int((y>0).sum())}/7")
