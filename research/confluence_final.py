"""Итоговая версия стратегии совпадения: вход 15/15 в монете из топ-3 по силе отн. BTC,
стоп 2 ATR, выход, когда «за» меньше 10 из 15, или через 72 ч.

  python research/confluence_final.py PANEL.parquet

Параметры выбраны из середины устойчивого «плато» (пороги 8–11, удержание 48–96 ч — все прибыльны),
см. research/CONFLUENCE_REPORT.md. Издержки: вход 15.5 б.п., выход по сигналу или времени 15.5 б.п.
(рыночный + ручная задержка), стоп 10.5 б.п.
"""
from __future__ import annotations

import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import confluence_v2 as V2
from ml_charts import BLUE, INK, ORANGE, SURFACE
from rotator.strategy import eligible, momentum_vs_btc
from signalbot.indicators import atr, votes

EXIT_VOTES, MAX_HOLD, TOP_K = 10, 72, 3


def trades(panel_path, exit_mode=1, top_k=TOP_K):
    panel = pd.read_parquet(panel_path)
    panel["timestamp"] = pd.to_datetime(panel.timestamp, utc=True)
    fr = {s.replace("USDT", "/USDT"): d.set_index("timestamp").sort_index() for s, d in panel.groupby("symbol")}
    idx = pd.date_range(pd.Timestamp("2020-01-01", tz="UTC"), panel.timestamp.max(), freq="h")
    close = pd.DataFrame({s: f.close for s, f in fr.items()}).reindex(idx).ffill(limit=3)
    rank = momentum_vs_btc(close, 168).where(eligible(close, 1440)).rank(axis=1, ascending=False)
    btc = close["BTC/USDT"]
    parts = []
    for sym, df in fr.items():
        df = df.reindex(idx).ffill(limit=3).dropna()
        Vt = votes(df, btc.reindex(df.index)).iloc[24 * 35:]
        d = df.reindex(Vt.index)
        A = atr(df.high, df.low, df.close).reindex(Vt.index).to_numpy()
        cnt = (Vt.to_numpy() > 0).sum(1)
        on = cnt >= 15
        e = np.where(on & ~np.r_[False, on[:-1]] & ~np.isnan(A))[0]
        e = e[e < len(cnt) - 2]
        rk = rank[sym].reindex(Vt.index).to_numpy()[e]
        if top_k:
            e = e[rk <= top_k]
        hold = MAX_HOLD if exit_mode == 1 else 24
        g, k, xi = V2.simulate(d.high.to_numpy(), d.low.to_numpy(), d.close.to_numpy(), A, cnt, e,
                               exit_mode, hold, EXIT_VOTES)
        cost = V2.ENTRY_BPS + np.where(k == -1, V2.STOP_BPS, np.where(k == 1, V2.TP_BPS, 15.5))
        parts.append(pd.DataFrame({"sym": sym, "time": Vt.index[e], "exit_time": Vt.index[xi], "kind": k,
                                   "R": (g - cost) / (2 * A[e] / d.close.to_numpy()[e] * 1e4), "hold": xi - e}))
    return V2.one_per_coin(pd.concat(parts, ignore_index=True)).sort_values("time")


def account(T):
    """Счёт при риске 1% депозита на сделку: результат сделки записывается в момент выхода."""
    pnl = T.set_index("exit_time").R.sort_index()
    daily = pnl.groupby(pnl.index.normalize()).sum()
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"), fill_value=0)
    return daily.cumsum()


def main():
    T = trades(sys.argv[1])
    base = trades(sys.argv[1], exit_mode=0, top_k=0)
    eq, eq0 = account(T), account(base)
    ev = sorted([(t, 1) for t in T.time] + [(t, -1) for t in T.exit_time])
    conc, mx = 0, 0
    for _, d in ev:
        conc += d
        mx = max(mx, conc)
    monthly = T.groupby([T.exit_time.dt.year, T.exit_time.dt.month]).R.sum()
    yearly = T.groupby(T.exit_time.dt.year).R.sum()
    yrs = (T.time.max() - T.time.min()).days / 365
    t = T.R.mean() / T.R.std() * np.sqrt(len(T))
    print(f"Сделок: {len(T)} ({len(T) / yrs:.0f}/год), прибыльных {(T.R > 0).mean():.0%}, "
          f"средняя {T.R.mean():+.3f}R, t = {t:+.1f}")
    print(f"Выходы: по сигналу {(T.kind == 2).mean():.0%}, стоп {(T.kind == -1).mean():.0%}, "
          f"по времени {(T.kind == 0).mean():.0%}; удержание медиана {T.hold.median():.0f} ч, "
          f"75% сделок до {T.hold.quantile(0.75):.0f} ч")
    print(f"Средняя прибыльная {T.R[T.R > 0].mean():+.2f}R, средняя убыточная {T.R[T.R <= 0].mean():+.2f}R")
    print(f"При риске 1%: {yearly.mean():+.0f}% депозита в год в среднем, макс. просадка {(eq.cummax() - eq).max():.0f}%, "
          f"прибыльных месяцев {(monthly > 0).mean():.0%}, одновременно открыто до {mx} позиций")
    print("По годам, % депозита:", {y: round(v, 1) for y, v in yearly.items()})
    fig, ax = plt.subplots(figsize=(9, 5))
    for s, col, lab, dy in ((eq0, ORANGE, "было: все 15 за, стоп 2 / тейк 4 ATR", -16),
                            (eq, BLUE, "стало: + топ-3 по силе, выход по сигналу", 6)):
        ax.plot(s.index, s.values, color=col, lw=2)
        ax.annotate(f"{lab}: {s.iloc[-1]:+.0f}%", (s.index[-1], s.iloc[-1]), xytext=(-8, dy),
                    textcoords="offset points", ha="right", color=INK, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec=col, lw=1))
    ax.axhline(0, color=INK, lw=1)
    ax.axvline(pd.Timestamp("2026-01-01", tz="UTC"), color=INK, lw=1, ls=":")
    ax.set_ylabel("Накопленный результат, % депозита (риск 1% на сделку)")
    ax.set_title("Стратегия совпадения: до и после улучшения выхода (10 монет, 2020–2026)")
    fig.tight_layout()
    fig.savefig("img/confluence_final.png", dpi=130)


if __name__ == "__main__":
    main()
