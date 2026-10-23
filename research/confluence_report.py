"""Итог по стратегии «все 15 согласны»: деньги на счёте при риске 1% на сделку.

  python research/confluence_report.py PANEL.parquet --out research/confluence_out

Берёт сигналы 15/15 (лонг, стоп 2 ATR, тейк 4 ATR, максимум 24 ч) и считает результат в «R»:
1R = расстояние до стопа. При риске 1% депозита на сделку прибыль в R = прибыль в % депозита.
Сравниваются два вида комиссий:
  * спот Bybit: 0.1% вход + 0.1% выход;
  * фьючерсы Bybit: вход рыночный 0.055%, тейк-профит лимитный 0.02%, стоп рыночный 0.055%.
К каждой сделке добавлено 10 б.п. на ручной вход и 5 б.п. проскальзывания на стопе.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import confluence_study as S
from ml_charts import BLUE, INK, ORANGE, SURFACE

KS, KT = 2.0, 4.0


def signals(panel_path):
    panel = pd.read_parquet(panel_path)
    panel["timestamp"] = pd.to_datetime(panel.timestamp, utc=True)
    fr = {s: d.set_index("timestamp").sort_index() for s, d in panel.groupby("symbol")}
    idx = pd.date_range(pd.Timestamp("2020-01-01", tz="UTC"), panel.timestamp.max(), freq="h")
    btc = fr["BTCUSDT"].close.reindex(idx).ffill()
    rows = []
    for sym, df in fr.items():
        df = df.reindex(idx).ffill(limit=3).dropna()
        V = S.votes(df, btc.reindex(df.index)).iloc[24 * 35:]
        H, L, C = (df[k].reindex(V.index).to_numpy() for k in ("high", "low", "close"))
        a = S.atr(df.high, df.low, df.close).reindex(V.index).to_numpy()
        e = np.where(S.first_cross((V.to_numpy() > 0).sum(1), 15) & ~np.isnan(a))[0]
        e = e[e < len(C) - 2]
        g, kind, hold = S.simulate(H, L, C, e, np.ones(len(e), np.int64), KS * a[e], KT * a[e], S.MAX_HOLD_H)
        rows.append(pd.DataFrame({"sym": sym, "time": V.index[e], "gross": g, "exit": kind, "hold_h": hold,
                                  "stop_bps": KS * a[e] / C[e] * 1e4}))
    return pd.concat(rows).sort_values("time", ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panel")
    ap.add_argument("--out", default="research/confluence_out")
    args = ap.parse_args()
    T = signals(args.panel)
    exit_fee_fut = np.where(T.exit == 1, 2.0, 5.5 + np.where(T.exit == -1, 5.0, 0.0))
    T["net_spot"] = T.gross - 20.0 - 10.0 - np.where(T.exit == -1, 5.0, 0.0)
    T["net_fut"] = T.gross - 5.5 - 10.0 - exit_fee_fut
    lines = ["| Комиссии | Период | Сделок/год | Прибыльных | Средняя сделка | В R | t | % депозита/год (риск 1%) | "
             "Макс. просадка | Прибыльных месяцев |", "|---|---|---|---|---|---|---|---|---|---|"]
    curves = {}
    for col, name in (("net_spot", "спот 0.1%"), ("net_fut", "фьючерсы")):
        T["R"] = T[col] / T.stop_bps
        for per, m in (("2020–2025", T.time.dt.year < 2026), ("2026", T.time.dt.year == 2026),
                       ("всё", slice(None))):
            x = T[m]
            yrs = (x.time.max() - x.time.min()).days / 365
            eq = x.set_index("time").R.cumsum()
            monthly = x.groupby([x.time.dt.year, x.time.dt.month]).R.sum()
            lines.append(f"| {name} | {per} | {len(x) / yrs:.0f} | {(x[col] > 0).mean() * 100:.0f}% | "
                         f"{x[col].mean():+.1f} б.п. | {x.R.mean():+.3f}R | "
                         f"{x[col].mean() / x[col].std() * np.sqrt(len(x)):+.1f} | {x.R.sum() / yrs:+.0f}% | "
                         f"{(eq.cummax() - eq).max():.0f}% | {(monthly > 0).mean() * 100:.0f}% |")
            if per == "всё":
                curves[name] = eq
        by_year = T.groupby(T.time.dt.year).R.sum().round(0)
        lines.append(f"| {name} | по годам, % депозита | " + ", ".join(f"{y}: {v:+.0f}" for y, v in by_year.items())
                     + " ||||||||")
    print("\n".join(lines))
    print(f"\nМедиана удержания: {T.hold_h.median():.0f} ч; выход по тейку {(T.exit == 1).mean():.0%}, "
          f"по стопу {(T.exit == -1).mean():.0%}, по времени {(T.exit == 0).mean():.0%}; "
          f"медианный стоп {T.stop_bps.median():.0f} б.п.")
    out = Path(args.out)
    (out / "REPORT_TABLE.md").write_text("\n".join(lines), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 5))
    for (name, eq), col in zip(curves.items(), (ORANGE, BLUE)):
        ax.plot(eq.index, eq.values, color=col, lw=2)
        ax.annotate(f"{name}: {eq.iloc[-1]:+.0f}%", (eq.index[-1], eq.iloc[-1]), xytext=(-8, 6),
                    textcoords="offset points", ha="right", color=INK, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec=col, lw=1))
    ax.axhline(0, color=INK, lw=1)
    ax.axvline(pd.Timestamp("2026-01-01", tz="UTC"), color=INK, lw=1, ls=":")
    ax.set_ylabel("Накопленный результат, % депозита (риск 1% на сделку)")
    ax.set_title("«Все 15 стратегий за»: лонг, стоп 2 ATR, тейк 4 ATR, 10 монет")
    fig.tight_layout()
    fig.savefig("img/confluence_equity.png", dpi=130)


if __name__ == "__main__":
    main()
