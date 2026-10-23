"""Графики к ML-исследованию (PNG). Использует кеш прогнозов из ml_signal.py.

  python research/ml_charts.py HIST.csv.gz LATEST.csv --out research/ml_out --img research/img
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml_signal import trade
from ml_study import bars, load, target

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
})


def decile_chart(pred, fwd, mask, path):
    d = pd.DataFrame({"p": pred[mask], "y": fwd[mask] * 1e4})
    d["dec"] = pd.qcut(d.p, 10, labels=False) + 1
    g = d.groupby("dec").y
    mean, err = g.mean(), 2 * g.std() / np.sqrt(g.count())
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(mean.index, mean.values, width=0.7, color=BLUE, yerr=err.values,
           error_kw=dict(ecolor=INK2, lw=1, capsize=3), zorder=3)
    ax.axhline(0, color=INK2, lw=1)
    for lvl, lab in ((4, "круг мейкером: 4 б.п."), (13, "круг тейкером: 13 б.п.")):
        ax.axhline(lvl, color=ORANGE, lw=2, ls="--")
        ax.text(0.6, lvl + 0.3, lab, color=INK, fontsize=10)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Дециль прогноза нейросети (1 = сильнее всего «вниз», 10 = «вверх»)")
    ax.set_ylabel("Фактическая доходность за 15 мин, б.п.")
    ax.set_title("Сигнал есть, но он в разы меньше комиссии (BTC, 15 мин, 2022–2026, тест)")
    ax.set_ylim(-2, 15)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def equity_chart(pred, tested, b, c1, horizon, path):
    exec_px = c1[np.minimum(b["end"].to_numpy() + 1, len(c1) - 1)]
    abs_p = pd.Series(np.abs(pred), index=b.index)
    win = int(90 * 24 * 60 / 15)
    thr = abs_p.rolling(win, min_periods=win // 3).quantile(0.90).shift(1).to_numpy().copy()
    thr[~tested] = np.nan
    fig, ax = plt.subplots(figsize=(9, 5))
    for cost, color, lab in ((0.0, BLUE, "без комиссий"), (2.0, AQUA, "мейкер 2 б.п."),
                             (6.5, ORANGE, "тейкер 6.5 б.п.")):
        _, rets, trades = trade(pred, exec_px, horizon, thr, cost)
        idx = pd.DatetimeIndex([b.index[j] for _, j, _ in trades])
        eq = pd.Series(rets, index=idx).cumsum() * 100
        ax.plot(eq.index, eq.values, color=color, lw=2)
        ax.annotate(f"{lab}: {eq.iloc[-1]:+.0f}%", (eq.index[-1], eq.iloc[-1]), xytext=(-8, 6),
                    textcoords="offset points", ha="right", color=INK, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec=color, lw=1))
    ax.axhline(0, color=INK2, lw=1)
    ax.set_ylabel("Накопленная доходность, % капитала")
    ax.set_title("Одна и та же модель: всё решают комиссии (BTC, 15 мин, топ-10% сигналов)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="research/ml_out")
    ap.add_argument("--img", default="research/img")
    args = ap.parse_args()
    img = Path(args.img)
    img.mkdir(parents=True, exist_ok=True)
    m = load(args.paths)
    c1 = m["close"].to_numpy()
    b = bars(m, 15)
    y, _ = target(b, c1, 1)
    z = np.load(Path(args.out) / "pred_tf15_h1.npz")
    pred, tested = z["pred"], z["tested"]
    mask = tested & ~np.isnan(pred) & ~np.isnan(y.to_numpy())
    decile_chart(pred, y.to_numpy(), mask, img / "deciles_15m.png")
    equity_chart(pred, tested, b, c1, 1, img / "equity_15m_costs.png")


if __name__ == "__main__":
    main()
