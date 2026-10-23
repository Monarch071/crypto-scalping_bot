"""График и статистика удержания для бота, выбирающего монеты (по результатам xs_study.py).

  python research/plot_xs.py PANEL.parquet --out research/xs_out
"""
import argparse

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

from ml_charts import AQUA, BLUE, INK, ORANGE, SURFACE
from xs_study import backtest, features, load_panel, rule_signals, weights

ap = argparse.ArgumentParser()
ap.add_argument("panel")
ap.add_argument("--out", default="research/xs_out")
args = ap.parse_args()

close, volume, alive = load_panel(args.panel)
btc = close["BTCUSDT"]
btc_up = btc > btc.ewm(span=24 * 30, adjust=False).mean()
w = weights(rule_signals(features(close, volume))["Моментум отн. BTC 168ч"], alive, 8, 2, "long + тренд BTC", btc_up)
held = (w > 0)
runs = []
for c in held.columns:
    s = held[c].astype(int)
    starts = s.diff().fillna(s) == 1
    runs += list(s.groupby((starts).cumsum()).sum()[lambda x: x > 0].values)
runs = np.array(runs)
print(f"Пример конфигурации (моментум отн. BTC 168ч, пересмотр 8ч, 2 монеты, фильтр BTC): "
      f"удержание монеты медиана {np.median(runs):.0f} ч, среднее {runs.mean():.0f} ч, "
      f"25–75%: {np.percentile(runs, 25):.0f}–{np.percentile(runs, 75):.0f} ч; в рынке {(w.sum(axis=1) > 0).mean() * 100:.0f}% времени")

wf = pd.read_parquet(f"{args.out}/wf_long+трендBTC.parquet")["ret"]
ew = alive.astype(float).div(alive.sum(axis=1), axis=0)
ew = ew.loc[ew.index.hour == 0].reindex(ew.index).ffill().fillna(0.0)
ew_ret = backtest(close, ew)[0].reindex(wf.index)
btc_ret = close["BTCUSDT"].pct_change().reindex(wf.index).fillna(0)
fig, ax = plt.subplots(figsize=(9, 5))
for s, col, lab, dy in ((btc_ret, ORANGE, "BTC купил и держи", -18), (ew_ret, AQUA, "Все 10 монет поровну", 8),
                        (wf, BLUE, "Бот: лучшие монеты по моментуму", 6)):
    eq = (1 + s).cumprod() * 100
    ax.plot(eq.index, eq.values, color=col, lw=2)
    ax.annotate(f"{lab}: {eq.iloc[-1]:.0f}", (eq.index[-1], eq.iloc[-1]), xytext=(-8, dy), textcoords="offset points",
                ha="right", color=INK, fontsize=10, bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec=col, lw=1))
ax.axvline(pd.Timestamp("2026-01-01", tz="UTC"), color=INK, lw=1, ls=":")
ax.text(pd.Timestamp("2026-01-10", tz="UTC"), 40, "экзамен\n2026", fontsize=9, color=INK)
ax.set_yscale("log")
ax.set_yticks([30, 50, 100, 200, 500, 1000])
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_ylabel("Капитал (старт = 100), лог. шкала")
ax.set_title("Бот выбирает монеты: walk-forward 2022–2026, издержки учтены")
fig.tight_layout()
fig.savefig("img/xs_vs_benchmarks.png", dpi=130)
