"""График: тренд-бот против «купил и держи» (walk-forward 2022–2026).

  python research/plot_trend.py HIST.csv.gz LATEST.csv
"""
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

from ml_charts import BLUE, INK, ORANGE, SURFACE
from ml_study import bars, load
from trend_study import GRID, direction, run

c = bars(load(sys.argv[1:]), 60).c
cache = {g: run(c, direction(c, g[0], g[1], g[2], "long/flat"), 0.4) for g in GRID}
parts = []
for y in range(2022, 2027):
    tr = (c.index.year >= y - 2) & (c.index.year < y)
    best = max(GRID, key=lambda g: (lambda x: x.mean() / x.std())(cache[g][0][tr]))
    parts.append(cache[best][0][c.index.year == y])
trend = pd.concat(parts)
bh = run(c, pd.Series(1.0, index=c.index), None)[0][c.index.year >= 2022]
fig, ax = plt.subplots(figsize=(9, 5))
for s, col, lab in ((bh, ORANGE, "Купил и держи"), (trend, BLUE, "Тренд-бот (лонг или вне рынка)")):
    eq = (1 + s).cumprod() * 100
    ax.plot(eq.index, eq.values, color=col, lw=2)
    ax.annotate(f"{lab}: {eq.iloc[-1]:.0f}", (eq.index[-1], eq.iloc[-1]), xytext=(-8, 8), textcoords="offset points",
                ha="right", color=INK, fontsize=10, bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec=col, lw=1))
ax.set_yscale("log")
ax.set_yticks([30, 50, 100, 200, 300])
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_ylabel("Капитал (старт = 100), лог. шкала")
ax.set_title("Тренд-бот против «купил и держи»: BTC, 2022–2026, вне выборки")
fig.tight_layout()
fig.savefig("img/trend_vs_bh.png", dpi=130)
