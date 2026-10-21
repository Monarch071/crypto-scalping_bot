"""Следование тренду с управлением риском: позиция живёт часы–дни, пока есть тренд.

  python research/trend_study.py HIST.csv.gz LATEST.csv

Так работают CTA-фонды: бот в рынке, пока тренд есть, вне рынка, когда его нет,
а размер позиции обратно пропорционален волатильности (volatility targeting).
  * сигналы: цена выше/ниже EMA(n) и пересечение двух EMA, на часовых барах;
  * режимы: «лонг или вне рынка» и «лонг или шорт»;
  * размер: target_vol / текущая волатильность, без плеча (≤ 1);
  * комиссия тейкера 6.5 б.п. с каждого изменения позиции (ребаланс, только
    если размер изменился больше чем на 20%);
  * параметры выбираются walk-forward: по двум прошлым годам, тест — следующий год.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from ml_study import bars, load

COST_BPS = 6.5
HOURS_PER_YEAR = 365 * 24


def ema(x, n):
    return x.ewm(span=n, adjust=False).mean()


def direction(c: pd.Series, kind: str, a: int, b: int, mode: str) -> pd.Series:
    if kind == "price>ema":
        d = np.sign(c - ema(c, a))
    else:
        d = np.sign(ema(c, a) - ema(c, b))
    if mode == "long/flat":
        d = d.clip(lower=0)
    return d.fillna(0)


def run(c: pd.Series, d: pd.Series, target_vol: float | None) -> pd.Series:
    r = c.pct_change().fillna(0)
    if target_vol is None:
        w = d.copy()
    else:
        vol = np.log(c).diff().rolling(24 * 30).std() * np.sqrt(HOURS_PER_YEAR)
        w = d * (target_vol / vol).clip(upper=1.0)
        # Ребалансируем только при заметном изменении размера, чтобы не платить за шум.
        held = np.zeros(len(w))
        cur = 0.0
        for i, x in enumerate(np.nan_to_num(w.to_numpy())):
            if np.sign(x) != np.sign(cur) or abs(x - cur) > 0.2 * max(abs(cur), 1e-9):
                cur = x
            held[i] = cur
        w = pd.Series(held, index=c.index)
    turnover = w.diff().abs().fillna(w.abs())
    return w.shift(1).fillna(0) * r - turnover * COST_BPS / 1e4, w


def stats(ret: pd.Series, w: pd.Series) -> dict:
    eq = (1 + ret).cumprod()
    yrs = len(ret) / HOURS_PER_YEAR
    daily = ret.groupby(ret.index.normalize()).apply(lambda x: (1 + x).prod() - 1)
    monthly = ret.groupby([ret.index.year, ret.index.month]).apply(lambda x: (1 + x).prod() - 1)
    changes = (w.diff().abs() > 0).sum()
    return dict(cagr=(eq.iloc[-1] ** (1 / yrs) - 1) * 100,
                sharpe=daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else 0,
                mdd=(1 - eq / eq.cummax()).max() * 100,
                pos_months=(monthly > 0).mean() * 100,
                in_market=(w.abs() > 0).mean() * 100,
                changes_y=changes / yrs)


GRID = [("price>ema", n, 0) for n in (24, 48, 96, 168, 336, 720)] + \
       [("ema cross", a, b) for a, b in itertools.product((12, 24, 48, 96), (96, 168, 336, 720)) if a < b]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    m = load(args.paths)
    c = bars(m, 60).c
    test_years = range(2022, 2027)
    lines = ["| Стратегия | Доходность/год | Шарп | Макс. просадка | Прибыльных месяцев | В рынке | Сделок/год | "
             + " | ".join(str(y) for y in test_years) + " |",
             "|---|---|---|---|---|---|---|" + "---|" * len(test_years)]

    def row(name, ret, w):
        s = stats(ret, w)
        yearly = [((1 + ret[ret.index.year == y]).prod() - 1) * 100 for y in test_years]
        lines.append(f"| {name} | {s['cagr']:+.1f}% | {s['sharpe']:.2f} | {s['mdd']:.0f}% | {s['pos_months']:.0f}% | "
                     f"{s['in_market']:.0f}% | {s['changes_y']:.0f} | " + " | ".join(f"{v:+.0f}%" for v in yearly) + " |")

    test_mask = c.index.year >= test_years[0]
    bh = pd.Series(1.0, index=c.index)
    r_bh, w_bh = run(c, bh, None)
    row("Купил и держи", r_bh[test_mask], w_bh[test_mask])
    r_bhv, w_bhv = run(c, bh, 0.4)
    row("Купил и держи, объём по волатильности (40%)", r_bhv[test_mask], w_bhv[test_mask])

    for mode, tv in itertools.product(("long/flat", "long/short"), (None, 0.4)):
        cache = {g: run(c, direction(c, g[0], g[1], g[2], mode), tv) for g in GRID}
        parts_r, parts_w, chosen = [], [], []
        for y in test_years:
            train = (c.index.year >= y - 2) & (c.index.year < y)
            best = max(GRID, key=lambda g: (lambda x: x.mean() / x.std() if x.std() > 0 else -9)(cache[g][0][train]))
            chosen.append(f"{y}:{best[0]} {best[1]}{'/' + str(best[2]) if best[2] else ''}")
            parts_r.append(cache[best][0][c.index.year == y])
            parts_w.append(cache[best][1][c.index.year == y])
        name = f"Тренд {mode}" + (", объём по волатильности" if tv else "")
        row(name, pd.concat(parts_r), pd.concat(parts_w))
        print(name, "→", "; ".join(chosen))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
