"""Шаг 2: превращаем слабый, но устойчивый сигнал нейросети в торговое правило.

  python research/ml_signal.py HIST.csv.gz LATEST.csv --out research/ml_out

* ансамбль из трёх MLP (разные сиды), walk-forward как в ml_study.py;
* децили прогноза против реальной будущей доходности;
* редкая торговля: вход только при прогнозе сильнее квантиля q прошлых
  прогнозов (за 90 дней, без заглядывания вперёд), удержание на весь горизонт;
* бутстрэп-интервалы для годовой доходности и t-тест среднего по сделкам.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml_study import COSTS, MIN_PER_YEAR, bars, features, fit_mlp, load, target, walk_forward


def ensemble_preds(b, f, y, step, horizon, tf, seeds=(0, 1, 2)):
    preds = []
    tested = None
    for s in seeds:
        _, _, p, tested, _ = walk_forward(b, f, y, step, horizon, tf, {"mlp": fit_mlp}, seed=s)
        preds.append(p["mlp"])
    return np.nanmean(np.vstack(preds), axis=0), tested


def deciles(pred, fwd, mask):
    d = pd.DataFrame({"p": pred[mask], "y": fwd[mask] * 1e4})
    d["dec"] = pd.qcut(d.p, 10, labels=False) + 1
    g = d.groupby("dec").y
    return pd.DataFrame({"средн. прогноз, б.п.": d.groupby("dec").p.mean() * 1e4,
                         "факт. доходность, б.п.": g.mean(),
                         "± 2 ст. ошибки": 2 * g.std() / np.sqrt(g.count()),
                         "доля роста, %": d.assign(u=d.y > 0).groupby("dec").u.mean() * 100})


def trade(pred, c1_idx_px, horizon, thr, cost_bps):
    """Вход при |pred| > thr[t], удержание horizon баров; продление, если сигнал в ту же сторону."""
    n = len(pred)
    pos = np.zeros(n)
    trades = []
    cur, left, entry_i = 0.0, 0, 0
    for t in range(n):
        p = pred[t]
        sig = 0.0 if np.isnan(p) or np.isnan(thr[t]) or abs(p) <= thr[t] else np.sign(p)
        if cur != 0 and sig == cur:
            left = horizon  # сигнал подтвердился: держим дальше
        elif cur != 0:
            left -= 1
            if left <= 0 or (sig != 0 and sig != cur):
                trades.append((entry_i, t, cur))
                cur = 0.0
        if cur == 0 and sig != 0:
            cur, left, entry_i = sig, horizon, t
        pos[t] = cur
    if cur != 0:
        trades.append((entry_i, n - 1, cur))
    px = c1_idx_px
    rets = np.array([s * (px[j] / px[i] - 1) - 2 * cost_bps / 1e4 for i, j, s in trades])
    return pos, rets, trades


def bootstrap_ann(daily: pd.Series, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = daily.to_numpy()
    means = np.array([rng.choice(x, len(x)).mean() for _ in range(n)]) * 365 * 100
    return np.percentile(means, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="research/ml_out")
    ap.add_argument("--configs", default="60:1,15:4")
    args = ap.parse_args()
    out = Path(args.out)
    m = load(args.paths)
    c1 = m["close"].to_numpy()
    lines = []
    w = lines.append
    for cfg in args.configs.split(","):
        tf, horizon = map(int, cfg.split(":"))
        b = bars(m, tf)
        f = features(b, tf)
        y, step = target(b, c1, horizon)
        pred, tested = ensemble_preds(b, f, y, step, horizon, tf)
        np.save(out / f"pred_tf{tf}_h{horizon}.npy", pred)
        fwd = y.to_numpy()
        mask = tested & ~np.isnan(pred) & ~np.isnan(fwd)
        from scipy.stats import spearmanr
        w(f"\n## Таймфрейм {tf} мин, горизонт {tf * horizon} мин (ансамбль 3 нейросетей)\n")
        w(f"IC (ранговая корреляция прогноза и факта) на тесте: **{spearmanr(pred[mask], fwd[mask]).statistic:.3f}**, "
          f"наблюдений {mask.sum():,}.\n")
        w(deciles(pred, fwd, mask).round(2).to_markdown())
        w("")
        exec_px = c1[np.minimum(b["end"].to_numpy() + 1, len(c1) - 1)]
        abs_p = pd.Series(np.abs(pred), index=b.index)
        win = int(90 * 24 * 60 / tf)
        w("| Порог (квантиль прошлых прогнозов) | Издержки | Сделок/год | Средняя сделка, б.п. | t-стат | "
          "Доходность/год | 95% интервал | Шарп | Прибыльных месяцев (с торговлей) | Макс. просадка |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        years = mask.sum() / (MIN_PER_YEAR / tf)
        for q in (0.9, 0.95, 0.98, 0.99):
            thr = abs_p.rolling(win, min_periods=win // 3).quantile(q).shift(1).to_numpy()
            thr[~tested] = np.nan
            for c, cost in [("без комиссий", 0.0)] + list(COSTS.items()):
                pos, rets, trades = trade(pred, exec_px, horizon, thr, cost)
                if len(rets) < 10:
                    continue
                exit_dates = pd.DatetimeIndex([b.index[j] for _, j, _ in trades])
                daily = pd.Series(rets, index=exit_dates).groupby(exit_dates.date).sum()
                daily = daily.reindex(pd.date_range(b.index[mask][0].date(), b.index[mask][-1].date()).date,
                                      fill_value=0.0)
                ann = daily.mean() * 365 * 100
                lo, hi = bootstrap_ann(daily)
                sharpe = daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else 0
                monthly = pd.Series(rets, index=exit_dates).groupby([exit_dates.year, exit_dates.month]).sum()
                eq = daily.cumsum()
                tstat = rets.mean() / (rets.std() / np.sqrt(len(rets)))
                w(f"| {q:.2f} | {c} | {len(rets) / years:.0f} | {rets.mean() * 1e4:+.1f} | {tstat:+.1f} | "
                  f"{ann:+.1f}% | [{lo:+.1f}%, {hi:+.1f}%] | {sharpe:+.2f} | {(monthly > 0).mean() * 100:.0f}% | "
                  f"{(eq.cummax() - eq).max() * 100:.0f}% |")
                if c == "maker" and q == 0.95:
                    daily.cumsum().to_csv(out / f"signal_equity_tf{tf}_h{horizon}_q95_maker.csv")
        print(f"done {cfg}", flush=True)
    (out / "SIGNAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
