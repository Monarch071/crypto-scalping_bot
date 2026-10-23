"""Случайные входы в монеты из топ-K с теми же стопами и выходами — контроль для mass_search.py.

  python research/mass_baseline.py PANEL.parquet
"""
import sys

import numpy as np

import mass_search as M

parts = M.build(sys.argv[1])
rng = np.random.default_rng(0)
cfg = (0.05, 10)
rows = {"текущие правила (все 15 «за»)": [], "случайный вход в топ-3": [], "случайный вход в любую монету": []}
for rep in range(200):
    yr = {k: np.zeros(len(M.YEARS)) for k in rows}
    for p in parts:
        R, X = M.exits_for_all_hours(p["H"], p["L"], p["C"], p["cnt"], *cfg)
        on = (p["bits"] & 0x7FFF) == 0x7FFF
        first = on & ~np.r_[False, on[:-1]]
        sig = np.where(first & (p["rank"] <= 3) & (X >= 0))[0]
        pools = {"текущие правила (все 15 «за»)": sig,
                 "случайный вход в топ-3": np.where((p["rank"] <= 3) & (X >= 0))[0],
                 "случайный вход в любую монету": np.where(X >= 0)[0]}
        for name, pool in pools.items():
            if name.startswith("текущие") and rep > 0:
                continue
            e = pool if name.startswith("текущие") else np.sort(rng.choice(pool, size=len(sig), replace=False))
            busy = -1
            for h in e:
                if h > busy:
                    yr[name][p["year"][h] - M.YEARS[0]] += R[h]
                    busy = X[h]
    for k in rows:
        if k.startswith("текущие") and rep > 0:
            continue
        rows[k].append(yr[k])
print("Сумма R по годам (стоп 5%, выход <10/15, столько же входов, сколько у правил; случайные — среднее 200 повторов)")
print("| Входы | " + " | ".join(str(y) for y in M.YEARS) + " | 2022–2026 | Лет в плюсе |")
print("|---|" + "---|" * (len(M.YEARS) + 2))
for k, v in rows.items():
    m = np.mean(v, axis=0)
    print(f"| {k} | " + " | ".join(f"{x:+.1f}" for x in m) + f" | {m[2:].sum():+.1f} | {(m > 0).sum()}/7 |")
    if not k.startswith("текущие"):
        tot = np.array(v)[:, 2:].sum(1)
        print(f"|   разброс случайных 2022–2026 (5%–95%) | | | | | | | | {np.percentile(tot, 5):+.1f} … {np.percentile(tot, 95):+.1f} | |")
