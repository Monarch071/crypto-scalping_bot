"""Массовый перебор ~1.5 млн вариантов стратегии с честной оценкой самого перебора.

  python research/mass_search.py PANEL.parquet

Варианты (список зафиксирован до запуска):
  * вход: любой непустой набор S из 15 стратегий — вход в первый час, когда ВСЕ стратегии из S «за»
    (2^15 − 1 = 32 767 наборов);
  * фильтр: монета в топ-K по силе отн. BTC, K ∈ {1, 2, 3, 5, без фильтра};
  * стоп: 3%, 5%, 10% от цены входа;
  * выход: когда из всех 15 стратегий «за» меньше 8, 10 или 12, или через 72 ч.
Итого 32 767 × 5 × 3 × 3 = 1 474 515 вариантов. Одна позиция на монету, только лонг, комиссии фьючерсов
+ ручной вход (как в research/confluence_final.py). Результат в R (1R = расстояние до стопа).

Честная оценка перебора (walk-forward): для каждого года Y из 2022–2026 выбираются лучшие варианты
по сумме R за три предыдущих года (минимум 30 сделок), и их результат считается только в году Y.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rotator.strategy import eligible, momentum_vs_btc  # noqa: E402
from signalbot.indicators import votes  # noqa: E402

YEARS = np.arange(2020, 2027)
TOPKS = (1, 2, 3, 5, 99)
STOPS = (0.03, 0.05, 0.10)
EXIT_VOTES = (8, 10, 12)
MAX_HOLD = 72
ENTRY_BPS, STOP_BPS, OTHER_BPS = 15.5, 10.5, 15.5
N_STRAT = 15


@njit(cache=True)
def exits_for_all_hours(H, L, C, cnt, stop_pct, exit_votes):
    """Итог сделки, открытой в каждый час: R и индекс выхода."""
    n = len(C)
    R = np.full(n, np.nan)
    X = np.full(n, -1, dtype=np.int64)
    for e in range(n - 2):
        p0 = C[e]
        stop = p0 * (1 - stop_pct)
        end = min(e + MAX_HOLD, n - 1)
        px, x, cost = C[end], end, OTHER_BPS
        for j in range(e + 1, end + 1):
            if L[j] <= stop:
                px, x, cost = stop, j, STOP_BPS
                break
            if cnt[j] < exit_votes:
                px, x, cost = C[j], j, OTHER_BPS
                break
        R[e] = ((px / p0 - 1) * 1e4 - ENTRY_BPS - cost) / (stop_pct * 1e4)
        X[e] = x
    return R, X


@njit(parallel=True, cache=True)
def search(bits, rank, year_idx, coin_start, coin_end, exitR, exitX, topks, n_years):
    """res[S, k, c, y] — сумма R; cnt[S, k, c, y] — число сделок."""
    n_sub = 1 << N_STRAT
    n_k, n_c = len(topks), exitR.shape[0]
    res = np.zeros((n_sub, n_k, n_c, n_years), dtype=np.float32)
    ntr = np.zeros((n_sub, n_k, n_c, n_years), dtype=np.int32)
    for S in prange(1, n_sub):
        busy = np.empty((n_k, n_c), dtype=np.int64)
        for ci in range(len(coin_start)):
            busy[:, :] = -1
            prev = False
            for h in range(coin_start[ci], coin_end[ci]):
                cond = (bits[h] & S) == S
                first = cond and not prev
                prev = cond
                if not first:
                    continue
                y = year_idx[h]
                for ki in range(n_k):
                    if rank[h] > topks[ki]:
                        continue
                    for c in range(n_c):
                        if h > busy[ki, c] and exitX[c, h] >= 0:
                            res[S, ki, c, y] += exitR[c, h]
                            ntr[S, ki, c, y] += 1
                            busy[ki, c] = exitX[c, h]
    return res, ntr


def build(panel_path):
    panel = pd.read_parquet(panel_path)
    panel["timestamp"] = pd.to_datetime(panel.timestamp, utc=True)
    fr = {s.replace("USDT", "/USDT"): d.set_index("timestamp").sort_index() for s, d in panel.groupby("symbol")}
    idx = pd.date_range(pd.Timestamp("2020-01-01", tz="UTC"), panel.timestamp.max(), freq="h")
    close = pd.DataFrame({s: f.close for s, f in fr.items()}).reindex(idx).ffill(limit=3)
    rank_all = momentum_vs_btc(close, 168).where(eligible(close, 1440)).rank(axis=1, ascending=False)
    btc = close["BTC/USDT"]
    parts = []
    for sym, df in fr.items():
        df = df.reindex(idx).ffill(limit=3).dropna()
        V = votes(df, btc.reindex(df.index)).iloc[24 * 35:]
        d = df.reindex(V.index)
        b = np.zeros(len(V), dtype=np.int64)
        for i, col in enumerate(V.columns):
            b |= (V[col].to_numpy() > 0).astype(np.int64) << i
        parts.append(dict(sym=sym, bits=b, cnt=(V.to_numpy() > 0).sum(1), H=d.high.to_numpy(), L=d.low.to_numpy(),
                          C=d.close.to_numpy(), rank=rank_all[sym].reindex(V.index).fillna(999).to_numpy(),
                          year=V.index.year.to_numpy(), names=list(V.columns)))
    return parts


def main():
    t0 = time.time()
    parts = build(sys.argv[1])
    names = parts[0]["names"]
    cfgs = [(s, xv) for s in STOPS for xv in EXIT_VOTES]
    bits = np.concatenate([p["bits"] for p in parts])
    rank = np.concatenate([p["rank"] for p in parts])
    year_idx = np.concatenate([p["year"] for p in parts]) - YEARS[0]
    lens = np.array([len(p["bits"]) for p in parts])
    coin_end = np.cumsum(lens)
    coin_start = coin_end - lens
    exitR = np.full((len(cfgs), len(bits)), np.nan)
    exitX = np.full((len(cfgs), len(bits)), -1, dtype=np.int64)
    for ci, (s, xv) in enumerate(cfgs):
        for p, a, b in zip(parts, coin_start, coin_end):
            r, x = exits_for_all_hours(p["H"], p["L"], p["C"], p["cnt"], s, xv)
            exitR[ci, a:b] = r
            exitX[ci, a:b] = np.where(x >= 0, x + a, -1)
    print(f"подготовка: {time.time() - t0:.0f} с", flush=True)
    res, ntr = search(bits, rank, year_idx.astype(np.int64), coin_start, coin_end, exitR, exitX,
                      np.array(TOPKS, dtype=np.float64), len(YEARS))
    print(f"перебор: {time.time() - t0:.0f} с", flush=True)

    flat_R = res[1:].reshape(-1, len(YEARS))
    flat_N = ntr[1:].reshape(-1, len(YEARS))
    n_var = flat_R.shape[0]
    S_of = np.repeat(np.arange(1, 1 << N_STRAT), len(TOPKS) * len(cfgs))
    k_of = np.tile(np.repeat(np.arange(len(TOPKS)), len(cfgs)), (1 << N_STRAT) - 1)
    c_of = np.tile(np.arange(len(cfgs)), ((1 << N_STRAT) - 1) * len(TOPKS))

    def describe(i):
        S = S_of[i]
        used = [names[j] for j in range(N_STRAT) if S >> j & 1]
        k = TOPKS[k_of[i]]
        s, xv = cfgs[c_of[i]]
        return (f"{len(used)} стратегий, топ-{k if k < 99 else 'все'}, стоп {s:.0%}, выход <{xv}/15: "
                + ", ".join(used))

    current = int(np.where((S_of == (1 << N_STRAT) - 1) & (k_of == TOPKS.index(3)) &
                           (c_of == cfgs.index((0.05, 10))))[0][0])
    lines = [f"Вариантов: {n_var:,}", ""]
    lines.append("## Честная оценка перебора: выбор по трём прошлым годам, результат в следующем году (R)")
    lines.append("")
    lines.append("| Год | Лучший 1 вариант | Среднее лучших 100 | Среднее лучших 1000 | Медиана всех | Текущие правила |")
    lines.append("|---|---|---|---|---|---|")
    tot = np.zeros(4)
    for y in range(2022, 2027):
        yi = y - YEARS[0]
        past = flat_R[:, yi - 3:yi].sum(1) if yi >= 3 else flat_R[:, :yi].sum(1)
        enough = flat_N[:, max(0, yi - 3):yi].sum(1) >= 30
        score = np.where(enough, past, -np.inf)
        order = np.argsort(-score)
        vals = [flat_R[order[0], yi], flat_R[order[:100], yi].mean(), flat_R[order[:1000], yi].mean()]
        med = np.median(flat_R[flat_N[:, yi] > 0, yi])
        tot += np.array(vals + [flat_R[current, yi]])
        lines.append(f"| {y} | {vals[0]:+.1f} | {vals[1]:+.1f} | {vals[2]:+.1f} | {med:+.1f} | {flat_R[current, yi]:+.1f} |")
        if y == 2026:
            best_desc = describe(order[0])
    lines.append(f"| **2022–2026** | **{tot[0]:+.1f}** | **{tot[1]:+.1f}** | **{tot[2]:+.1f}** | | **{tot[3]:+.1f}** |")
    lines += ["", f"Вариант, который перебор выбрал бы на 2026 год: {best_desc}", ""]

    # Насколько лучшие «на истории» варианты лучше на экзамене: 2020–2025 против 2026.
    in_s = flat_R[:, :6].sum(1)
    ok = flat_N[:, :6].sum(1) >= 60
    q = np.quantile(in_s[ok], [0.5, 0.9, 0.99, 0.999])
    lines.append("## Лучшие на 2020–2025 против 2026 года")
    lines.append("")
    lines.append("| Группа по 2020–2025 | Средний R за 2020–2025 | Средний R в 2026 | Доля с плюсом в 2026 |")
    lines.append("|---|---|---|---|")
    for lo, label in zip([-np.inf] + list(q), ["нижние 50%", "50–90%", "90–99%", "99–99.9%", "лучшие 0.1%"]):
        hi = q[[-np.inf, *q].index(lo)] if lo != q[-1] else np.inf
        m = ok & (in_s > lo) & (in_s <= hi)
        lines.append(f"| {label} | {in_s[m].mean():+.1f} | {flat_R[m, 6].mean():+.1f} | {(flat_R[m, 6] > 0).mean():.0%} |")

    # Какие стратегии чаще всего входят в устойчиво хорошие варианты (плюс в 6 из 7 лет и выше).
    yrs_pos = (flat_R > 0).sum(1)
    good = ok & (yrs_pos >= 6) & (flat_N[:, :6].sum(1) >= 100)
    share = [np.mean([(S_of[i] >> j) & 1 for i in np.where(good)[0]]) for j in range(N_STRAT)] if good.any() else []
    lines += ["", f"## Устойчивые варианты (в плюсе ≥ 6 лет из 7, ≥ 100 сделок): {good.sum():,}", ""]
    if good.any():
        lines.append("| Стратегия | Доля в устойчивых вариантах |")
        lines.append("|---|---|")
        for j in np.argsort(share)[::-1]:
            lines.append(f"| {names[j]} | {share[j]:.0%} |")
        topks_share = pd.Series([TOPKS[k] for k in k_of[good]]).value_counts(normalize=True)
        cfg_share = pd.Series([cfgs[c] for c in c_of[good]]).value_counts(normalize=True)
        lines += ["", "Фильтр топ-K среди устойчивых: " + ", ".join(f"{k if k < 99 else 'нет'}: {v:.0%}"
                                                               for k, v in topks_share.items()),
                  "Стоп и выход среди устойчивых: " + ", ".join(f"{s:.0%}/<{x}: {v:.0%}"
                                                             for (s, x), v in cfg_share.head(5).items())]
    out = "\n".join(lines)
    print(out)
    Path("mass_out").mkdir(exist_ok=True)
    Path("mass_out/MASS_TABLES.md").write_text(out, encoding="utf-8")
    np.savez_compressed("mass_out/results.npz", R=flat_R, N=flat_N)
    print(f"всего: {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
