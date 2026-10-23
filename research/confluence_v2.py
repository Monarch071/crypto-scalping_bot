"""Улучшение стратегии «совпадение 15 стратегий» по заранее зафиксированному плану.

  python research/confluence_v2.py PANEL.parquet --out research/confluence_out

Защита от подгонки под историю:
  * список вариантов зафиксирован в коде ДО запуска (EXITS × FILTERS + ML-взвешивание);
  * лучший вариант выбирается walk-forward: на каждый год — по сумме R за два предыдущих года;
    итог процесса улучшения считается только по годам, не участвовавшим в выборе;
  * сравнение с медианой всех вариантов показывает, сколько «выигрыша» даёт сам выбор.

Сделки: только лонг (шорты в первом исследовании хуже случайных), одна позиция на монету за раз,
комиссии фьючерсов Bybit/Binance + 10 б.п. на ручной вход, 5 б.п. проскальзывания на стопах.
Результат в R (1R = 2 ATR — стартовое расстояние до стопа); при риске 1% на сделку R = % депозита.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rotator.strategy import eligible, momentum_vs_btc  # noqa: E402
from signalbot.indicators import atr, ema, votes  # noqa: E402

ENTRY_BPS, TP_BPS, STOP_BPS, MKT_BPS = 15.5, 2.0, 10.5, 5.5
CANDIDATE_VOTES = 13          # ML видит и «почти согласные» входы
EXITS = {  # mode, max_hold
    "стоп 2 / тейк 4 ATR, 24 ч": (0, 24),
    "выход, когда «за» < 10 из 15": (1, 48),
    "скользящий стоп 2 ATR": (2, 48),
    "безубыток после +2 ATR": (3, 24),
}
FILTERS = ("без фильтра", "топ-3 по силе отн. BTC", "спокойный рынок", "не догонять цену")


@njit(cache=True)
def simulate(H, L, C, A, cnt, entries, mode, max_hold, exit_votes):
    n = len(entries)
    gross = np.empty(n)
    kind = np.empty(n, dtype=np.int8)  # 1 тейк, -1 стоп, 0 время, 2 сигнал
    xi = np.empty(n, dtype=np.int64)
    last = len(C) - 1
    for i in range(n):
        e = entries[i]
        p0, a = C[e], A[e]
        stop, tp = p0 - 2 * a, p0 + 4 * a
        hh = p0
        end = min(e + max_hold, last)
        k, px, x = 0, C[end], end
        for j in range(e + 1, end + 1):
            if L[j] <= stop:                       # стоп проверяется первым (пессимистично)
                k, px, x = -1, stop, j
                break
            if mode in (0, 3) and H[j] >= tp:
                k, px, x = 1, tp, j
                break
            if mode == 1 and cnt[j] < exit_votes:   # стратегии перестали соглашаться — выход по закрытию
                k, px, x = 2, C[j], j
                break
            hh = max(hh, H[j])
            if mode == 2:
                stop = max(stop, hh - 2 * a)
            if mode == 3 and hh >= p0 + 2 * a:
                stop = max(stop, p0)
        gross[i] = (px / p0 - 1) * 1e4
        kind[i] = k
        xi[i] = x
    return gross, kind, xi


def build(panel_path):
    panel = pd.read_parquet(panel_path)
    panel["timestamp"] = pd.to_datetime(panel.timestamp, utc=True)
    fr = {s.replace("USDT", "/USDT"): d.set_index("timestamp").sort_index() for s, d in panel.groupby("symbol")}
    idx = pd.date_range(pd.Timestamp("2020-01-01", tz="UTC"), panel.timestamp.max(), freq="h")
    close = pd.DataFrame({s: f.close for s, f in fr.items()}).reindex(idx).ffill(limit=3)
    rank = momentum_vs_btc(close, 168).where(eligible(close, 1440)).rank(axis=1, ascending=False)
    btc = close["BTC/USDT"]
    rows = []
    for sym, df in fr.items():
        df = df.reindex(idx).ffill(limit=3).dropna()
        V = votes(df, btc.reindex(df.index)).iloc[24 * 35:]
        d = df.reindex(V.index)
        H, L, C = d.high.to_numpy(), d.low.to_numpy(), d.close.to_numpy()
        A = atr(df.high, df.low, df.close).reindex(V.index).to_numpy()
        cnt = (V.to_numpy() > 0).sum(1)
        # Кандидаты: первый час с ≥13 «за» (для ML) и первый час со всеми 15 «за» (для правил).
        fc = {}
        for thr in (CANDIDATE_VOTES, 15):
            on = cnt >= thr
            fc[thr] = on & ~np.r_[False, on[:-1]] & ~np.isnan(A)
        e = np.where(fc[CANDIDATE_VOTES] | fc[15])[0]
        e = e[e < len(C) - 2]
        if not len(e):
            continue
        atr_pct = pd.Series(A / C, index=V.index)
        calm = (atr_pct <= atr_pct.rolling(24 * 30, min_periods=24 * 7).median()).to_numpy()
        ext = ((d.close - ema(d.close, 20)) / A).to_numpy()
        feats = pd.DataFrame(V.to_numpy()[e], columns=[f"v{i}" for i in range(V.shape[1])])
        feats["count"] = cnt[e]
        feats["atr_pct"] = atr_pct.to_numpy()[e]
        feats["extension"] = ext[e]
        feats["rank"] = rank[sym].reindex(V.index).to_numpy()[e] if sym in rank else np.nan
        feats["btc_24h"] = np.log(btc / btc.shift(24)).reindex(V.index).to_numpy()[e]
        feats["hour"] = V.index[e].hour
        base = dict(sym=sym, time=V.index[e], votes=cnt[e], fc15=fc[15][e], fc13=fc[CANDIDATE_VOTES][e],
                    calm=calm[e], ext=ext[e],
                    rank=feats["rank"].to_numpy(), stop_bps=2 * A[e] / C[e] * 1e4)
        for xname, (mode, hold) in EXITS.items():
            g, k, xi = simulate(H, L, C, A, cnt, e, mode, hold, 10)
            cost = ENTRY_BPS + np.where(k == 1, TP_BPS, np.where(k == -1, STOP_BPS, MKT_BPS))
            rows.append(pd.concat([pd.DataFrame({**base, "exit": xname, "kind": k, "exit_time": V.index[xi],
                                                 "net": g - cost}), feats.add_prefix("f_")], axis=1))
        print(f"{sym}: {len(e)} кандидатов", flush=True)
    T = pd.concat(rows, ignore_index=True)
    T["R"] = T.net / T.stop_bps
    return T


def one_per_coin(df: pd.DataFrame) -> pd.DataFrame:
    """Новая сделка по монете только после выхода из предыдущей."""
    keep = []
    for _, g in df.sort_values("time").groupby("sym"):
        busy = None
        for i, t, x in zip(g.index, g.time, g.exit_time):
            if busy is None or t > busy:
                keep.append(i)
                busy = x
    return df.loc[keep]


def rule_mask(T, flt):
    m = T.fc15.copy()
    if flt == "топ-3 по силе отн. BTC":
        m &= T["rank"] <= 3
    elif flt == "спокойный рынок":
        m &= T.calm
    elif flt == "не догонять цену":
        m &= T.ext <= 3
    return m


def ml_mask(T, xname):
    """LightGBM учит по прошлым двум годам, каким кандидатам (≥13 голосов) доверять; порог — тоже по прошлому."""
    feats = [c for c in T.columns if c.startswith("f_")]
    D = T[(T.exit == xname) & T.fc13]
    take = pd.Series(False, index=T.index)
    for y in range(2022, 2027):
        tr = D[(D.time.dt.year >= y - 2) & (D.time.dt.year < y)]
        te = D[D.time.dt.year == y]
        if len(tr) < 300 or not len(te):
            continue
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=50,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                               random_state=0, verbose=-1, n_jobs=4)
        cut = int(len(tr) * 0.7)
        m.fit(tr[feats].iloc[:cut], (tr.R.iloc[:cut] > 0).astype(int))
        pv = m.predict_proba(tr[feats].iloc[cut:])[:, 1]
        best_q, best = 0.5, -1e9
        for q in (0.3, 0.5, 0.7, 0.8):
            s = tr.R.iloc[cut:][pv > np.quantile(pv, q)].sum()
            if s > best:
                best, best_q = s, np.quantile(pv, q)
        m.fit(tr[feats], (tr.R > 0).astype(int))
        take.loc[te.index] = m.predict_proba(te[feats])[:, 1] > best_q
    return take


def yearly(sel: pd.DataFrame) -> pd.Series:
    return sel.groupby(sel.time.dt.year).R.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panel")
    ap.add_argument("--out", default="research/confluence_out")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    T = build(args.panel)
    T.to_parquet(out / "v2_candidates.parquet")

    variants = {}
    for xname, flt in itertools.product(EXITS, FILTERS):
        variants[(xname, flt)] = one_per_coin(T[(T.exit == xname) & rule_mask(T, flt)])
    for xname in EXITS:
        variants[(xname, "ML-взвешивание")] = one_per_coin(T[(T.exit == xname) & ml_mask(T, xname)])

    years = range(2020, 2027)
    table = pd.DataFrame({k: yearly(v).reindex(years, fill_value=0.0) for k, v in variants.items()}).T
    table["2022–2026"] = table[[2022, 2023, 2024, 2025, 2026]].sum(axis=1)
    n = {k: len(v[v.time.dt.year >= 2022]) for k, v in variants.items()}
    table["сделок 2022–26"] = pd.Series(n)
    table["R/сделку"] = table["2022–2026"] / table["сделок 2022–26"]
    pd.set_option("display.width", 250)
    print(table.round(1).sort_values("2022–2026", ascending=False).to_string())

    # Walk-forward выбор: на год Y берём вариант с лучшей суммой R за Y-2..Y-1.
    wf, chosen = [], []
    for y in range(2022, 2027):
        best = table[[y - 2, y - 1]].sum(axis=1).idxmax()
        chosen.append(f"{y}: {best[0]} + {best[1]}")
        wf.append(table.loc[best, y])
    print("\nWalk-forward (выбор варианта по двум прошлым годам):")
    print("\n".join(f"  {c} → {r:+.1f}R" for c, r in zip(chosen, wf)))
    print(f"  Итого 2022–2026: {sum(wf):+.1f}R (% депозита при риске 1%); "
          f"медиана всех {len(table)} вариантов: {table['2022–2026'].median():+.1f}R; "
          f"исходный (стоп 2/тейк 4, без фильтра): {table.loc[(list(EXITS)[0], 'без фильтра'), '2022–2026']:+.1f}R")
    table.to_csv(out / "v2_table.csv")


if __name__ == "__main__":
    main()
