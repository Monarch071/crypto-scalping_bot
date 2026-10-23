"""Торговый (не скальпинг) бот: вход по сигналу, выход по цели / стопу / времени.

  python research/swing_study.py HIST.csv.gz LATEST.csv --out research/swing_out

Метод тройного барьера (López de Prado, AFML гл. 3) + meta-labeling:
  1. Первичный сигнал даёт направление: пробой дневного экстремума, откат в
     тренде или сильное отклонение от среднего.
  2. Сделка живёт до первого барьера: цель (k_tp·σ), стоп (k_sl·σ) или лимит
     времени (1 или 4 часа). σ = текущая волатильность на горизонт удержания.
     Барьеры проверяются по минутным high/low; если в одной минуте задеты
     оба — считаем, что сработал стоп (пессимистично).
  3. Издержки: вход тейкером (6.5 б.п. с проскальзыванием), цель лимитным
     ордером (2 б.п.), стоп рыночным (6.5 + 2 б.п. проскальзывания на стопе),
     выход по времени тейкером.
  4. ML-фильтр (LightGBM) учится отличать сигналы, которые закончатся
     прибылью; walk-forward 2 года / 3 месяца с зазором, контроль с
     перемешанной меткой.
  5. Выбор лучшей конфигурации — только по 2022–2025; 2026 год — экзамен.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from numba import njit

from ml_study import bars, features, load

TF = 15
ENTRY_BPS, TP_BPS, SL_BPS, TIME_BPS = 6.5, 2.0, 8.5, 6.5
HOLDOUT_YEAR = 2026
BARRIERS = {  # (цель, стоп) в σ горизонта; inf = без цели, держим до стопа или времени
    "цель 1σ / стоп 1σ": (1.0, 1.0),
    "цель 2σ / стоп 1σ": (2.0, 1.0),
    "цель 1σ / стоп 2σ": (1.0, 2.0),
    "без цели / стоп 1.5σ": (np.inf, 1.5),
}
HOLDS = (60, 240)  # минут


# ------------------------------------------------------------------ сигналы
def first(cond: pd.Series) -> pd.Series:
    return cond & ~cond.shift(1, fill_value=False)


def signals(b: pd.DataFrame) -> dict[str, pd.Series]:
    c, day = b.c, 96
    hi = b.h.rolling(day).max().shift(1)
    lo = b.l.rolling(day).min().shift(1)
    ema50, ema200 = c.ewm(span=50, adjust=False).mean(), c.ewm(span=200, adjust=False).mean()
    d = np.log(c).diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    z = (c - c.rolling(day).mean()) / c.rolling(day).std()

    def mk(long, short):
        s = pd.Series(0, index=b.index, dtype=np.int8)
        s[first(long.fillna(False))] = 1
        s[first(short.fillna(False))] = -1
        return s

    return {
        "Пробой дневного экстремума": mk(c > hi, c < lo),
        "Откат в тренде": mk((ema50 > ema200) & (rsi < 30), (ema50 < ema200) & (rsi > 70)),
        "Сильное отклонение от среднего": mk(z < -2.5, z > 2.5),
    }


# ------------------------------------------------------------------ барьеры
@njit(cache=True)
def simulate(H, L, C, entry_idx, side, tp, sl, hold, entry_bps, tp_bps, sl_bps, time_bps):
    n = len(entry_idx)
    net = np.empty(n)
    exit_idx = np.empty(n, dtype=np.int64)
    kind = np.empty(n, dtype=np.int8)  # 1 цель, -1 стоп, 0 время
    last = len(C) - 1
    for i in range(n):
        e = entry_idx[i]
        p0 = C[e]
        s = side[i]
        tp_px = p0 * np.exp(s * tp[i])
        sl_px = p0 * np.exp(-s * sl[i])
        end = min(e + hold, last)
        k, x, px = 0, end, C[end]
        for j in range(e + 1, end + 1):
            if s > 0:
                if L[j] <= sl_px:
                    k, x, px = -1, j, sl_px
                    break
                if H[j] >= tp_px:
                    k, x, px = 1, j, tp_px
                    break
            else:
                if H[j] >= sl_px:
                    k, x, px = -1, j, sl_px
                    break
                if L[j] <= tp_px:
                    k, x, px = 1, j, tp_px
                    break
        gross = s * (px / p0 - 1) * 1e4
        cost = entry_bps + (tp_bps if k == 1 else (sl_bps if k == -1 else time_bps))
        net[i] = gross - cost
        exit_idx[i] = x
        kind[i] = k
    return net, exit_idx, kind


def sequential(entry_idx, exit_idx, take):
    """Одна позиция за раз: сигнал пропускается, пока предыдущая сделка открыта."""
    out = np.zeros(len(take), dtype=bool)
    busy_until = -1
    for i in np.argsort(entry_idx, kind="stable"):
        if take[i] and entry_idx[i] > busy_until:
            out[i] = True
            busy_until = exit_idx[i]
    return out


# ------------------------------------------------------------------ ML-фильтр
def meta_filter(X, y_net, times, hold_bars, shuffle=False, seed=0):
    """Walk-forward: вероятность прибыльной сделки и решение брать/не брать."""
    rng = np.random.default_rng(seed)
    take = np.zeros(len(y_net), dtype=bool)
    tested = np.zeros(len(y_net), dtype=bool)
    t0 = times[0] + pd.DateOffset(years=2)
    while t0 < times[-1]:
        t1 = t0 + pd.DateOffset(months=3)
        embargo = pd.Timedelta(minutes=hold_bars * TF)
        tr = np.where((times >= t0 - pd.DateOffset(years=2)) & (times < t0 - embargo))[0]
        te = np.where((times >= t0) & (times < t1))[0]
        t0 = t1
        if len(te) == 0:
            continue
        tested[te] = True
        if len(tr) < 300:
            take[te] = True
            continue
        cut = int(len(tr) * 0.75)
        fi, vi = tr[:cut], tr[cut:]
        lab = (y_net[fi] > 0).astype(int)
        if shuffle:
            lab = rng.permutation(lab)
        model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=7,
                                   min_child_samples=40, subsample=0.8, subsample_freq=1,
                                   colsample_bytree=0.7, reg_lambda=5.0, random_state=seed,
                                   verbose=-1, n_jobs=4)
        model.fit(X[fi], lab)
        pv, pt = model.predict_proba(X[vi])[:, 1], model.predict_proba(X[te])[:, 1]
        # Порог по валидации; «брать всё» тоже вариант, «ничего» — если всё в минус.
        best_q, best = None, 0.0
        for q in (0.0, 0.3, 0.5, 0.7, 0.8, 0.9):
            thr = np.quantile(pv, q) if q > 0 else -1
            s = y_net[vi][pv > thr].sum()
            if s > best:
                best, best_q = s, thr
        if best_q is not None:
            take[te] = pt > best_q
    return take, tested


# ------------------------------------------------------------------ метрики
def metrics(net, entry_times, exit_times, side, days):
    if len(net) < 5:
        return dict(trades=len(net), trades_y=0, win=np.nan, avg_bps=np.nan, t=np.nan, ann=np.nan,
                    sharpe=np.nan, pos_months=np.nan, mdd=np.nan, long_ann=np.nan, short_ann=np.nan)
    years = days / 365
    s = pd.Series(net / 1e4, index=exit_times)
    daily = s.groupby(s.index.normalize()).sum()
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"), fill_value=0)
    monthly = s.groupby([s.index.year, s.index.month]).sum()
    eq = daily.cumsum()
    return dict(
        trades=len(net), trades_y=len(net) / years, win=(net > 0).mean() * 100, avg_bps=net.mean(),
        t=net.mean() / (net.std() / np.sqrt(len(net))) if net.std() > 0 else 0.0,
        ann=net.sum() / 1e4 / years * 100,
        sharpe=daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else 0.0,
        pos_months=(monthly > 0).mean() * 100, mdd=(eq.cummax() - eq).max() * 100,
        long_ann=net[side > 0].sum() / 1e4 / years * 100, short_ann=net[side < 0].sum() / 1e4 / years * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="research/swing_out")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    m = load(args.paths)
    H, L, C = (m[k].to_numpy() for k in ("high", "low", "close"))
    b = bars(m, TF)
    f = features(b, TF)
    feat_cols = list(f.columns)
    sigs = signals(b)
    vol = f["vol_day"].to_numpy()
    end = b["end"].to_numpy()

    rows, trades_store = [], {}
    for sname, sig in sigs.items():
        idx = np.where((sig.to_numpy() != 0) & ~np.isnan(vol))[0]
        side = sig.to_numpy()[idx].astype(np.int64)
        entry = np.minimum(end[idx] + 1, len(C) - 1)
        times = b.index[idx]
        X = f.to_numpy()[idx].copy()
        # Направленные признаки умножаем на сторону сделки: модель видит «по ходу/против хода».
        for j, col in enumerate(feat_cols):
            if col.startswith(("ret_", "ema_", "upper", "lower", "skew")) or col in ("rsi", "donchian_pos"):
                X[:, j] = X[:, j] * side if col not in ("rsi", "donchian_pos") else \
                    (X[:, j] - (50 if col == "rsi" else 0.5)) * side
        X = np.column_stack([X, side])
        for hold in HOLDS:
            sigma_h = vol[idx] * np.sqrt(hold / TF)
            for bname, (ktp, ksl) in BARRIERS.items():
                net, exit_idx, kind = simulate(H, L, C, entry, side, ktp * sigma_h, ksl * sigma_h, hold,
                                               ENTRY_BPS, TP_BPS, SL_BPS, TIME_BPS)
                exit_times = m.index[exit_idx]
                variants = {"все сигналы": (np.ones(len(net), bool), times >= times[0] + pd.DateOffset(years=2))}
                for label, shuf in (("ML-фильтр", False), ("контроль (перемешано)", True)):
                    variants[label] = meta_filter(X, net, times, hold // TF, shuffle=shuf)
                for vname, (take, tested) in variants.items():
                    chosen = sequential(entry, exit_idx, take & tested)
                    for period, pmask in (("2022–2025", times.year < HOLDOUT_YEAR), ("2026 (экзамен)", times.year == HOLDOUT_YEAR)):
                        sel = chosen & pmask
                        span = tested & pmask
                        days = (times[span].max() - times[span].min()).days + 1 if span.any() else 1
                        r = metrics(net[sel], times[sel], exit_times[sel], side[sel], days)
                        r.update(signal=sname, hold_min=hold, barriers=bname, variant=vname, period=period,
                                 tp_share=(kind[sel] == 1).mean() * 100 if sel.any() else np.nan,
                                 sl_share=(kind[sel] == -1).mean() * 100 if sel.any() else np.nan)
                        rows.append(r)
                    key = (sname, hold, bname, vname)
                    trades_store[key] = pd.DataFrame({"entry": times[chosen], "exit": exit_times[chosen],
                                                      "side": side[chosen], "net_bps": net[chosen],
                                                      "kind": kind[chosen]})
        print(f"done {sname}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "swing_results.csv", index=False)
    pd.to_pickle(trades_store, out / "trades.pkl")

    # Буй-энд-холд на тех же периодах для сравнения.
    cl = b.c
    for period, mask in (("2022–2025", (cl.index.year >= 2022) & (cl.index.year < HOLDOUT_YEAR)),
                         ("2026 (экзамен)", cl.index.year == HOLDOUT_YEAR)):
        x = cl[mask]
        yrs = (x.index[-1] - x.index[0]).days / 365
        print(f"Купил и держи {period}: {(x.iloc[-1] / x.iloc[0] - 1) * 100:+.1f}% всего, "
              f"{np.log(x.iloc[-1] / x.iloc[0]) / yrs * 100:+.1f}% лог/год")


if __name__ == "__main__":
    main()
