"""Машинное обучение для прогноза BTC на 5–60 минут: градиентный бустинг и нейросеть.

  python research/ml_study.py HIST.csv.gz LATEST.csv --out research/ml_out

Защита от самообмана:
  * все признаки считаются только по прошлым данным (rolling, shift);
  * walk-forward: модель учится на 2 годах, торгует следующие 3 месяца, затем
    переобучается; между обучением и тестом есть зазор (embargo) длиной
    в горизонт прогноза, чтобы метка не «протекала» в тест;
  * порог входа подбирается на валидационном хвосте обучающей выборки, не на тесте;
  * исполнение через 1 минуту после закрытия бара, комиссия с каждого изменения позиции;
  * контроль: та же процедура с перемешанной целевой переменной должна давать ноль.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

START = "2020-01-01"
MIN_PER_YEAR = 365 * 24 * 60
COSTS = {"maker": 2.0, "taker": 6.5}  # б.п. на сторону (тейкер включает 1 б.п. проскальзывания)
torch.set_num_threads(4)


def load(paths, start=START):
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames).drop_duplicates("timestamp").sort_values("timestamp")
    df.index = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df.loc[start:, ["open", "high", "low", "close", "volume"]].astype(float)


def bars(m: pd.DataFrame, tf: int) -> pd.DataFrame:
    n = len(m)
    starts = np.arange(0, n, tf)
    end = np.minimum(starts + tf - 1, n - 1)
    return pd.DataFrame({
        "o": m["open"].to_numpy()[starts],
        "h": np.maximum.reduceat(m["high"].to_numpy(), starts),
        "l": np.minimum.reduceat(m["low"].to_numpy(), starts),
        "c": m["close"].to_numpy()[end],
        "v": np.add.reduceat(m["volume"].to_numpy(), starts),
        "end": end,
    }, index=m.index[end])


# ------------------------------------------------------------------ признаки
def features(b: pd.DataFrame, tf: int) -> pd.DataFrame:
    c, h, l, v = np.log(b.c), np.log(b.h), np.log(b.l), b.v
    r = c.diff()
    day = int(24 * 60 / tf)
    f = pd.DataFrame(index=b.index)
    vol = r.rolling(day).std()
    f["vol_day"] = vol
    for k in (1, 2, 4, 8, 16, 32, day, 3 * day):
        f[f"ret_{k}"] = (c - c.shift(k)) / (vol * np.sqrt(k))  # доходность в единицах волатильности
    for k in (4, 16):
        f[f"vol_{k}_ratio"] = r.rolling(k).std() / vol
    f["range"] = (h - l) / vol
    f["range_ratio"] = (h - l).rolling(4).mean() / (h - l).rolling(day).mean()
    f["upper_wick"] = (h - np.maximum(c, np.log(b.o))) / vol
    f["lower_wick"] = (np.minimum(c, np.log(b.o)) - l) / vol
    lv = np.log1p(v)
    f["vol_z"] = (lv - lv.rolling(day).mean()) / lv.rolling(day).std()
    f["vol_trend"] = lv.rolling(4).mean() - lv.rolling(day).mean()
    for n in (20, 100, 400):
        f[f"ema_{n}"] = (c - c.ewm(span=n, adjust=False).mean()) / vol
    d = r
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    hi, lo = h.rolling(day).max(), l.rolling(day).min()
    f["donchian_pos"] = (c - lo) / (hi - lo)
    f["skew_day"] = r.rolling(day).skew()
    f["autocorr"] = (r * r.shift(1)).rolling(day).mean() / r.rolling(day).var()
    hour = b.index.hour + b.index.minute / 60
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f["dow"] = b.index.dayofweek
    f["vol_regime"] = vol / vol.rolling(30 * day).median()
    return f.replace([np.inf, -np.inf], np.nan)


def target(b: pd.DataFrame, c1: np.ndarray, horizon: int, delay: int = 1) -> tuple[pd.Series, np.ndarray]:
    """Будущая лог-доходность от (закрытие + delay мин) на horizon баров вперёд."""
    idx = np.minimum(b["end"].to_numpy() + delay, len(c1) - 1)
    px = c1[idx]
    fwd = np.full(len(px), np.nan)
    fwd[:-horizon] = np.log(px[horizon:] / px[:-horizon])
    step = np.full(len(px), np.nan)  # доходность удержания на один бар (для бэктеста)
    step[:-1] = px[1:] / px[:-1] - 1
    return pd.Series(fwd, index=b.index), step


# ------------------------------------------------------------------ модели
def fit_lgbm(X, y, Xv, yv, seed=0):
    m = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=15, min_child_samples=500,
                          subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=10.0,
                          random_state=seed, verbose=-1, n_jobs=4)
    m.fit(X, y, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(100, verbose=False)])
    return m.predict


class MLP(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n, 64), torch.nn.GELU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 32), torch.nn.GELU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_mlp(X, y, Xv, yv, seed=0):
    torch.manual_seed(seed)
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-9
    prep = lambda a: torch.tensor(np.nan_to_num((a - mu) / sd).clip(-6, 6), dtype=torch.float32)
    Xt, Xvt = prep(X), prep(Xv)
    ys = y.std()
    yt, yvt = torch.tensor(y / ys, dtype=torch.float32), torch.tensor(yv / ys, dtype=torch.float32)
    model = MLP(X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best, best_state, bad = np.inf, None, 0
    for _ in range(60):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 1024):
            bi = perm[i:i + 1024]
            opt.zero_grad()
            loss = torch.nn.functional.huber_loss(model(Xt[bi]), yt[bi])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = torch.nn.functional.huber_loss(model(Xvt), yvt).item()
        if vl < best - 1e-5:
            best, best_state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 6:
                break
    model.load_state_dict(best_state)
    model.eval()

    def predict(A):
        with torch.no_grad():
            return model(prep(A)).numpy() * ys
    return predict


# ------------------------------------------------------------------ торговля
def positions(pred: np.ndarray, thr: float) -> np.ndarray:
    return np.where(pred > thr, 1.0, np.where(pred < -thr, -1.0, 0.0))


def pnl(pos: np.ndarray, step: np.ndarray, cost_bps: float) -> np.ndarray:
    turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))
    return pos * np.nan_to_num(step) - turnover * cost_bps / 1e4


def pick_threshold(pred_val, step_val, cost_bps):
    """Порог входа по валидации: максимизируем сумму PnL; 'не торговать' тоже допустимо."""
    qs = np.quantile(np.abs(pred_val), [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99])
    best_thr, best = np.inf, 0.0
    for q in qs:
        s = pnl(positions(pred_val, q), step_val, cost_bps).sum()
        if s > best:
            best, best_thr = s, q
    return best_thr


def walk_forward(b, f, y, step, horizon, tf, models, shuffle=False, train_years=2, test_months=3, seed=0):
    X = f.to_numpy(dtype=np.float64)
    yv = y.to_numpy()
    idx = b.index
    out = {name: {c: np.zeros(len(b)) for c in COSTS} for name in models}
    out_pos = {name: {c: np.zeros(len(b)) for c in COSTS} for name in models}
    preds = {name: np.full(len(b), np.nan) for name in models}
    tested = np.zeros(len(b), dtype=bool)
    rng = np.random.default_rng(seed)
    vol = f["vol_day"].to_numpy()
    t0 = idx[0] + pd.DateOffset(years=train_years)
    folds = []
    while t0 < idx[-1]:
        t1 = t0 + pd.DateOffset(months=test_months)
        tr = np.where((idx >= t0 - pd.DateOffset(years=train_years)) & (idx < t0))[0]
        tr = tr[:-horizon]  # embargo: метка последних баров обучения заходит в тест
        tr = tr[~np.isnan(yv[tr]) & ~np.isnan(vol[tr])]
        te = np.where((idx >= t0) & (idx < t1))[0]
        if len(te) == 0 or len(tr) < 1000:
            break
        cut = int(len(tr) * 0.8)
        fit_i, val_i = tr[:cut - horizon], tr[cut:]
        y_fit = yv[fit_i].copy()
        if shuffle:
            rng.shuffle(y_fit)
        for name, fit in models.items():
            # Цель нормируем на волатильность: так модель учится на сопоставимых величинах.
            predict = fit(X[fit_i], y_fit / vol[fit_i], X[val_i], yv[val_i] / vol[val_i], seed)
            pv = predict(X[val_i]) * vol[val_i]
            pt = predict(X[te]) * vol[te]
            preds[name][te] = pt
            for c, cost in COSTS.items():
                thr = pick_threshold(pv, step[val_i], cost)
                pos = positions(pt, thr)
                out_pos[name][c][te] = pos
                out[name][c][te] = pnl(pos, step[te], cost)
        tested[te] = True
        folds.append((str(t0.date()), len(tr), len(te)))
        t0 = t1
    return out, out_pos, preds, tested, folds


def summarize(r: pd.Series, pos: pd.Series, tf: int) -> dict:
    per_year = MIN_PER_YEAR / tf
    ann = r.mean() * per_year
    sd = r.std() * np.sqrt(per_year)
    eq = r.cumsum()
    monthly = r.groupby([r.index.year, r.index.month]).sum()
    active = monthly[pos.abs().groupby([pos.index.year, pos.index.month]).sum() > 0]
    trades = (pos.diff().abs() > 0).sum() / 2
    return dict(ann_pct=ann * 100, sharpe=ann / sd if sd > 0 else 0.0,
                max_dd_pct=(eq.cummax() - eq).max() * 100,
                pos_months_pct=(monthly > 0).mean() * 100,
                active_months=int(len(active)),
                pos_active_months_pct=(active > 0).mean() * 100 if len(active) else 0.0,
                trades_per_year=trades / (len(r) / per_year),
                exposure_pct=(pos != 0).mean() * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="research/ml_out")
    ap.add_argument("--start", default=START)
    ap.add_argument("--configs", default="15:4,15:1,5:3,60:1")  # таймфрейм:горизонт в барах
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    m = load(args.paths, args.start)
    c1 = m["close"].to_numpy()
    results = []
    for cfg in args.configs.split(","):
        tf, horizon = map(int, cfg.split(":"))
        t = time.time()
        b = bars(m, tf)
        f = features(b, tf)
        y, step = target(b, c1, horizon)
        bh = pd.Series(np.nan_to_num(step), index=b.index)
        for shuffle in (False, True):
            models = {"LightGBM": fit_lgbm, "Нейросеть (MLP)": fit_mlp}
            out, out_pos, preds, tested, folds = walk_forward(b, f, y, step, horizon, tf, models, shuffle)
            tag = "контроль (перемешанная цель)" if shuffle else "реальная"
            # Ансамбль: среднее прогнозов двух моделей с тем же порогом-правилом не повторяем,
            # а считаем только информационный коэффициент.
            for name in models:
                mask = tested & ~np.isnan(preds[name]) & ~np.isnan(y.to_numpy())
                ic = spearmanr(preds[name][mask], y.to_numpy()[mask]).statistic
                # IC по кварталам: насколько стабилен сигнал
                q = pd.Series(np.arange(mask.sum()), index=b.index[mask])
                qic = []
                for _, g in q.groupby([q.index.year, q.index.quarter]):
                    ii = np.where(mask)[0][g.to_numpy()]
                    qic.append(spearmanr(preds[name][ii], y.to_numpy()[ii]).statistic)
                for c in COSTS:
                    r = pd.Series(out[name][c][tested], index=b.index[tested])
                    p = pd.Series(out_pos[name][c][tested], index=b.index[tested])
                    s = summarize(r, p, tf)
                    s.update(tf=tf, horizon_min=tf * horizon, model=name, target=tag, cost=c,
                             ic=float(ic), ic_quarters_pos_pct=float(np.mean(np.array(qic) > 0) * 100))
                    results.append(s)
                    if not shuffle:
                        r.cumsum().to_frame("equity").assign(position=p).iloc[::max(1, 60 // tf)].to_csv(
                            out_dir / f"equity_tf{tf}_h{horizon}_{name.split()[0]}_{c}.csv")
        bh_t = bh[tested]
        results.append(dict(tf=tf, horizon_min=tf * horizon, model="Купил и держи", target="—", cost="—",
                            **summarize(bh_t, pd.Series(1.0, index=bh_t.index), tf), ic=np.nan,
                            ic_quarters_pos_pct=np.nan))
        print(f"tf={tf} h={horizon}: {time.time() - t:.0f}s, folds={len(folds)}", flush=True)
        pd.DataFrame(results).to_csv(out_dir / "results.csv", index=False)
    json.dump({"folds_last_config": folds}, open(out_dir / "folds.json", "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
