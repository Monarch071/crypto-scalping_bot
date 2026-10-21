"""Бот, который выбирает монеты из рынка: кросс-секционные стратегии с удержанием от часов до суток.

  python research/xs_study.py PANEL.parquet --out research/xs_out

PANEL.parquet: часовые свечи многих монет с колонками symbol, timestamp, open, high, low, close, volume
(например, https://github.com/Speirsy11/crypto-dataset или research/download_binance.py --interval 1h).

Каждые R часов (4, 8, 24) бот ранжирует монеты по сигналу и держит лучшие до следующего пересмотра.
  * сигналы: моментум и разворот на разных окнах, моментум относительно BTC, всплеск объёма,
    моментум с поправкой на волатильность, ML-ранжирование (LightGBM) на всех признаках;
  * портфели: лонг k лучших; лонг k лучших + шорт k худших (нейтрально к рынку);
    лонг k лучших только при восходящем тренде BTC (цена выше EMA за 30 дней);
  * издержки: 8.5 б.п. на единицу оборота (тейкер + проскальзывание на альткоинах);
  * сигнал по закрытию часа t, позиция с часа t+1 — без заглядывания вперёд;
  * выбор конфигурации walk-forward по двум прошлым годам (2022–2025), 2026 — экзамен.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

COST_BPS = 8.5
HPY = 365 * 24
TEST_YEARS = (2022, 2023, 2024, 2025)
HOLDOUT = 2026
MIN_HISTORY_H = 24 * 60


def load_panel(path: str):
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    piv = {k: df.pivot_table(index="timestamp", columns="symbol", values=k) for k in ("close", "volume")}
    idx = pd.date_range(piv["close"].index.min(), piv["close"].index.max(), freq="h")
    close = piv["close"].reindex(idx).ffill(limit=3)
    volume = piv["volume"].reindex(idx).fillna(0.0)
    # Монета участвует в отборе, только когда у неё есть 60 дней истории.
    alive = close.notna() & (close.notna().cumsum() > MIN_HISTORY_H)
    return close.loc["2020-06-01":], volume.loc["2020-06-01":], alive.loc["2020-06-01":]


# ------------------------------------------------------------------ сигналы
def features(close: pd.DataFrame, volume: pd.DataFrame) -> dict[str, pd.DataFrame]:
    lr = np.log(close).diff()
    vol = lr.rolling(24 * 7, min_periods=24 * 3).std()
    btc = lr["BTCUSDT"]
    beta = lr.rolling(24 * 30, min_periods=24 * 10).cov(btc).div(btc.rolling(24 * 30, min_periods=24 * 10).var(), axis=0)
    f = {}
    for L in (4, 24, 72, 168, 336):
        mom = np.log(close / close.shift(L))
        f[f"mom_{L}"] = mom
        f[f"mom_voladj_{L}"] = mom / (vol * np.sqrt(L))
        f[f"resid_{L}"] = mom - beta.mul(np.log(close["BTCUSDT"] / close["BTCUSDT"].shift(L)), axis=0)
    lv = np.log1p(volume)
    f["volume_shock"] = lv.rolling(24).mean() - lv.rolling(24 * 30).mean()
    f["vol_ratio"] = lr.rolling(24).std() / vol
    f["dist_ema168"] = np.log(close / close.ewm(span=168, adjust=False).mean()) / vol
    f["beta"] = beta
    return f


def rule_signals(f):
    s = {}
    for L in (24, 72, 168, 336):
        s[f"Моментум {L}ч"] = f[f"mom_{L}"]
        s[f"Моментум отн. BTC {L}ч"] = f[f"resid_{L}"]
        s[f"Моментум с поправкой на волатильность {L}ч"] = f[f"mom_voladj_{L}"]
    for L in (4, 24, 72):
        s[f"Разворот {L}ч"] = -f[f"mom_{L}"]
    s["Всплеск объёма × направление 24ч"] = f["volume_shock"] * np.sign(f["mom_24"])
    return s


def ml_signal(f, close, alive, R, seed=0):
    """LightGBM предсказывает будущую относительную доходность монеты за R часов (walk-forward)."""
    names = list(f)
    t_idx = close.index[::R]
    fwd = np.log(close.shift(-R) / close)
    rows = []
    for t in t_idx:
        a = alive.loc[t]
        if a.sum() < 4:
            continue
        X = pd.DataFrame({n: f[n].loc[t] for n in names})[a]
        # Признаки в ранги внутри момента времени: модель сравнивает монеты между собой.
        X = X.rank(pct=True)
        y = fwd.loc[t][a]
        y = y - y.mean()
        X["t"], X["y"], X["sym"] = t, y, X.index
        rows.append(X)
    D = pd.concat(rows).dropna(subset=["y"])
    score = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    start = pd.Timestamp(f"{TEST_YEARS[0]}-01-01", tz="UTC")
    t0 = start
    while t0 <= close.index[-1]:
        t1 = t0 + pd.DateOffset(months=3)
        tr = D[(D.t >= t0 - pd.DateOffset(years=2)) & (D.t < t0 - pd.Timedelta(hours=R))]
        te = D[(D.t >= t0) & (D.t < t1)]
        if len(te) and len(tr) > 500:
            m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=100,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
                                  random_state=seed, verbose=-1, n_jobs=4)
            m.fit(tr[names], tr["y"])
            p = m.predict(te[names])
            for (t, sym), v in zip(te[["t", "sym"]].itertuples(index=False), p):
                score.at[t, sym] = v
        t0 = t1
    return score


# ------------------------------------------------------------------ портфель
def weights(score: pd.DataFrame, alive: pd.DataFrame, R: int, k: int, mode: str, btc_up: pd.Series):
    rows = score.where(alive).iloc[::R]
    n = rows.notna().sum(axis=1).to_numpy()[:, None]
    rk = rows.rank(axis=1, method="first").to_numpy()
    valid = n >= 2 * k
    top = (rk > n - k) & valid
    bot = (rk <= k) & valid
    if mode == "long/short":
        w = top * (0.5 / k) - bot * (0.5 / k)
    else:
        w = top * (1.0 / k)
        if mode == "long + тренд BTC":
            w = w * btc_up.iloc[::R].to_numpy()[:, None]
    w = pd.DataFrame(w, index=rows.index, columns=score.columns)
    return w.reindex(score.index).ffill().fillna(0.0)


def backtest(close, w):
    r = close.pct_change().fillna(0.0)
    held = w.shift(1).fillna(0.0)
    gross = (held * r).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1))
    return gross - turnover.shift(1).fillna(0.0) * COST_BPS / 1e4, turnover


def stats(ret: pd.Series, turnover: pd.Series) -> dict:
    if len(ret) == 0:
        return {}
    eq = (1 + ret).cumprod()
    yrs = len(ret) / HPY
    daily = (1 + ret).groupby(ret.index.normalize()).prod() - 1
    monthly = (1 + ret).groupby([ret.index.year, ret.index.month]).prod() - 1
    return dict(cagr=(eq.iloc[-1] ** (1 / yrs) - 1) * 100,
                sharpe=daily.mean() / daily.std() * np.sqrt(365) if daily.std() > 0 else 0.0,
                mdd=(1 - eq / eq.cummax()).max() * 100, pos_months=(monthly > 0).mean() * 100,
                turnover_y=turnover.sum() / yrs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panel")
    ap.add_argument("--out", default="research/xs_out")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    close, volume, alive = load_panel(args.panel)
    f = features(close, volume)
    btc = close["BTCUSDT"]
    btc_up = btc > btc.ewm(span=24 * 30, adjust=False).mean()
    years = close.index.year
    sigs = rule_signals(f)
    rets = {}
    for R in (4, 8, 24):
        sigs_R = dict(sigs)
        sigs_R["ML-ранжирование (LightGBM)"] = ml_signal(f, close, alive, R)
        for (sname, score), k, mode in itertools.product(sigs_R.items(), (2, 3),
                                                         ("long", "long/short", "long + тренд BTC")):
            w = weights(score, alive, R, k, mode, btc_up)
            ret, to = backtest(close, w)
            rets[(sname, R, k, mode)] = (ret, to)
        print(f"R={R}: done", flush=True)

    # Бенчмарки: равные доли всех живых монет (ребаланс раз в сутки) и BTC.
    ew = alive.astype(float).div(alive.sum(axis=1), axis=0)
    ew = ew.where(ew.index.hour == 0).ffill().fillna(0.0)
    bench = {"Равные доли всех монет": backtest(close, ew),
             "BTC купил и держи": backtest(close, pd.DataFrame({c: 1.0 if c == "BTCUSDT" else 0.0
                                                               for c in close.columns}, index=close.index))}

    rows = []
    for key, (ret, to) in list(rets.items()) + [((n, "—", "—", "—"), v) for n, v in bench.items()]:
        for period, mask in (("2022–2025", (years >= 2022) & (years < HOLDOUT)), ("2026", years == HOLDOUT)):
            s = stats(ret[mask], to[mask])
            s.update(signal=key[0], R=key[1], k=key[2], mode=key[3], period=period)
            for y in range(2022, 2027):
                s[str(y)] = ((1 + ret[years == y]).prod() - 1) * 100
            rows.append(s)
    res = pd.DataFrame(rows)
    res.to_csv(out / "xs_results.csv", index=False)

    # Walk-forward выбор конфигурации внутри каждого режима портфеля: по Шарпу двух прошлых лет.
    lines = []
    for mode in ("long", "long/short", "long + тренд BTC"):
        keys = [k for k in rets if k[3] == mode]
        parts, chosen = [], []
        for y in list(TEST_YEARS) + [HOLDOUT]:
            tr = (years >= y - 2) & (years < y)
            best = max(keys, key=lambda k: (lambda x: x.mean() / x.std() if x.std() > 0 else -9)(rets[k][0][tr]))
            chosen.append(f"{y}: {best[0]}, R={best[1]}ч, k={best[2]}")
            parts.append((rets[best][0][years == y], rets[best][1][years == y]))
        ret = pd.concat(p[0] for p in parts)
        to = pd.concat(p[1] for p in parts)
        for period, ymask in (("2022–2025", ret.index.year < HOLDOUT), ("2026", ret.index.year == HOLDOUT)):
            s = stats(ret[ymask], to[ymask])
            lines.append(f"WF {mode} {period}: CAGR {s['cagr']:+.1f}% Sharpe {s['sharpe']:.2f} "
                         f"MDD {s['mdd']:.0f}% pos.months {s['pos_months']:.0f}% turnover {s['turnover_y']:.0f}x")
        lines.append("   выбор: " + " | ".join(chosen))
        ret.to_frame("ret").to_parquet(out / f"wf_{mode.replace('/', '_').replace(' ', '')}.parquet")
    (out / "walk_forward.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Проверено конфигураций: {len(rets)}")


if __name__ == "__main__":
    main()
