"""Совпадение многих стратегий: входить, только когда за сделку «голосуют» почти все.

  python research/confluence_study.py PANEL.parquet --out research/confluence_out

15 классических стратегий голосуют за каждую монету каждый час: +1 (покупать), −1 (продавать), 0.
Сигнал — первый час, когда «за» проголосовали не меньше N стратегий (N = 8, 10, 12, 14, 15).
Выход: стоп-лосс и тейк-профит в единицах ATR или через 24 часа.

Реализм для ручной торговли:
  * вход по цене закрытия часа сигнала + 10 б.п. на задержку и проскальзывание ручного входа;
  * если за один час задеты и стоп, и тейк, засчитывается стоп;
  * комиссия: спот Bybit 0.1% за сторону (лонги), фьючерсы 0.055% (шорты);
  * контроль: случайные входы в те же монеты и те же часы суток с теми же стопами;
  * выбор настроек — по 2020–2025, 2026 — экзамен.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from signalbot.indicators import atr, votes  # noqa: F401  (общий код с ботом)

HOLDOUT = 2026
MAX_HOLD_H = 24
SLIP_BPS = 10.0
FEE_LONG_BPS, FEE_SHORT_BPS = 10.0, 5.5
THRESHOLDS = (8, 10, 12, 14, 15)
BARRIERS = {"стоп 1.5 ATR / тейк 3 ATR": (1.5, 3.0), "стоп 2 ATR / тейк 2 ATR": (2.0, 2.0),
            "стоп 2 ATR / тейк 4 ATR": (2.0, 4.0), "стоп 1 ATR / тейк 2 ATR": (1.0, 2.0)}


# ------------------------------------------------------------------ сделки
@njit(cache=True)
def simulate(H, L, C, entries, sides, sl_d, tp_d, max_hold):
    n = len(entries)
    gross = np.empty(n)
    kind = np.empty(n, dtype=np.int8)
    hold = np.empty(n, dtype=np.int64)
    last = len(C) - 1
    for i in range(n):
        e, s = entries[i], sides[i]
        p0 = C[e]
        sl = p0 - s * sl_d[i]
        tp = p0 + s * tp_d[i]
        end = min(e + max_hold, last)
        k, px, x = 0, C[end], end
        for j in range(e + 1, end + 1):
            hit_sl = L[j] <= sl if s > 0 else H[j] >= sl
            hit_tp = H[j] >= tp if s > 0 else L[j] <= tp
            if hit_sl:
                k, px, x = -1, sl, j
                break
            if hit_tp:
                k, px, x = 1, tp, j
                break
        gross[i] = s * (px / p0 - 1) * 1e4
        kind[i] = k
        hold[i] = x - e
    return gross, kind, hold


def first_cross(count: np.ndarray, thr: int) -> np.ndarray:
    on = count >= thr
    prev = np.concatenate([[False], on[:-1]])
    return on & ~prev


def summarize(net, years, times):
    if len(net) < 5:
        return dict(n=len(net), per_year=len(net) / years, win=np.nan, avg=np.nan, t=np.nan, pos_years=np.nan)
    by_year = pd.Series(net, index=times).groupby(times.year).sum()
    return dict(n=len(net), per_year=len(net) / years, win=(net > 0).mean() * 100, avg=net.mean(),
                t=net.mean() / (net.std() / np.sqrt(len(net))), pos_years=(by_year > 0).mean() * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panel")
    ap.add_argument("--out", default="research/confluence_out")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    panel = pd.read_parquet(args.panel)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    frames = {s: g.set_index("timestamp").sort_index() for s, g in panel.groupby("symbol")}
    idx = pd.date_range(pd.Timestamp("2020-01-01", tz="UTC"), panel.timestamp.max(), freq="h")
    btc = frames["BTCUSDT"].close.reindex(idx).ffill()

    trades = []  # все кандидаты: монета, время, сторона, порог, барьер, gross
    rng = np.random.default_rng(0)
    for sym, df in frames.items():
        df = df.reindex(idx).ffill(limit=3).dropna()
        if len(df) < 24 * 60:
            continue
        V = votes(df, btc.reindex(df.index))
        V = V.iloc[24 * 35:]  # прогрев индикаторов (EMA 720 считается по BTC с 2020 года)
        H, L, C = (df[k].reindex(V.index).to_numpy() for k in ("high", "low", "close"))
        a = atr(df.high, df.low, df.close).reindex(V.index).to_numpy()
        longs, shorts = (V.to_numpy() > 0).sum(1), (V.to_numpy() < 0).sum(1)
        for thr in THRESHOLDS:
            for side, cnt in ((1, longs), (-1, shorts)):
                ent = np.where(first_cross(cnt, thr) & ~np.isnan(a))[0]
                ent = ent[ent < len(C) - 2]
                # Контроль: столько же случайных входов в эту монету с тем же распределением по годам.
                rnd = np.sort(rng.choice(np.where(~np.isnan(a))[0][:-2], size=len(ent), replace=False)) \
                    if len(ent) else ent
                for label, e in (("сигнал", ent), ("случайный вход", rnd)):
                    if len(e) == 0:
                        continue
                    for bname, (ks, kt) in BARRIERS.items():
                        g, kind, hold = simulate(H, L, C, e, np.full(len(e), side, np.int64),
                                                 ks * a[e], kt * a[e], MAX_HOLD_H)
                        trades.append(pd.DataFrame({"sym": sym, "time": V.index[e], "side": side, "thr": thr,
                                                    "barrier": bname, "kind": label, "gross": g,
                                                    "exit": kind, "hold_h": hold}))
        print(f"{sym}: ok", flush=True)

    T = pd.concat(trades, ignore_index=True)
    fee = np.where(T.side > 0, FEE_LONG_BPS, FEE_SHORT_BPS)
    T["net"] = T.gross - 2 * fee - SLIP_BPS
    T.to_parquet(out / "trades.parquet")

    rows = []
    span_in = (pd.Timestamp("2020-03-01", tz="UTC"), pd.Timestamp(f"{HOLDOUT}-01-01", tz="UTC"))
    for (thr, side, bname, kind), g in T.groupby(["thr", "side", "barrier", "kind"]):
        for period, m in (("2020–2025", g.time < span_in[1]), ("2026", g.time >= span_in[1])):
            years = (span_in[1] - span_in[0]).days / 365 if period == "2020–2025" else 9.6 / 12
            s = summarize(g.net[m].to_numpy(), years, pd.DatetimeIndex(g.time[m]))
            s.update(thr=thr, side="лонг" if side > 0 else "шорт", barrier=bname, kind=kind, period=period,
                     tp_share=(g.exit[m] == 1).mean() * 100, hold_med=g.hold_h[m].median())
            rows.append(s)
    R = pd.DataFrame(rows)
    R.to_csv(out / "results.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    cols = ["thr", "side", "barrier", "kind", "n", "per_year", "win", "avg", "t", "pos_years", "tp_share", "hold_med"]
    print(R[(R.period == "2020–2025")][cols].sort_values(["side", "thr", "barrier", "kind"]).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
