"""Исследование: можно ли зарабатывать на таймфреймах 5–15 минут.

Данные: минутные свечи BTC/USD с Bitstamp
(https://github.com/ff137/bitstamp-btcusd-minute-data).

  python research/timeframe_study.py HIST.csv.gz LATEST.csv > research/REPORT.md

Методика без подглядывания в будущее:
  * сигнал считается по закрытию бара t, сделка исполняется по цене через
    `delay` минут после закрытия, позиция держится до такого же момента бара t+1;
  * параметры выбираются walk-forward: лучший набор по прошлому году
    проверяется на следующем, который при подборе не использовался;
  * комиссия списывается с каждого изменения позиции.
"""
from __future__ import annotations

import itertools
import sys

import numpy as np
import pandas as pd

START = "2020-01-01"
TIMEFRAMES = [1, 5, 15, 60]
# Издержки на одну сторону сделки, б.п.
COSTS = {"без комиссий": 0.0, "мейкер 2 б.п.": 2.0, "тейкер 5.5 б.п. + 1 б.п. проскальз.": 6.5}
MIN_PER_YEAR = 365 * 24 * 60


def load(paths: list[str]) -> pd.DataFrame:
    """Минутные свечи с 2020 года (в источнике каждая минута на месте, пропуски заполнены)."""
    frames = [pd.read_csv(p, usecols=["timestamp", "open", "high", "low", "close"]) for p in paths]
    df = pd.concat(frames).drop_duplicates("timestamp").sort_values("timestamp")
    df.index = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df.loc[START:, ["open", "high", "low", "close"]].astype(float)


# --------------------------------------------------------------------- бары
def bars(m: pd.DataFrame, tf: int) -> pd.DataFrame:
    """Бары tf минут и индекс их последней минуты в минутном массиве."""
    n = len(m)
    starts = np.arange(0, n, tf)
    end = np.minimum(starts + tf - 1, n - 1)
    b = pd.DataFrame({
        "o": m["open"].to_numpy()[starts],
        "h": np.maximum.reduceat(m["high"].to_numpy(), starts),
        "l": np.minimum.reduceat(m["low"].to_numpy(), starts),
        "c": m["close"].to_numpy()[end],
        "end": end,
    }, index=m.index[end])
    return b


# --------------------------------------------------------------- стратегии
def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False).mean()


def hold_until(enter_long, enter_short, exit_long, exit_short) -> pd.Series:
    """Состояние позиции из событий входа и выхода (через ffill, без циклов)."""
    ev = pd.Series(np.nan, index=enter_long.index)
    ev[exit_long | exit_short] = 0.0
    ev[enter_long] = 1.0
    ev[enter_short] = -1.0
    return ev.ffill().fillna(0.0)


def s_ema_cross(b, fast, slow):
    return np.sign(ema(b.c, fast) - ema(b.c, slow))


def s_donchian(b, n):
    hi = b.h.rolling(n).max().shift(1)
    lo = b.l.rolling(n).min().shift(1)
    up, dn = b.c > hi, b.c < lo
    none = pd.Series(False, index=b.index)
    return hold_until(up, dn, none, none)


def s_momentum(b, n):
    return np.sign(b.c / b.c.shift(n) - 1).fillna(0.0)


def s_rsi_revert(b, n, band):
    d = b.c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    prev = rsi.shift(1)
    cross = ((rsi >= 50) & (prev < 50)) | ((rsi <= 50) & (prev > 50))
    return hold_until(rsi < 50 - band, rsi > 50 + band, cross, cross)


def s_boll_revert(b, n, k):
    m = b.c.rolling(n).mean()
    z = (b.c - m) / b.c.rolling(n).std()
    prev = z.shift(1)
    cross = ((z >= 0) & (prev < 0)) | ((z <= 0) & (prev > 0))
    return hold_until(z < -k, z > k, cross, cross)


STRATEGIES = {
    "EMA-кросс (тренд)": (s_ema_cross, [dict(fast=f, slow=s) for f, s in
                                        itertools.product([5, 10, 20, 50], [50, 100, 200, 400]) if f < s]),
    "Пробой Дончиана": (s_donchian, [dict(n=n) for n in [10, 20, 50, 100, 200, 400]]),
    "Моментум": (s_momentum, [dict(n=n) for n in [3, 6, 12, 24, 48, 96, 192]]),
    "RSI возврат к среднему": (s_rsi_revert, [dict(n=n, band=bd) for n, bd in
                                              itertools.product([7, 14, 28], [15, 20, 25, 30])]),
    "Боллинджер возврат к среднему": (s_boll_revert, [dict(n=n, k=k) for n, k in
                                                      itertools.product([20, 50, 100], [1.5, 2.0, 2.5, 3.0])]),
}


# --------------------------------------------------------------- бэктест
def run(close_1m: np.ndarray, b: pd.DataFrame, pos: pd.Series, delay: int, cost_bps: float) -> pd.Series:
    """Доходность стратегии по барам (в долях), с задержкой исполнения и комиссией."""
    idx = np.minimum(b["end"].to_numpy() + delay, len(close_1m) - 1)
    px = close_1m[idx]
    p = pos.to_numpy()
    ret = np.zeros(len(p))
    ret[1:] = p[:-1] * (px[1:] / px[:-1] - 1)
    turnover = np.abs(np.diff(np.concatenate([[0.0], p])))
    cost = np.zeros(len(p))
    cost[1:] = turnover[:-1] * cost_bps / 1e4  # позиция из бара t открывается в бар t+1
    return pd.Series(ret - cost, index=b.index)


def stats(r: pd.Series, pos: pd.Series, tf: int) -> dict:
    if len(r) == 0:
        return {}
    per_year = MIN_PER_YEAR / tf
    ann = r.mean() * per_year
    sd = r.std() * np.sqrt(per_year)
    eq = r.cumsum()
    trades = (pos.diff().abs() > 0).sum() / 2  # вход + выход = одна сделка
    years = len(r) / per_year
    return dict(ann=ann, sharpe=ann / sd if sd > 0 else 0.0, mdd=(eq.cummax() - eq).max(),
                trades_y=trades / years if years else 0.0,
                edge_bps=r.sum() / trades * 1e4 if trades else 0.0)


def walk_forward(close_1m, b, tf, cost_bps, fn, grid, delay=0):
    """Подбор параметров на году Y-1, проверка на году Y. Возвращает OOS-доходности и выбор."""
    positions = [fn(b, **kw) for kw in grid]
    rets = [run(close_1m, b, p, delay, cost_bps) for p in positions]
    years = sorted(set(b.index.year))
    oos, chosen = [], []
    for y in years[1:]:
        train = [r[r.index.year == y - 1] for r in rets]
        best = int(np.argmax([t.mean() / t.std() if t.std() > 0 else -1 for t in train]))
        chosen.append((y, grid[best]))
        oos.append((rets[best][rets[best].index.year == y], positions[best][positions[best].index.year == y]))
    r = pd.concat([o[0] for o in oos])
    p = pd.concat([o[1] for o in oos])
    # Для сравнения: «лучший задним числом» на всём периоде (то, что обычно показывают в рекламе).
    full = [x.mean() / x.std() if x.std() > 0 else -1 for x in rets]
    hindsight = int(np.argmax(full))
    return r, p, chosen, (rets[hindsight], positions[hindsight], grid[hindsight])


# --------------------------------------------------------------- отчёт
def fmt_pct(x):
    return f"{x * 100:+.1f}%"


def main(paths):
    m = load(paths)
    close = m["close"]
    c1 = close.to_numpy()
    out = []
    w = out.append
    w("# Таймфреймы 5–15 минут: исследование на реальных данных BTC/USD\n")
    w(f"Данные: Bitstamp, минутные свечи с {close.index[0]:%Y-%m-%d} по {close.index[-1]:%Y-%m-%d} "
      f"({len(close):,} минут).\n")

    # 1. размер движения против издержек
    w("## 1. Сколько цена проходит за бар и сколько стоит сделка\n")
    w("| Таймфрейм | Медиана \\|движения\\| за бар | Среднее \\|движения\\| | Круг тейкером (13 б.п.) в медианах | Круг мейкером (4 б.п.) в медианах |")
    w("|---|---|---|---|---|")
    bars_by_tf = {tf: bars(m, tf) for tf in TIMEFRAMES}
    for tf, b in bars_by_tf.items():
        r = np.abs(np.log(b.c).diff().dropna()) * 1e4
        med, mean = np.median(r), r.mean()
        w(f"| {tf} мин | {med:.1f} б.п. | {mean:.1f} б.п. | {13 / med:.2f} | {4 / med:.2f} |")
    w("")

    # 2. предсказуемость
    w("## 2. Есть ли в ценах закономерность\n")
    w("Автокорреляция доходностей соседних баров по годам (0 = случайное блуждание, "
      "плюс = продолжение движения, минус = откат):\n")
    years = sorted(set(close.index.year))
    w("| Таймфрейм | " + " | ".join(str(y) for y in years) + " |")
    w("|---|" + "---|" * len(years))
    for tf in (5, 15, 60):
        b = bars_by_tf[tf]
        r = np.log(b.c).diff()
        vals = [r[r.index.year == y].autocorr(1) for y in years]
        w(f"| {tf} мин | " + " | ".join(f"{v:+.3f}" for v in vals) + " |")
    w("\nVariance ratio VR(q) на 5-минутных барах (1 = случайное блуждание, "
      ">1 = тренд, <1 = возврат к среднему):\n")
    r5 = np.log(bars_by_tf[5].c).diff().dropna()
    w("| q (баров) | " + " | ".join(str(q) for q in (2, 3, 6, 12, 48, 288)) + " |")
    w("|---|" + "---|" * 6)
    vr = [r5.rolling(q).sum().var() / (q * r5.var()) for q in (2, 3, 6, 12, 48, 288)]
    w("| VR | " + " | ".join(f"{v:.3f}" for v in vr) + " |\n")

    # 3. стратегии walk-forward
    w("## 3. Классические стратегии, честная проверка (walk-forward)\n")
    w("Параметры каждый год выбираются по предыдущему году, результат считается только на годах, "
      "которые при подборе не использовались. Плечо 1x, доходность без реинвестирования.\n")
    best_cases = []
    for tf in (5, 15, 60):
        b = bars_by_tf[tf]
        w(f"### Таймфрейм {tf} минут\n")
        w("| Стратегия | Издержки | Доходность/год | Шарп | Макс. просадка | Сделок/год | Средняя сделка | "
          "«Лучший задним числом»: доходность/год |")
        w("|---|---|---|---|---|---|---|---|")
        for name, (fn, grid) in STRATEGIES.items():
            for cname, cost in COSTS.items():
                r, p, chosen, (hr, hp, hparams) = walk_forward(c1, b, tf, cost, fn, grid)
                s, hs = stats(r, p, tf), stats(hr, hp, tf)
                w(f"| {name} | {cname} | {fmt_pct(s['ann'])} | {s['sharpe']:+.2f} | {s['mdd'] * 100:.0f}% | "
                  f"{s['trades_y']:.0f} | {s['edge_bps']:+.1f} б.п. | {fmt_pct(hs['ann'])} |")
                if cost == 0.0:
                    best_cases.append((s["sharpe"], tf, name, fn, grid))
        w("")

    # 4. влияние задержки
    w("## 4. Правда ли, что на 5–15 минутах время не важно\n")
    w("Те же стратегии без комиссий, но исполнение через d минут после закрытия бара. "
      "Показана средняя сделка (б.п.) по walk-forward.\n")
    delays = [0, 1, 2, 5]
    w("| Таймфрейм | Стратегия | " + " | ".join(f"d={d} мин" for d in delays) + " |")
    w("|---|---|" + "---|" * len(delays))
    for tf in (1, 5, 15):
        b = bars_by_tf[tf]
        for name in ("Моментум", "Боллинджер возврат к среднему", "Пробой Дончиана"):
            fn, grid = STRATEGIES[name]
            vals = []
            for d in delays:
                r, p, _, _ = walk_forward(c1, b, tf, 0.0, fn, grid, delay=d)
                vals.append(stats(r, p, tf)["edge_bps"])
            w(f"| {tf} мин | {name} | " + " | ".join(f"{v:+.1f}" for v in vals) + " |")
    w("")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1:])
