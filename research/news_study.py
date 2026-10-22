"""Новости как 16-я стратегия: предсказывает ли тон новостей движение цены?

  python research/news_study.py ARTICLES.parquet PANEL.parquet

Новости: https://github.com/Speirsy11/crypto-news-dataset (заголовки с оценкой тона FinBERT, время
публикации). Плотные данные есть только с апреля по июль 2026 года, поэтому проверка короткая.

План зафиксирован до запуска:
  1. Тон новостей о монете и общий фон рынка за последние 24 ч против доходности монеты за следующие
     4 и 24 ч (ранговая корреляция, доверительный интервал бутстрэпом по дням).
  2. То же против доходности за ПРОШЛЫЕ 24 ч: если связь с прошлым сильнее, новости в основном
     описывают уже случившееся движение.
  3. Новостной фон как 16-й голос: сигналы 15/15 (итоговые правила) в этом периоде при позитивном
     и при негативном фоне.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
START, END = "2026-04-01", "2026-07-09"
COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "TRX", "DOGE", "ZEC", "ADA", "BCH"]


def sentiment_hourly(art: pd.DataFrame, assets: list[str], idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Средний тон и число новостей за последние 24 ч на конец каждого часа (только прошлые новости)."""
    a = art[art.asset.isin(assets)].copy()
    a["hour"] = a.published_at.dt.ceil("h")  # новость, вышедшая в 10:20, известна к закрытию часа 10:00–11:00
    g = a.groupby("hour").combined_score.agg(["sum", "count"]).reindex(idx, fill_value=0)
    s = g["sum"].rolling(24, min_periods=1).sum()
    n = g["count"].rolling(24, min_periods=1).sum()
    return pd.DataFrame({"tone": (s / n.replace(0, np.nan)), "n": n})


def day_bootstrap(x: pd.Series, y: pd.Series, n=2000, seed=0):
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    days = d.index.normalize()
    uniq = days.unique()
    rng = np.random.default_rng(seed)
    groups = {k: v for k, v in d.groupby(days)}
    vals = []
    for _ in range(n):
        pick = pd.concat([groups[k] for k in rng.choice(uniq, len(uniq))])
        vals.append(spearmanr(pick.x, pick.y).statistic)
    return spearmanr(d.x, d.y).statistic, np.percentile(vals, [2.5, 97.5]), len(d)


def main():
    art = pd.read_parquet(sys.argv[1])
    art = art[art.source_type == "news"].copy()
    art["published_at"] = pd.to_datetime(art.published_at, utc=True)
    panel = pd.read_parquet(sys.argv[2])
    panel["timestamp"] = pd.to_datetime(panel.timestamp, utc=True)
    close = panel.pivot_table(index="timestamp", columns="symbol", values="close")
    # Метка часовой свечи — её открытие; цена закрытия известна в t + 1 ч. Сдвигаем индекс к закрытию.
    close.index = close.index + pd.Timedelta(hours=1)
    idx = close.loc[START:END].index
    lines = [f"Период: {START} — {END}, новостей {len(art[(art.published_at >= START) & (art.published_at < END)]):,}", ""]

    lines += ["## 1–2. Тон новостей за 24 ч и доходность монеты", "",
              "| Новости | Против доходности | Корреляция | 95% интервал | Наблюдений |", "|---|---|---|---|---|"]
    market = sentiment_hourly(art, ["GENERAL_CRYPTO", "BTC"], idx)
    rows = {"о самой монете": [], "общий фон рынка": []}
    for coin in COINS:
        c = close[f"{coin}USDT"].reindex(idx)
        past = np.log(c / c.shift(24))
        f4, f24 = np.log(c.shift(-4) / c), np.log(c.shift(-24) / c)
        own = sentiment_hourly(art, [coin], idx)
        own_tone = own.tone.where(own.n >= 2)  # минимум две новости за сутки
        for name, tone in (("о самой монете", own_tone), ("общий фон рынка", market.tone)):
            rows[name].append(pd.DataFrame({"tone": tone, "past": past, "f4": f4, "f24": f24}))
    for name, parts in rows.items():
        d = pd.concat(parts)
        d.index = pd.DatetimeIndex(d.index)
        for col, label in (("past", "прошлые 24 ч"), ("f4", "следующие 4 ч"), ("f24", "следующие 24 ч")):
            r, ci, n = day_bootstrap(d.tone, d[col])
            lines.append(f"| {name} | {label} | {r:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {n:,} |")

    # 3. Новостной фон как 16-й голос для итоговых сигналов 15/15.
    import confluence_final as F

    T = F.trades(sys.argv[2])
    T = T[(T.time >= START) & (T.time < END)].copy()
    entry_close_time = T.time + pd.Timedelta(hours=1)
    T["market_tone"] = market.tone.reindex(entry_close_time).to_numpy()
    med = market.tone.median()
    lines += ["", "## 3. Новостной фон как 16-й голос (сигналы 15/15 в этом периоде)", "",
              f"Порог: медиана тона рынка за период ({med:+.3f}).", "",
              "| Фон | Сделок | Прибыльных | Средняя сделка | Сумма |", "|---|---|---|---|---|"]
    for label, m in (("все сигналы", T.market_tone.notna()), ("фон выше медианы", T.market_tone > med),
                     ("фон ниже медианы", T.market_tone <= med)):
        x = T[m]
        if len(x):
            lines.append(f"| {label} | {len(x)} | {(x.R > 0).mean():.0%} | {x.R.mean():+.2f}R | {x.R.sum():+.1f}R |")
    out = "\n".join(lines)
    print(out)
    Path("news_out").mkdir(exist_ok=True)
    Path("news_out/NEWS_TABLES.md").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
