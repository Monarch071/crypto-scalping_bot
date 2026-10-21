"""Скачивает минутные свечи с data.binance.vision в формат исследовательских скриптов.

Запускать на своём компьютере (из облачной песочницы Binance недоступен):

  python research/download_binance.py SOLUSDT --start 2022-01 --end 2026-09 --market futures
  python research/ml_study.py data/binance_SOLUSDT_futures_1m.csv --configs 15:1,15:4

Много монет сразу (часовые свечи) для бота, выбирающего монеты из рынка:

  python research/download_binance.py BTCUSDT ETHUSDT SOLUSDT ... --interval 1h \
      --start 2020-01 --end 2026-09 --panel data/panel_1h.parquet
  python research/xs_study.py data/panel_1h.parquet

Включайте и монеты, которые уже убрали с биржи: без них результат завышен (ошибка выжившего).

Результат: CSV на монету с колонками timestamp (секунды, UTC, открытие свечи), open, high, low, close,
volume; с --panel — ещё и общий parquet с колонкой symbol.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

BASE = {
    "futures": "https://data.binance.vision/data/futures/um/monthly/klines/{s}/{i}/{s}-{i}-{m}.zip",
    "spot": "https://data.binance.vision/data/spot/monthly/klines/{s}/{i}/{s}-{i}-{m}.zip",
}


def months(start: str, end: str):
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    while (y, m) <= (ey, em):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m > 12:
            y, m = y + 1, 1


def rows_from_zip(data: bytes):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        with z.open(z.namelist()[0]) as f:
            for row in csv.reader(io.TextIOWrapper(f)):
                if not row or not row[0].isdigit():
                    continue  # строка заголовка в новых файлах
                ts = int(row[0])
                if ts > 10**14:  # с 2025 года спот в микросекундах
                    ts //= 1000
                yield [ts // 1000, row[1], row[2], row[3], row[4], row[5]]


def download(symbol, interval, market, start, end, out_dir) -> Path:
    out = Path(out_dir) / f"binance_{symbol}_{market}_{interval}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for m in months(start, end):
            url = BASE[market].format(s=symbol, i=interval, m=m)
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    data = r.read()
            except urllib.error.HTTPError as e:
                print(f"{symbol} {m}: нет файла ({e.code}), пропускаю", file=sys.stderr)
                continue
            for row in rows_from_zip(data):
                w.writerow(row)
                n += 1
            print(f"{symbol} {m}: ok", file=sys.stderr)
    print(f"Готово: {out} ({n:,} свечей)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="+")
    ap.add_argument("--interval", default="1m", help="1m, 5m, 15m, 1h, 4h, 1d")
    ap.add_argument("--start", default="2022-01")
    ap.add_argument("--end", required=True, help="последний полный месяц, например 2026-09")
    ap.add_argument("--market", choices=list(BASE), default="futures")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--panel", help="собрать все монеты в один parquet (нужны pandas и pyarrow)")
    args = ap.parse_args()
    files = {s: download(s, args.interval, args.market, args.start, args.end, args.out_dir) for s in args.symbols}
    if args.panel:
        import pandas as pd

        frames = []
        for sym, path in files.items():
            df = pd.read_csv(path)
            if df.empty:
                continue
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
            df["symbol"] = sym
            frames.append(df)
        pd.concat(frames).to_parquet(args.panel, index=False)
        print(f"Панель: {args.panel}")


if __name__ == "__main__":
    main()
