"""Скачивает минутные свечи с data.binance.vision в формат исследовательских скриптов.

Запускать на своём компьютере (из облачной песочницы Binance недоступен):

  python research/download_binance.py SOLUSDT --start 2022-01 --end 2026-09 --market futures
  python research/ml_study.py data/binance_SOLUSDT_futures_1m.csv --configs 15:1,15:4
  python research/ml_signal.py data/binance_SOLUSDT_futures_1m.csv --configs 15:1,15:4

Результат: CSV с колонками timestamp (секунды, UTC, открытие минуты), open, high, low, close, volume.
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
    "futures": "https://data.binance.vision/data/futures/um/monthly/klines/{s}/1m/{s}-1m-{m}.zip",
    "spot": "https://data.binance.vision/data/spot/monthly/klines/{s}/1m/{s}-1m-{m}.zip",
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol")
    ap.add_argument("--start", default="2022-01")
    ap.add_argument("--end", required=True, help="последний полный месяц, например 2026-09")
    ap.add_argument("--market", choices=list(BASE), default="futures")
    ap.add_argument("--out-dir", default="data")
    args = ap.parse_args()
    out = Path(args.out_dir) / f"binance_{args.symbol}_{args.market}_1m.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for m in months(args.start, args.end):
            url = BASE[args.market].format(s=args.symbol, m=m)
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    data = r.read()
            except urllib.error.HTTPError as e:
                print(f"{m}: нет файла ({e.code}), пропускаю", file=sys.stderr)
                continue
            for row in rows_from_zip(data):
                w.writerow(row)
                n += 1
            print(f"{m}: ok", file=sys.stderr)
    print(f"Готово: {out} ({n:,} минут)")


if __name__ == "__main__":
    main()
