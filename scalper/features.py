"""Потоковые (онлайн) признаки: O(1) по памяти и CPU на каждое событие.

Здесь нет pandas-окон, только экспоненциальные средние по времени. Так считают
в продакшн-системах: работает одинаково быстро на любом железе и не зависит
от частоты событий.
"""
from __future__ import annotations

import math


def _alpha(dt: float, halflife_s: float) -> float:
    return 1.0 - math.exp(-max(dt, 0.0) * math.log(2) / halflife_s)


class EwmaVolatility:
    """Оценка волатильности цены в секунду, взвешенная по времени."""

    def __init__(self, halflife_s: float = 30.0, min_samples: int = 50) -> None:
        self.halflife_s = halflife_s
        self.min_samples = min_samples
        self._last_price: float | None = None
        self._last_ts: float | None = None
        self._var_per_s = 0.0
        self.samples = 0

    def update(self, price: float, ts: float) -> None:
        if self._last_price is None or self._last_ts is None:
            self._last_price, self._last_ts = price, ts
            return
        dt = ts - self._last_ts
        if dt <= 0:
            return
        r = math.log(price / self._last_price)
        a = _alpha(dt, self.halflife_s)
        self._var_per_s = (1 - a) * self._var_per_s + a * (r * r / dt)
        self._last_price, self._last_ts = price, ts
        self.samples += 1

    @property
    def ready(self) -> bool:
        return self.samples >= self.min_samples

    def sigma_bps(self, horizon_s: float) -> float:
        """Ожидаемое стандартное отклонение цены за horizon_s секунд, в б.п."""
        return math.sqrt(self._var_per_s * horizon_s) * 1e4


class TradeFlow:
    """Дисбаланс агрессивных сделок от -1 до +1: кто сейчас «бьёт по рынку»."""

    def __init__(self, halflife_s: float = 10.0) -> None:
        self.halflife_s = halflife_s
        self._signed = 0.0
        self._total = 0.0
        self._last_ts: float | None = None

    def update(self, size: float, sign: int, ts: float) -> None:
        if self._last_ts is not None:
            decay = 1 - _alpha(ts - self._last_ts, self.halflife_s)
            self._signed *= decay
            self._total *= decay
        self._last_ts = ts
        self._signed += sign * size
        self._total += size

    @property
    def value(self) -> float:
        return 0.0 if self._total <= 0 else self._signed / self._total
