"""Локальная копия стакана и микроструктурные метрики."""
from __future__ import annotations

from .events import Level


class OrderBook:
    def __init__(self) -> None:
        self.bids: list[Level] = []
        self.asks: list[Level] = []
        self.ts: float = 0.0

    def update(self, bids: list[Level], asks: list[Level], ts: float) -> None:
        self.bids = bids
        self.asks = asks
        self.ts = ts

    @property
    def ready(self) -> bool:
        return bool(self.bids) and bool(self.asks) and self.bids[0][0] < self.asks[0][0]

    @property
    def best_bid(self) -> float:
        return self.bids[0][0]

    @property
    def best_ask(self) -> float:
        return self.asks[0][0]

    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread_bps(self) -> float:
        return (self.best_ask - self.best_bid) / self.mid * 1e4

    @property
    def microprice(self) -> float:
        """Средняя цена, взвешенная объёмами лучших уровней.

        Если на биде объём больше, чем на аске, справедливая цена ближе к аску:
        давление покупателей сильнее. Это стандартная оценка «справедливой цены»
        у маркет-мейкеров, она точнее простого mid.
        """
        (bp, bs), (ap, as_) = self.bids[0], self.asks[0]
        total = bs + as_
        if total <= 0:
            return self.mid
        return (bp * as_ + ap * bs) / total

    def imbalance(self, levels: int = 5) -> float:
        """Дисбаланс объёмов в стакане, от -1 (давят продавцы) до +1 (давят покупатели)."""
        b = sum(s for _, s in self.bids[:levels])
        a = sum(s for _, s in self.asks[:levels])
        total = a + b
        return 0.0 if total <= 0 else (b - a) / total
