"""Позиция и PnL."""
from __future__ import annotations

from .events import FillEvent


class Portfolio:
    def __init__(self) -> None:
        self.position = 0.0
        self.avg_price = 0.0
        self.realized = 0.0
        self.fees = 0.0
        self.volume = 0.0
        self.fills = 0

    def on_fill(self, ev: FillEvent) -> None:
        qty = ev.side.sign * ev.size
        self.fees += ev.fee
        self.volume += ev.size * ev.price
        self.fills += 1
        pos = self.position
        if pos == 0 or (pos > 0) == (qty > 0):
            # Наращиваем позицию: пересчитываем среднюю цену.
            new = pos + qty
            self.avg_price = (self.avg_price * abs(pos) + ev.price * abs(qty)) / abs(new)
            self.position = new
            return
        # Сокращаем или переворачиваем позицию.
        closed = min(abs(qty), abs(pos))
        self.realized += closed * (ev.price - self.avg_price) * (1 if pos > 0 else -1)
        new = pos + qty
        if abs(new) < 1e-12:
            self.position, self.avg_price = 0.0, 0.0
        elif (new > 0) != (pos > 0):
            self.position, self.avg_price = new, ev.price
        else:
            self.position = new

    def unrealized(self, mark: float) -> float:
        return self.position * (mark - self.avg_price) if self.position else 0.0

    def total_pnl(self, mark: float) -> float:
        """Чистый PnL после комиссий, в валюте котировки."""
        return self.realized + self.unrealized(mark) - self.fees
