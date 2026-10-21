"""Order Management System: состояние каждого ордера на протяжении его жизни."""
from __future__ import annotations

import itertools
import logging
import time
from dataclasses import dataclass

from .events import FillEvent, OrderStatus, OrderStatusEvent, Side

log = logging.getLogger(__name__)


@dataclass
class Order:
    client_id: str
    side: Side
    price: float
    size: float
    created_ts: float
    status: OrderStatus = OrderStatus.PENDING_NEW
    filled: float = 0.0
    exchange_id: str | None = None
    reduce_only: bool = False
    market: bool = False

    @property
    def remaining(self) -> float:
        return max(self.size - self.filled, 0.0)

    @property
    def is_working(self) -> bool:
        return not self.status.is_terminal


class OrderManager:
    def __init__(self, prefix: str = "sc") -> None:
        # Уникальный префикс на запуск, чтобы id не совпали с ордерами прошлых сессий.
        self._prefix = f"{prefix}{int(time.time()) % 100000}"
        self._seq = itertools.count(1)
        self.orders: dict[str, Order] = {}   # все ордера, включая недавно завершённые
        self._active: dict[str, Order] = {}  # только рабочие: по ним идёт каждый тик

    def create(self, side: Side, price: float, size: float, ts: float,
               reduce_only: bool = False, market: bool = False) -> Order:
        cid = f"{self._prefix}x{next(self._seq)}"
        order = Order(cid, side, price, size, ts, reduce_only=reduce_only, market=market)
        self.orders[cid] = order
        self._active[cid] = order
        return order

    def working(self, side: Side | None = None) -> list[Order]:
        return [o for o in self._active.values() if side is None or o.side is side]

    def _settle(self, order: Order) -> None:
        if order.status.is_terminal:
            self._active.pop(order.client_id, None)

    def on_status(self, ev: OrderStatusEvent) -> Order | None:
        order = self.orders.get(ev.client_id)
        if order is None:
            log.warning("Статус по неизвестному ордеру %s", ev.client_id)
            return None
        if ev.exchange_id:
            order.exchange_id = ev.exchange_id
        if order.status.is_terminal:
            return order
        # Пока мы ждём отмену, подтверждение «open» статус не меняет.
        if not (ev.status is OrderStatus.OPEN and order.status is OrderStatus.PENDING_CANCEL):
            order.status = ev.status
        if ev.status is OrderStatus.REJECTED:
            log.info("Ордер %s отклонён: %s", order.client_id, ev.reason)
        self._settle(order)
        return order

    def on_fill(self, ev: FillEvent) -> Order | None:
        order = self.orders.get(ev.client_id)
        if order is None:
            log.warning("Исполнение по неизвестному ордеру %s", ev.client_id)
            return None
        order.filled += ev.size
        if order.remaining <= order.size * 1e-9:
            order.status = OrderStatus.FILLED
        self._settle(order)
        return order

    def prune(self, keep: int = 1000) -> None:
        """Удаляет старые завершённые ордера, чтобы память не росла в долгой работе."""
        excess = len(self.orders) - len(self._active) - keep
        if excess <= keep // 10:  # чистим пачками, а не на каждом тике
            return
        for cid in [c for c in self.orders if c not in self._active][:excess]:
            del self.orders[cid]
