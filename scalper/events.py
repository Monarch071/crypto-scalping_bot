"""Все сообщения, которые ходят внутри системы.

Архитектура событийная (event-driven), как у профессиональных систем:
маркет-дата, ответы биржи и таймеры превращаются в события и проходят
через один и тот же движок. Поэтому бэктест, paper и live используют
один и тот же код стратегии и риск-менеджмента.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderStatus(str, Enum):
    PENDING_NEW = "pending_new"
    OPEN = "open"
    PENDING_CANCEL = "pending_cancel"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"

    @property
    def is_terminal(self) -> bool:
        return self in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED)


Level = tuple[float, float]  # (price, size)


@dataclass(slots=True)
class BookEvent:
    ts: float
    bids: list[Level]
    asks: list[Level]


@dataclass(slots=True)
class TradeEvent:
    ts: float
    price: float
    size: float
    side: Side | None  # сторона агрессора, если биржа её сообщает


@dataclass(slots=True)
class OrderStatusEvent:
    ts: float
    client_id: str
    status: OrderStatus
    exchange_id: str | None = None
    reason: str = ""


@dataclass(slots=True)
class FillEvent:
    ts: float
    client_id: str
    side: Side
    price: float
    size: float
    fee: float  # в валюте котировки, положительное значение = мы заплатили
    maker: bool = True


Event = BookEvent | TradeEvent | OrderStatusEvent | FillEvent
