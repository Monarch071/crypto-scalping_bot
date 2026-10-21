"""Общий интерфейс шлюза к бирже. Движок не знает, live это, paper или бэктест."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from ..events import Event
from ..oms import Order
from ..strategy import Instrument


class Gateway(ABC):
    def __init__(self) -> None:
        self.emit: Callable[[Event], None] = lambda ev: None

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    @abstractmethod
    def instrument(self) -> Instrument: ...

    @abstractmethod
    async def place(self, order: Order) -> None:
        """Отправить ордер. Результат приходит позже событиями OrderStatusEvent/FillEvent."""

    @abstractmethod
    async def cancel(self, order: Order) -> None: ...

    @abstractmethod
    async def cancel_all(self) -> None: ...

    def on_market_event(self, ev: Event) -> None:
        """Нужен только симулятору: даёт ему увидеть рынок и исполнить ордера."""
