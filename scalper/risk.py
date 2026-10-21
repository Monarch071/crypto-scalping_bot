"""Риск-менеджмент: pre-trade проверки и аварийный рубильник (kill switch).

Каждый ордер проходит через check_order() до отправки на биржу. Стратегия
не может обойти эти проверки, как и в настоящих торговых системах.
"""
from __future__ import annotations

import logging
from collections import deque

from .config import RiskConfig
from .events import Side
from .oms import Order

log = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg
        self.killed = False
        self.kill_reason = ""
        self._order_times: deque[float] = deque()
        self._last_md_ts: float | None = None
        self._day: int | None = None
        self._day_start_pnl = 0.0

    # --- аварийный рубильник ---
    def kill(self, reason: str) -> None:
        if not self.killed:
            log.error("KILL SWITCH: %s", reason)
        self.killed = True
        self.kill_reason = reason

    def reset(self) -> None:
        log.warning("Kill switch сброшен вручную")
        self.killed = False
        self.kill_reason = ""

    # --- маркет-дата ---
    def on_market_data(self, ts: float) -> None:
        self._last_md_ts = ts

    def data_is_stale(self, now: float) -> bool:
        return self._last_md_ts is None or now - self._last_md_ts > self.cfg.stale_data_s

    # --- PnL ---
    def daily_pnl(self, total_pnl: float, ts: float) -> float:
        day = int(ts // 86400)
        if day != self._day:
            self._day = day
            self._day_start_pnl = total_pnl
        return total_pnl - self._day_start_pnl

    def check_pnl(self, total_pnl: float, ts: float) -> None:
        pnl = self.daily_pnl(total_pnl, ts)
        if pnl <= -self.cfg.max_daily_loss:
            self.kill(f"дневной убыток {pnl:.2f} превысил лимит {self.cfg.max_daily_loss}")

    # --- pre-trade ---
    def check_order(self, side: Side, price: float, size: float, position: float,
                    working: list[Order], mid: float, ts: float) -> str | None:
        """Возвращает причину отказа или None, если ордер можно отправлять."""
        if self.killed:
            return f"kill switch: {self.kill_reason}"
        if self.data_is_stale(ts):
            return "устаревшие рыночные данные"
        if size <= 0 or price <= 0:
            return "некорректная цена или объём"
        if price * size > self.cfg.max_order_notional:
            return f"объём ордера {price * size:.2f} > {self.cfg.max_order_notional}"
        dev = abs(price / mid - 1) * 1e4
        if dev > self.cfg.max_price_deviation_bps:
            return f"цена отклоняется от рынка на {dev:.1f} б.п. (fat finger)"
        # Худший случай: исполнятся все рабочие ордера этой стороны и новый ордер.
        same_side = sum(o.remaining for o in working if o.side is side)
        worst = position + side.sign * (same_side + size)
        if abs(worst) > self.cfg.max_position + 1e-12:
            return f"позиция может достичь {worst:.6f}, лимит {self.cfg.max_position}"
        while self._order_times and ts - self._order_times[0] > 60:
            self._order_times.popleft()
        if len(self._order_times) >= self.cfg.max_orders_per_minute:
            return "превышен лимит ордеров в минуту"
        self._order_times.append(ts)
        return None
