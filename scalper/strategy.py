"""Стратегия маркет-мейкинга с управлением инвентарём.

Так скальпят крупные игроки: не угадывают направление, а постоянно стоят
лимитными ордерами с обеих сторон стакана, зарабатывают на спреде
и получают маркет-мейкерскую (низкую) комиссию.

Идея основана на модели Авелланеды–Стойкова (2008):
  * справедливая цена считается по microprice, а не по mid;
  * ширина котировки растёт вместе с волатильностью;
  * при накоплении позиции котировки смещаются так, чтобы позиция
    закрывалась (inventory skew);
  * короткосрочные сигналы (дисбаланс стакана и потока сделок) слегка
    сдвигают котировки в сторону ожидаемого движения;
  * при «токсичном» рынке (широкий спред, всплеск волатильности) котировки
    снимаются.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .config import StrategyConfig
from .events import Side, TradeEvent
from .features import EwmaVolatility, TradeFlow
from .orderbook import OrderBook


@dataclass(frozen=True, slots=True)
class Quote:
    side: Side
    price: float
    size: float


@dataclass(frozen=True, slots=True)
class Instrument:
    tick_size: float
    lot_size: float

    def round_price(self, price: float, side: Side) -> float:
        # Бид округляем вниз, аск вверх, чтобы никогда не встать агрессивнее, чем решили.
        ticks = price / self.tick_size
        ticks = math.floor(ticks + 1e-9) if side is Side.BUY else math.ceil(ticks - 1e-9)
        return round(ticks * self.tick_size, 12)

    def round_size(self, size: float) -> float:
        return round(math.floor(size / self.lot_size + 1e-9) * self.lot_size, 12)


class MarketMaker:
    def __init__(self, cfg: StrategyConfig, instrument: Instrument, max_position: float) -> None:
        self.cfg = cfg
        self.instrument = instrument
        self.max_position = max_position
        self.vol = EwmaVolatility(cfg.vol_halflife_s, cfg.warmup_samples)
        self.flow = TradeFlow(cfg.flow_halflife_s)
        self.state = "warmup"

    def on_book(self, book: OrderBook) -> None:
        self.vol.update(book.mid, book.ts)

    def on_trade(self, trade: TradeEvent) -> None:
        if trade.side is not None:
            self.flow.update(trade.size, trade.side.sign, trade.ts)

    def quotes(self, book: OrderBook, position: float) -> list[Quote]:
        cfg = self.cfg
        if not book.ready or not self.vol.ready:
            self.state = "warmup"
            return []
        sigma = self.vol.sigma_bps(cfg.vol_horizon_s)
        if book.spread_bps > cfg.max_spread_bps or sigma > cfg.max_sigma_bps:
            self.state = f"toxic (spread={book.spread_bps:.1f}bps sigma={sigma:.1f}bps)"
            return []
        self.state = "quoting"

        fair = book.microprice
        half = max(cfg.min_half_spread_bps, cfg.vol_mult * sigma)
        inv = max(-1.0, min(1.0, position / self.max_position)) if self.max_position else 0.0
        skew = -inv * cfg.inventory_skew_bps
        alpha = (cfg.alpha_imbalance_bps * book.imbalance(cfg.imbalance_levels)
                 + cfg.alpha_flow_bps * self.flow.value)
        center = fair * (1 + (skew + alpha) / 1e4)

        tick = self.instrument.tick_size
        # Post-only: не пересекаем спред, иначе станем тейкером и заплатим высокую комиссию.
        bid = min(center * (1 - half / 1e4), book.best_ask - tick)
        ask = max(center * (1 + half / 1e4), book.best_bid + tick)
        bid = self.instrument.round_price(bid, Side.BUY)
        ask = self.instrument.round_price(ask, Side.SELL)

        out: list[Quote] = []
        bid_size = self.instrument.round_size(min(cfg.order_size, self.max_position - position))
        ask_size = self.instrument.round_size(min(cfg.order_size, self.max_position + position))
        if bid_size > 0:
            out.append(Quote(Side.BUY, bid, bid_size))
        if ask_size > 0:
            out.append(Quote(Side.SELL, ask, ask_size))
        return out
