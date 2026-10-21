"""Настройки бота сигналов. Ключи API не нужны: бот только читает рынок и пишет в Telegram."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields


@dataclass
class SignalConfig:
    exchange: str = "binance"
    quote: str = "USDT"
    universe_size: int = 20
    exclude: list[str] = field(default_factory=list)

    # Правило входа
    min_votes: int = 15              # сколько из 15 стратегий должны быть «за»
    sides: list[str] = field(default_factory=lambda: ["long"])  # шорты в исследовании убыточны
    top_k_filter: int = 3            # только монеты из топ-N по моментуму отн. BTC (0 — выкл.)

    # Выход (research/confluence_v2.py: выход по сигналу лучше фиксированного тейка во всех проверках)
    sl_atr: float = 2.0              # стоп-лосс = 2 ATR(14, 1ч) от входа — ставится сразу
    tp_atr: float = 0.0              # тейк-профит в ATR; 0 — без тейка, выход по сигналу бота
    exit_votes: int = 10             # выходить, когда «за» стало меньше 10 из 15 (0 — выкл.)
    max_hold_h: int = 72             # в любом случае закрыть через 72 ч после входа

    # Размер позиции: сколько потерять, если сработает стоп
    deposit_usdt: float = 1000.0
    risk_pct: float = 1.0

    # Комиссии для учёта результатов сигналов (фьючерсы Binance/Bybit; ручной вход + проскальзывание)
    entry_cost_bps: float = 15.5     # 0.055% рыночный вход + 0.1% на ручную задержку
    tp_cost_bps: float = 2.0         # тейк лимитным ордером 0.02%
    sl_cost_bps: float = 10.5        # стоп рыночный 0.055% + 0.05% проскальзывания
    time_exit_cost_bps: float = 15.5  # рыночный выход 0.055% + 0.1% на ручную задержку
    signal_exit_cost_bps: float = 15.5

    check_minutes: int = 5           # как часто проверять открытые сигналы по текущей цене
    history_h: int = 1000
    btc_history_h: int = 3000        # для EMA BTC за 30 дней нужна длинная история
    state_file: str = "data/signals_state.json"
    log_level: str = "INFO"


def load_signal_config(path: str | None) -> SignalConfig:
    cfg = SignalConfig()
    if path is None:
        return cfg
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    unknown = set(raw) - {f.name for f in fields(cfg)}
    if unknown:
        raise ValueError(f"Неизвестные параметры: {sorted(unknown)}")
    for k, v in raw.items():
        setattr(cfg, k, v)
    if not 1 <= cfg.min_votes <= 15:
        raise ValueError("min_votes должен быть от 1 до 15")
    if set(cfg.sides) - {"long", "short"}:
        raise ValueError("sides: только 'long' и/или 'short'")
    return cfg
