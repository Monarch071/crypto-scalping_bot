"""Конфигурация: TOML-файл и секреты из переменных окружения.

Ключи API и токены никогда не хранятся в коде или в конфиге из репозитория.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class ExchangeConfig:
    name: str = "bybit"
    symbol: str = "BTC/USDT:USDT"
    testnet: bool = True
    depth: int = 20
    # Используются в paper/backtest; в live берутся из описания рынка на бирже.
    tick_size: float = 0.1
    lot_size: float = 0.001
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.5


@dataclass
class StrategyConfig:
    order_size: float = 0.001
    min_half_spread_bps: float = 3.0
    vol_mult: float = 1.0
    vol_horizon_s: float = 10.0
    vol_halflife_s: float = 30.0
    inventory_skew_bps: float = 3.0
    alpha_imbalance_bps: float = 1.0
    alpha_flow_bps: float = 1.0
    imbalance_levels: int = 5
    flow_halflife_s: float = 10.0
    requote_threshold_bps: float = 1.0
    min_requote_interval_s: float = 0.25
    max_spread_bps: float = 20.0
    max_sigma_bps: float = 30.0
    warmup_samples: int = 50


@dataclass
class RiskConfig:
    max_position: float = 0.005
    max_order_notional: float = 500.0
    max_daily_loss: float = 20.0
    max_price_deviation_bps: float = 50.0
    max_orders_per_minute: int = 120
    stale_data_s: float = 5.0
    flatten_on_kill: bool = True


@dataclass
class PaperConfig:
    latency_ms: float = 80.0
    start_ts: float = 0.0


@dataclass
class Config:
    mode: str = "paper"  # paper | live
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    paper: PaperConfig = field(default_factory=PaperConfig)
    record_path: str = ""
    log_level: str = "INFO"

    @property
    def api_key(self) -> str:
        return os.environ.get("BOT_API_KEY", "")

    @property
    def api_secret(self) -> str:
        return os.environ.get("BOT_API_SECRET", "")

    @property
    def telegram_token(self) -> str:
        return os.environ.get("TELEGRAM_BOT_TOKEN", "")

    @property
    def telegram_chat_id(self) -> str:
        return os.environ.get("TELEGRAM_CHAT_ID", "")


def _fill(dc, data: dict):
    known = {f.name for f in fields(dc)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Неизвестные параметры в секции {type(dc).__name__}: {sorted(unknown)}")
    for k, v in data.items():
        setattr(dc, k, v)
    return dc


def load_config(path: str | Path | None) -> Config:
    cfg = Config()
    if path is None:
        return cfg
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    for section in ("exchange", "strategy", "risk", "paper"):
        _fill(getattr(cfg, section), raw.pop(section, {}))
    _fill(cfg, raw)
    if cfg.mode not in ("paper", "live"):
        raise ValueError("mode должен быть 'paper' или 'live'")
    return cfg
