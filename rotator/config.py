"""Настройки бота ротации. Ключи только из переменных окружения: BOT_API_KEY, BOT_API_SECRET,
TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields


@dataclass
class RotationConfig:
    mode: str = "paper"                  # paper | live
    exchange: str = "binance"
    testnet: bool = False                # live на тестовой бирже: настоящий API, ненастоящие деньги
    quote: str = "USDT"
    universe_size: int = 20              # топ-N спотовых пар по объёму за 24 ч (+ BTC всегда)
    exclude: list[str] = field(default_factory=list)  # например ["DOGE/USDT"]

    # Стратегия (значения проверены в research/xs_study.py)
    lookback_h: int = 168                # окно моментума
    rebalance_h: int = 8                 # пересмотр портфеля каждые N часов (по UTC: 0, 8, 16)
    top_k: int = 3                       # сколько монет держать
    trend_ema_h: int = 720               # тренд BTC: цена выше EMA за 30 дней
    min_history_h: int = 1440            # монета без 60 дней истории не участвует
    history_h: int = 2000                # сколько часовых свечей загружать

    # Исполнение
    capital_fraction: float = 0.95       # доля капитала в позициях, остальное — запас на комиссии
    rebalance_band: float = 0.05         # не трогать позицию, если доля изменилась меньше чем на 5% капитала
    min_trade_usdt: float = 10.0
    max_slippage_bps: float = 30.0       # лимитный IOC-ордер не дальше 0.3% от цены
    fee_bps: float = 10.0                # спот Binance и Bybit: 0.1% (для paper и отчётов)

    # Риск
    max_capital_usdt: float = 0.0        # потолок капитала бота (0 — без потолка); остальное на счёте не трогает
    # Обычные просадки стратегии доходят почти до 50% (прогон 2022–2026), поэтому стоп стоит за этой
    # границей и ловит поломку или смену рынка. Сколько не жалко потерять — задаёт max_capital_usdt.
    max_drawdown: float = 0.55           # от пика капитала → продать всё и остановиться
    paper_start_usdt: float = 1000.0
    state_file: str = "data/rotator_state.json"
    log_level: str = "INFO"

    @property
    def api_key(self) -> str:
        return os.environ.get("BOT_API_KEY", "")

    @property
    def api_secret(self) -> str:
        return os.environ.get("BOT_API_SECRET", "")


def load_rotation_config(path: str | None) -> RotationConfig:
    cfg = RotationConfig()
    if path is None:
        return cfg
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    known = {f.name for f in fields(cfg)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Неизвестные параметры: {sorted(unknown)}")
    for k, v in raw.items():
        setattr(cfg, k, v)
    if cfg.mode not in ("paper", "live"):
        raise ValueError("mode должен быть 'paper' или 'live'")
    if cfg.rebalance_h not in (1, 2, 3, 4, 6, 8, 12, 24):
        raise ValueError("rebalance_h должен делить 24")
    return cfg
