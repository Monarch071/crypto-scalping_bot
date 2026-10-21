import asyncio

import numpy as np
import pandas as pd
import pytest

from rotator.config import RotationConfig
from rotator.market import PaperAccount
from rotator.strategy import BTC, btc_uptrend, momentum_vs_btc, target_weights
from rotator.trader import Order, Trader, plan_orders

SYMS = [BTC, "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT", "BNB/USDT"]


def panel(hours=2500, seed=0, btc_drift=2e-4, winner="SOL/USDT"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=hours, freq="h", tz="UTC")
    btc = 50000 * np.exp(np.cumsum(rng.normal(btc_drift, 0.004, hours)))
    data = {BTC: btc}
    for s in SYMS[1:]:
        drift = 6e-4 if s == winner else 0.0
        data[s] = 100 * np.exp(np.cumsum(rng.normal(drift, 0.006, hours)) + np.log(btc / btc[0]))
    return pd.DataFrame(data, index=idx)


# --- стратегия ---
def test_momentum_picks_outperformer_and_trend_filter():
    close = panel()
    scores = momentum_vs_btc(close, 168).iloc[-1]
    ok = pd.Series(True, index=close.columns)
    w = target_weights(scores, ok, True, 2)
    assert "SOL/USDT" in w and sum(w.values()) == pytest.approx(1.0)
    assert target_weights(scores, ok, False, 2) == {}  # BTC не в тренде — вне рынка
    assert target_weights(scores, ok, True, 4) == {}   # монет меньше 2·k — не торгуем


def test_residual_momentum_of_btc_is_zero():
    close = panel()
    assert momentum_vs_btc(close, 168)[BTC].dropna().abs().max() == pytest.approx(0, abs=1e-9)


def test_btc_uptrend():
    up = btc_uptrend(pd.Series(np.linspace(1, 2, 1000)), 100)
    assert bool(up.iloc[-1]) and not bool(btc_uptrend(pd.Series(np.linspace(2, 1, 1000)), 100).iloc[-1])


# --- план ордеров ---
def test_plan_orders_sells_first_and_skips_small_changes():
    cfg = RotationConfig(capital_fraction=1.0, rebalance_band=0.05, min_trade_usdt=10)
    values = {"ETH/USDT": 500.0, "SOL/USDT": 480.0, "XRP/USDT": 5.0}
    orders = plan_orders(values, 1000.0, {"SOL/USDT": 0.5, "ADA/USDT": 0.5}, cfg)
    assert orders == [Order("ETH/USDT", "sell", 500.0), Order("ADA/USDT", "buy", 500.0)]
    # SOL: 480 → 500 — изменение меньше полосы, не трогаем; XRP 5 USDT — меньше минимума.


def test_plan_orders_full_exit_when_out_of_market():
    cfg = RotationConfig()
    assert plan_orders({"ETH/USDT": 300.0}, 1000.0, {}, cfg) == [Order("ETH/USDT", "sell", 300.0)]


# --- бумажный счёт ---
def test_paper_account_fees_and_limits():
    acc = PaperAccount({"USDT": 1000.0}, fee_bps=10, slippage_bps=5)
    fill = asyncio.run(acc.trade("ETH/USDT", "buy", 1.0, last=500.0, limit=510.0))
    assert fill.price == pytest.approx(500.25) and fill.amount == pytest.approx(1.0)
    assert acc.bal["USDT"] == pytest.approx(1000 - 500.25 - 0.50025)
    assert asyncio.run(acc.trade("ETH/USDT", "buy", 1.0, last=500.0, limit=499.0)).amount == 0  # лимит
    fill = asyncio.run(acc.trade("ETH/USDT", "sell", 5.0, last=600.0, limit=590.0))
    assert fill.amount == pytest.approx(1.0) and acc.bal.get("ETH", 0) == pytest.approx(0)


# --- бот целиком на подставных данных ---
class FakeMarket:
    def __init__(self, close):
        self.close = close
        self.t = close.index[-1]

    async def universe(self, quote, n, exclude, btc):
        return list(self.close.columns)

    async def closes(self, symbols, hours, now=None):
        return self.close.loc[:self.t, [s for s in symbols if s in self.close]]

    async def prices(self, symbols):
        return {s: float(self.close.loc[self.t, s]) for s in symbols}


def make_trader(tmp_path, close, **kw):
    cfg = RotationConfig(state_file=str(tmp_path / "state.json"), top_k=2, **kw)
    md = FakeMarket(close)
    t = Trader(cfg, md, None)
    t.account = PaperAccount(t.state.setdefault("paper_balances", {"USDT": 1000.0}), cfg.fee_bps)
    return t, md


def test_trader_rebalances_once_per_slot_and_buys_leaders(tmp_path):
    close = panel()
    trader, md = make_trader(tmp_path, close)
    now = md.t.timestamp() + 3600
    asyncio.run(trader.step(now))
    assert "SOL/USDT" in trader.state["targets"]
    bal = trader.account.bal
    assert bal["SOL"] > 0 and bal["USDT"] < 100
    before = dict(bal)
    asyncio.run(trader.step(now + 60))  # тот же 8-часовой слот — повторного пересмотра нет
    assert trader.account.bal == before
    # Состояние переживает перезапуск.
    t2, _ = make_trader(tmp_path, close)
    assert t2.state["managed"] == trader.state["managed"]


def test_trader_goes_flat_when_btc_trend_breaks(tmp_path):
    close = panel()
    trader, md = make_trader(tmp_path, close, max_drawdown=0.99)  # проверяем фильтр тренда, а не стоп
    asyncio.run(trader.step(md.t.timestamp() + 3600))
    assert trader.state["managed"]
    crash = panel(btc_drift=-6e-4, seed=1)  # BTC в нисходящем тренде
    md.close = crash.set_axis(close.index)
    trader.state["last_rebalance_slot"] = None  # следующий слот пересмотра с теми же свежими свечами
    asyncio.run(trader.step(md.t.timestamp() + 3600))
    assert trader.state["targets"] == {}
    assert all(trader.account.bal.get(s.split("/")[0], 0) < 1e-9 for s in SYMS)


def test_kill_switch_on_drawdown_sells_everything(tmp_path):
    close = panel()
    trader, md = make_trader(tmp_path, close, max_drawdown=0.2)
    asyncio.run(trader.step(md.t.timestamp() + 3600))
    md.close = close * 0.5  # всё упало вдвое
    md.close[BTC] = close[BTC]
    asyncio.run(trader.step(md.t.timestamp() + 7200))
    assert trader.state["killed"]
    assert all(trader.account.bal.get(s.split("/")[0], 0) < 1e-9 for s in SYMS[1:])
    assert asyncio.run(trader.resume()).startswith("Возобновлён") and not trader.state["killed"]


def test_stale_data_skips_rebalance(tmp_path):
    close = panel()
    trader, md = make_trader(tmp_path, close)
    asyncio.run(trader.step(md.t.timestamp() + 3600 * 9))  # последней свече 9 часов
    assert trader.state["targets"] == {} and trader.account.bal == {"USDT": 1000.0}
