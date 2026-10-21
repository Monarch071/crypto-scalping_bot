import asyncio

import pytest

from scalper.backtest import run_backtest
from scalper.config import Config, ExchangeConfig, RiskConfig, StrategyConfig
from scalper.engine import Engine
from scalper.events import BookEvent, FillEvent, OrderStatus, Side, TradeEvent
from scalper.gateways.paper import PaperGateway
from scalper.marketdata import synthetic
from scalper.oms import OrderManager
from scalper.orderbook import OrderBook
from scalper.portfolio import Portfolio
from scalper.risk import RiskManager
from scalper.strategy import Instrument, MarketMaker


def book(bid=100.0, ask=100.1, bs=1.0, as_=1.0, ts=0.0):
    return BookEvent(ts, [(bid, bs)], [(ask, as_)])


# --- стакан ---
def test_microprice_leans_to_heavier_side():
    ob = OrderBook()
    ob.update([(100.0, 3.0)], [(101.0, 1.0)], 0)
    assert ob.microprice == pytest.approx(100.75)  # на биде больше объёма: цена ближе к аску
    assert ob.imbalance() == pytest.approx(0.5)


# --- портфель ---
def test_portfolio_round_trip_and_flip():
    p = Portfolio()
    p.on_fill(FillEvent(0, "a", Side.BUY, 100, 1, 0.01))
    p.on_fill(FillEvent(0, "b", Side.SELL, 110, 1, 0.01))
    assert p.position == 0 and p.realized == pytest.approx(10) and p.fees == pytest.approx(0.02)
    p.on_fill(FillEvent(0, "c", Side.SELL, 100, 1, 0))
    p.on_fill(FillEvent(0, "d", Side.BUY, 90, 3, 0))   # закрыли шорт и перевернулись в лонг
    assert p.realized == pytest.approx(20)
    assert p.position == pytest.approx(2) and p.avg_price == 90
    assert p.unrealized(95) == pytest.approx(10)


# --- риск ---
def make_risk(**kw):
    r = RiskManager(RiskConfig(**kw))
    r.on_market_data(0)
    return r


def test_risk_limits():
    r = make_risk(max_position=1.0, max_order_notional=1000, max_price_deviation_bps=50)
    assert r.check_order(Side.BUY, 100, 0.5, 0.0, [], 100, 0) is None
    assert "позиция" in r.check_order(Side.BUY, 100, 0.5, 0.8, [], 100, 0)
    assert r.check_order(Side.SELL, 100, 0.5, 0.8, [], 100, 0) is None  # сокращение позиции можно
    assert "объём" in r.check_order(Side.BUY, 100, 20, 0.0, [], 100, 0)
    assert "fat finger" in r.check_order(Side.BUY, 101, 0.1, 0.0, [], 100, 0)
    assert "устаревшие" in r.check_order(Side.BUY, 100, 0.1, 0.0, [], 100, 100)


def test_risk_counts_working_orders_in_position_limit():
    r = make_risk(max_position=1.0)
    oms = OrderManager()
    oms.create(Side.BUY, 100, 0.8, 0)
    assert "позиция" in r.check_order(Side.BUY, 100, 0.5, 0.0, oms.working(), 100, 0)


def test_risk_rate_limit_and_daily_loss():
    r = make_risk(max_orders_per_minute=2, max_daily_loss=10)
    assert r.check_order(Side.BUY, 100, 0.001, 0, [], 100, 0) is None
    assert r.check_order(Side.BUY, 100, 0.001, 0, [], 100, 1) is None
    r.on_market_data(2)
    assert "лимит ордеров" in r.check_order(Side.BUY, 100, 0.001, 0, [], 100, 2)
    r.check_pnl(0, 0)
    r.check_pnl(-9, 10)
    assert not r.killed
    r.check_pnl(-10, 20)
    assert r.killed


# --- симулятор биржи ---
def paper(latency_ms=100):
    gw = PaperGateway(ExchangeConfig(maker_fee_bps=2.0), latency_ms=latency_ms)
    events = []
    gw.emit = events.append
    return gw, events


def run(coro):
    return asyncio.run(coro)


def test_paper_latency_and_fill_through_not_touch():
    gw, ev = paper()
    oms = OrderManager()
    gw.on_market_event(book(ts=0))
    o = oms.create(Side.BUY, 99.9, 1.0, 0)
    run(gw.place(o))
    gw.on_market_event(TradeEvent(0.05, 99.8, 5, Side.SELL))  # ордер ещё «летит» по сети
    assert ev == []
    gw.on_market_event(book(ts=0.11))
    assert ev[-1].status is OrderStatus.OPEN
    gw.on_market_event(TradeEvent(0.2, 99.9, 5, Side.SELL))  # касание цены не даёт исполнения
    assert not any(isinstance(e, FillEvent) for e in ev)
    gw.on_market_event(TradeEvent(0.3, 99.8, 0.4, Side.SELL))  # сделка ниже нашей цены
    fill = ev[-1]
    assert isinstance(fill, FillEvent) and fill.price == 99.9 and fill.size == 0.4
    assert fill.fee == pytest.approx(99.9 * 0.4 * 2 / 1e4)


def test_paper_post_only_reject_and_early_cancel():
    gw, ev = paper()
    oms = OrderManager()
    gw.on_market_event(book(ts=0))
    crossing = oms.create(Side.BUY, 100.1, 1, 0)
    run(gw.place(crossing))
    gw.on_market_event(book(ts=0.2))
    assert ev[-1].status is OrderStatus.REJECTED
    o = oms.create(Side.SELL, 100.5, 1, 0.2)
    run(gw.place(o))
    run(gw.cancel(o))
    gw.on_market_event(book(ts=0.5))
    assert ev[-1].status is OrderStatus.CANCELED


# --- стратегия ---
def warm_strategy(position_limit=1.0, **kw):
    cfg = StrategyConfig(warmup_samples=5, alpha_imbalance_bps=0, alpha_flow_bps=0, **kw)
    mm = MarketMaker(cfg, Instrument(0.01, 0.001), position_limit)
    ob = OrderBook()
    for i in range(10):
        ob.update([(100.0, 1)], [(100.01, 1)], float(i))
        mm.on_book(ob)
    return mm, ob


def test_inventory_skew_and_position_cap():
    mm, ob = warm_strategy(order_size=0.1)
    flat = {q.side: q for q in mm.quotes(ob, 0.0)}
    long = {q.side: q for q in mm.quotes(ob, 0.5)}
    assert flat[Side.BUY].price < ob.best_bid and flat[Side.SELL].price > ob.best_ask
    assert long[Side.SELL].price < flat[Side.SELL].price  # в лонге охотнее продаём
    full = {q.side: q for q in mm.quotes(ob, 1.0)}
    assert Side.BUY not in full and Side.SELL in full


def test_toxic_market_pulls_quotes():
    mm, ob = warm_strategy(max_spread_bps=5)
    ob.update([(100.0, 1)], [(100.2, 1)], 11.0)  # спред 20 б.п.
    assert mm.quotes(ob, 0) == []
    assert mm.state.startswith("toxic")


# --- движок целиком ---
def test_kill_switch_cancels_and_flattens():
    async def go():
        cfg = Config()
        cfg.strategy.warmup_samples = 1
        engine = Engine(cfg, PaperGateway(cfg.exchange, latency_ms=0))
        await engine.start()
        for i in range(5):
            engine.emit(BookEvent(i, [(60000.0, 1)], [(60000.1, 1)]))
            await engine.drain()
        assert engine.oms.working()
        # Цена резко уходит вниз: бид исполняется, позиция в лонге.
        engine.emit(BookEvent(6, [(59900.0, 1)], [(59900.1, 1)]))
        await engine.drain()
        assert engine.portfolio.position > 0
        await engine.kill("test")
        for i in range(7, 10):
            engine.emit(BookEvent(i, [(59900.0, 1)], [(59900.1, 1)]))
            await engine.drain()
        assert engine.portfolio.position == pytest.approx(0)
        assert not engine.oms.working()
    run(go())


def test_backtest_runs_and_respects_position_limit():
    cfg = Config()
    report = run(run_backtest(cfg, synthetic(1800, seed=3)))
    assert report.fills > 0
    assert report.max_abs_position <= cfg.risk.max_position + 1e-12
