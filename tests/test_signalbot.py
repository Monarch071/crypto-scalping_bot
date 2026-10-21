import asyncio

import numpy as np
import pandas as pd
import pytest

from signalbot.bot import SignalBot
from signalbot.config import SignalConfig
from signalbot.logic import Signal, check_exit, close_signal, find_signal

BTC = "BTC/USDT"


def candles(close: np.ndarray, start="2026-01-01", vol=None) -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(close), freq="h", tz="UTC")
    c = pd.Series(close, index=idx)
    o = c.shift(1).fillna(c.iloc[0])
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.002, "low": np.minimum(o, c) * 0.998,
                         "close": c, "volume": vol if vol is not None else np.full(len(c), 100.0)}, index=idx)


def trend_then_breakout(n=1500, seed=0):
    """Долгий подъём рынка, затем короткая пауза в монете и резкий рост с объёмом — голоса сходятся."""
    rng = np.random.default_rng(seed)
    btc = 30000 * np.exp(np.cumsum(rng.normal(3e-4, 0.002, n)))
    coin = 100 * np.exp(np.cumsum(np.r_[rng.normal(4e-4, 0.003, n - 60), rng.normal(-2e-3, 0.002, 40),
                                       rng.normal(4e-3, 0.002, 20)]))
    vol = np.r_[np.full(n - 30, 100.0), np.full(30, 400.0)]
    return candles(btc), candles(coin, vol=vol)


def cfg(tmp_path=None, **kw):
    base = dict(min_votes=14, top_k_filter=0, tp_atr=4.0, exit_votes=0, max_hold_h=24, sl_pct=0.0)
    base.update(kw)
    if tmp_path is not None:
        base["state_file"] = str(tmp_path / "s.json")
    return SignalConfig(**base)


def first_signal(btc, coin, c):
    for i in range(400, len(coin)):
        s = find_signal("X/USDT", coin.iloc[:i], btc.close.iloc[:i], c)
        if s:
            return i, s
    return None, None


def test_signal_levels_size_and_single_trigger():
    btc, coin = trend_then_breakout()
    c = cfg()
    i, s = first_signal(btc, coin, c)
    assert s is not None and s.side == "long" and len(s.agree) >= 14
    assert s.sl < s.entry < s.tp
    assert (s.tp - s.entry) == pytest.approx(2 * (s.entry - s.sl))  # тейк 4 ATR, стоп 2 ATR
    assert s.size * (s.entry - s.sl) == pytest.approx(10.0)          # риск 1% от 1000 USDT
    # Следующий час: голосов по-прежнему много, но это уже не новый сигнал.
    assert find_signal("X/USDT", coin.iloc[:i + 1], btc.close.iloc[:i + 1], c) is None


def test_no_shorts_by_default_and_no_signal_in_flat_market():
    rng = np.random.default_rng(3)
    flat = candles(100 * np.exp(np.cumsum(rng.normal(0, 0.002, 1500))))
    btc = candles(30000 * np.exp(np.cumsum(rng.normal(0, 0.002, 1500))))
    c = cfg(min_votes=15)
    assert all(find_signal("X/USDT", flat.iloc[:i], btc.close.iloc[:i], c) is None for i in range(1300, 1500, 7))


def sig(**kw):
    base = dict(id="x", symbol="X/USDT", side="long", time="2026-01-01T00:00:00+00:00", entry=100.0, sl=98.0,
                tp=104.0, expires="2026-01-02T00:00:00+00:00", size=5.0, risk_usdt=10.0, agree=[])
    base.update(kw)
    return Signal(**base)


def test_check_exit_rules():
    t = pd.Timestamp("2026-01-01T05:00", tz="UTC")
    assert check_exit(sig(), 105, 97, 100, t) == ("sl", 98.0)   # в одном баре оба — считаем стоп
    assert check_exit(sig(), 104.5, 99, 104, t) == ("tp", 104.0)
    assert check_exit(sig(), 101, 99, 100, t) is None
    assert check_exit(sig(), 101, 99, 100.5, pd.Timestamp("2026-01-02T01:00", tz="UTC")) == ("time", 100.5)
    assert check_exit(sig(tp=None), 150, 99, 140, t) is None  # без тейка рост не закрывает сделку
    short = sig(side="short", sl=102.0, tp=96.0)
    assert check_exit(short, 101, 95.5, 96, t) == ("tp", 96.0)


def test_result_in_r_includes_costs():
    c = cfg()
    s = close_signal(sig(), "tp", 104.0, pd.Timestamp("2026-01-01T06:00", tz="UTC"), c)
    assert s.net_bps == pytest.approx(400 - 15.5 - 2.0)
    assert s.r == pytest.approx((400 - 17.5) / 200)
    s = close_signal(sig(), "sl", 98.0, pd.Timestamp("2026-01-01T06:00", tz="UTC"), c)
    assert s.r == pytest.approx((-200 - 26.0) / 200)
    s = close_signal(sig(), "signal", 103.0, pd.Timestamp("2026-01-01T06:00", tz="UTC"), c)
    assert s.r == pytest.approx((300 - 31.0) / 200)


class FakeMarket:
    def __init__(self, frames):
        self.frames = frames
        self.t = None

    async def universe(self, quote, n, exclude, btc):
        return list(self.frames)

    async def candles(self, symbols, hours, now=None):
        return {s: self.frames[s].loc[:self.t].iloc[-hours:] for s in symbols if s in self.frames}

    async def prices(self, symbols):
        return {s: float(self.frames[s].close.loc[:self.t].iloc[-1]) for s in symbols}


def test_bot_end_to_end_signal_then_exit_and_stats(tmp_path):
    btc, coin = trend_then_breakout()
    c = cfg(tmp_path)
    i, expected = first_signal(btc, coin, c)
    # После сигнала цена уходит вверх до тейк-профита.
    after = coin.iloc[:i].copy()
    up = candles(np.r_[after.close.to_numpy(), after.close.iloc[-1] * np.exp(np.linspace(0.002, 0.2, 30))],
                 vol=np.r_[after.volume.to_numpy(), np.full(30, 400.0)])
    btc2 = candles(np.r_[btc.close.iloc[:i].to_numpy(), btc.close.iloc[i - 1] * np.exp(np.linspace(0, 0.01, 30))])
    md = FakeMarket({BTC: btc2, "X/USDT": up})
    msgs = []

    async def notify(text):
        msgs.append(text)

    bot = SignalBot(c, md, notify)
    md.t = up.index[i - 1]
    asyncio.run(bot.step(md.t.timestamp() + 3600 + 30))
    assert len(bot.open) == 1 and bot.open[0].entry == pytest.approx(expected.entry)
    assert "ЛОНГ" in msgs[-1] and "Стоп-лосс" in msgs[-1]
    md.t = up.index[i + 25]
    asyncio.run(bot.step(md.t.timestamp() + 3600 + 30))
    assert not bot.open and bot.closed[0].status == "tp"
    assert "Тейк-профит" in msgs[-1]
    assert "Закрыто сигналов: 1" in bot.stats()
    # Журнал переживает перезапуск.
    assert len(SignalBot(c, None).closed) == 1


def test_default_rules_no_take_profit_and_exit_when_strategies_disagree(tmp_path):
    """Правила по умолчанию: без тейка, выход по сообщению бота, когда «за» станет меньше 10 из 15."""
    btc, coin = trend_then_breakout()
    c = SignalConfig(state_file=str(tmp_path / "s.json"), min_votes=14, top_k_filter=0)
    assert c.tp_atr == 0 and c.exit_votes == 10 and c.max_hold_h == 72 and c.sl_pct == 5.0
    i, expected = first_signal(btc, coin, c)
    assert expected.tp is None
    assert expected.sl == pytest.approx(expected.entry * 0.95)                  # стоп 5% ниже входа
    assert expected.size * expected.entry == pytest.approx(10.0 / 0.05)         # риск $10 → покупка на $200
    # После сигнала монета резко разворачивается вниз (но не до стопа) — согласие стратегий пропадает.
    last = coin.close.iloc[i - 1]
    path = last * np.exp(np.r_[np.linspace(0.001, 0.004, 3), np.linspace(0.003, -0.012, 12)])
    down = candles(np.r_[coin.close.iloc[:i].to_numpy(), path], vol=np.r_[coin.volume.iloc[:i].to_numpy(),
                                                                          np.full(15, 100.0)])
    btc2 = candles(np.r_[btc.close.iloc[:i].to_numpy(), btc.close.iloc[i - 1] * np.ones(15)])
    md = FakeMarket({BTC: btc2, "X/USDT": down})
    msgs = []

    async def notify(text):
        msgs.append(text)

    bot = SignalBot(c, md, notify)
    md.t = down.index[i - 1]
    asyncio.run(bot.step(md.t.timestamp() + 3600 + 30))
    assert len(bot.open) == 1 and "не ставить" in msgs[-1]
    for k in range(i, i + 15):
        md.t = down.index[k]
        asyncio.run(bot.step(md.t.timestamp() + 3600 + 30))
        if not bot.open:
            break
    assert bot.closed and bot.closed[0].status == "signal"
    assert "больше не согласны" in msgs[-1]
