"""Бот ротации: раз в час проверяет риск, каждые rebalance_h часов перекладывает портфель.

Бот управляет только монетами, которые купил сам (список managed в state). Чужие монеты на счёте он не
трогает, но для надёжности лучше выделить ему отдельный субаккаунт.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import RotationConfig
from .market import Account, MarketData
from .strategy import BTC, btc_uptrend, eligible, momentum_vs_btc, target_weights

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Order:
    symbol: str
    side: str
    usdt: float


def plan_orders(values: dict[str, float], equity: float, targets: dict[str, float],
                cfg: RotationConfig) -> list[Order]:
    """Какие сделки нужны, чтобы прийти к целевым долям. Сначала продажи, затем покупки.

    Мелкие изменения (меньше rebalance_band от капитала или min_trade_usdt) пропускаются, кроме полного
    выхода из монеты, которой нет в цели: её продаём целиком, чтобы не оставлять «хвостов».
    """
    invest = equity * cfg.capital_fraction
    band = max(cfg.rebalance_band * equity, cfg.min_trade_usdt)
    sells, buys = [], []
    for sym in sorted(set(values) | set(targets)):
        cur = values.get(sym, 0.0)
        tgt = targets.get(sym, 0.0) * invest
        diff = tgt - cur
        if sym not in targets and cur >= cfg.min_trade_usdt:
            sells.append(Order(sym, "sell", cur))
        elif diff <= -band:
            sells.append(Order(sym, "sell", -diff))
        elif diff >= band:
            buys.append(Order(sym, "buy", diff))
    return sells + buys


class Trader:
    def __init__(self, cfg: RotationConfig, md: MarketData, account: Account, notify=None) -> None:
        self.cfg = cfg
        self.md = md
        self.account = account
        self.notify = notify
        self.path = Path(cfg.state_file)
        self.state = self._load()
        self._now: float | None = None

    # --- состояние ---
    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {"peak_equity": 0.0, "killed": False, "kill_reason": "", "last_rebalance_slot": None,
                "managed": [], "targets": {}, "equity": 0.0, "log": []}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=1))
        tmp.replace(self.path)

    async def _say(self, text: str) -> None:
        log.info(text)
        if self.notify:
            await self.notify(text)

    def _event(self, text: str) -> None:
        when = datetime.fromtimestamp(self._now, timezone.utc) if self._now else datetime.now(timezone.utc)
        self.state["log"] = (self.state["log"] + [f"{when:%Y-%m-%d %H:%M} {text}"])[-200:]

    # --- оценка портфеля ---
    async def valuation(self) -> tuple[float, dict[str, float], dict[str, float]]:
        bal = await self.account.balances()
        managed = [s for s in self.state["managed"] if bal.get(s.split("/")[0], 0) > 0]
        prices = await self.md.prices(managed) if managed else {}
        values = {s: bal[s.split("/")[0]] * prices[s] for s in managed if s in prices}
        held = sum(values.values())
        free = bal.get(self.cfg.quote, 0.0)
        if self.cfg.max_capital_usdt > 0:
            # Бот распоряжается только своим потолком: позиции + столько USDT, сколько осталось до потолка.
            free = min(free, max(self.cfg.max_capital_usdt - held, 0.0))
        return free + held, values, prices

    # --- основной шаг (раз в час) ---
    async def step(self, now: float | None = None) -> None:
        now = now or time.time()
        self._now = now
        equity, values, _ = await self.valuation()
        self.state["equity"] = equity
        self.state["peak_equity"] = max(self.state["peak_equity"], equity)
        dd = 1 - equity / self.state["peak_equity"] if self.state["peak_equity"] > 0 else 0.0
        if not self.state["killed"] and dd >= self.cfg.max_drawdown:
            await self.kill(f"просадка {dd:.0%} от пика {self.state['peak_equity']:.2f} {self.cfg.quote}")
        if self.state["killed"]:
            self.save()
            return
        slot = int(now // 3600) // self.cfg.rebalance_h
        if self.state["last_rebalance_slot"] != slot:
            await self.rebalance(now, equity, values)
            self.state["last_rebalance_slot"] = slot
        self.save()

    async def rebalance(self, now: float, equity: float, values: dict[str, float]) -> None:
        cfg = self.cfg
        universe = await self.md.universe(cfg.quote, cfg.universe_size, set(cfg.exclude), BTC)
        close = await self.md.closes(sorted(set(universe) | set(values)), cfg.history_h, now)
        if BTC not in close or close.index[-1].timestamp() < now - 3 * 3600:
            await self._say("⚠️ Нет свежих данных по BTC, пересмотр пропущен")
            return
        scores = momentum_vs_btc(close, cfg.lookback_h).iloc[-1]
        ok = eligible(close, cfg.min_history_h).iloc[-1]
        ok = ok & ok.index.isin(universe)
        up = bool(btc_uptrend(close[BTC], cfg.trend_ema_h).iloc[-1])
        targets = target_weights(scores, ok, up, cfg.top_k)
        orders = plan_orders(values, equity, targets, cfg)
        prices = await self.md.prices(sorted({o.symbol for o in orders})) if orders else {}

        done = []
        for o in orders:
            last = prices.get(o.symbol)
            if not last:
                continue
            slip = cfg.max_slippage_bps / 1e4
            limit = last * (1 + slip) if o.side == "buy" else last * (1 - slip)
            amount = o.usdt / last
            if o.side == "buy":
                quote_free = (await self.account.balances()).get(cfg.quote, 0.0)
                if cfg.max_capital_usdt > 0:
                    held = sum(v for s, v in (await self.valuation())[1].items())
                    quote_free = min(quote_free, max(cfg.max_capital_usdt - held, 0.0))
                amount = min(amount, quote_free / (limit * (1 + cfg.fee_bps / 1e4)))
            try:
                fill = await self.account.trade(o.symbol, o.side, amount, last, limit)
            except Exception as e:  # noqa: BLE001
                await self._say(f"⚠️ Ордер {o.side} {o.symbol} не прошёл: {e}")
                continue
            if fill.amount > 0:
                self.state["trades"] = self.state.get("trades", 0) + 1
                done.append(f"{'🟢 купил' if o.side == 'buy' else '🔴 продал'} {o.symbol} "
                            f"на {fill.amount * fill.price:.2f} по {fill.price:.6g}")
                self._event(f"{o.side} {o.symbol} {fill.amount:.8g} @ {fill.price:.8g} fee {fill.fee_usdt:.4f}")

        bal = await self.account.balances()
        self.state["managed"] = sorted({s for s in set(self.state["managed"]) | set(targets)
                                        if bal.get(s.split("/")[0], 0) > 0})
        self.state["targets"] = targets
        trend = "BTC в тренде ↑" if up else "BTC не в тренде → вне рынка"
        top = ", ".join(f"{s} {scores[s]:+.3f}" for s in scores[ok].sort_values(ascending=False).index[:5])
        await self._say(f"🔄 Пересмотр: {trend}\nЦель: {', '.join(targets) or 'USDT'}\n"
                        f"Лучшие по моментуму: {top}\n" + ("\n".join(done) or "Сделок нет"))

    # --- управление ---
    async def kill(self, reason: str) -> str:
        self.state["killed"] = True
        self.state["kill_reason"] = reason
        _, values, _ = await self.valuation()
        orders = [Order(s, "sell", v) for s, v in values.items()]
        prices = await self.md.prices([o.symbol for o in orders]) if orders else {}
        for o in orders:
            last = prices[o.symbol]
            bal = await self.account.balances()
            try:
                await self.account.trade(o.symbol, "sell", bal.get(o.symbol.split("/")[0], 0.0), last,
                                         last * (1 - 3 * self.cfg.max_slippage_bps / 1e4))
            except Exception as e:  # noqa: BLE001
                await self._say(f"⚠️ Не удалось продать {o.symbol}: {e}")
        self._event(f"KILL: {reason}")
        self.save()
        await self._say(f"⛔ Бот остановлен: {reason}. Позиции проданы.")
        return "Остановлен"

    async def resume(self) -> str:
        self.state["killed"] = False
        self.state["kill_reason"] = ""
        self.state["peak_equity"] = self.state["equity"]  # отсчёт просадки заново
        self.save()
        return "Возобновлён, просадка считается от текущего капитала"

    def status(self) -> str:
        s = self.state
        dd = 1 - s["equity"] / s["peak_equity"] if s["peak_equity"] else 0.0
        state = f"ОСТАНОВЛЕН ({s['kill_reason']})" if s["killed"] else "работает"
        return (f"Режим: {self.cfg.mode}, {state}\nКапитал: {s['equity']:.2f} {self.cfg.quote} "
                f"(пик {s['peak_equity']:.2f}, просадка {dd:.1%})\n"
                f"Цель: {', '.join(s['targets']) or 'вне рынка'}\nПоследние события:\n" + "\n".join(s["log"][-5:]))
