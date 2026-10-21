"""Бот сигналов: раз в час ищет входы, каждые несколько минут проверяет открытые сигналы.

Бот не торгует. Он присылает в Telegram монету, вход, стоп-лосс, тейк-профит и размер позиции, а потом
сообщает, когда сработал выход. Все сигналы записываются в журнал, и бот ведёт честную статистику
«как если бы вы входили по каждому» (с комиссиями и проскальзыванием ручного входа).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

from rotator.strategy import BTC, eligible, momentum_vs_btc

from .config import SignalConfig
from .logic import (Signal, agree_count, check_exit, close_signal, find_signal, format_signal, from_dict, stats,
                    to_dict)

log = logging.getLogger(__name__)


class SignalBot:
    def __init__(self, cfg: SignalConfig, md, notify=None) -> None:
        self.cfg = cfg
        self.md = md
        self.notify = notify
        self.path = Path(cfg.state_file)
        raw = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.open: list[Signal] = [from_dict(d) for d in raw.get("open", [])]
        self.closed: list[Signal] = [from_dict(d) for d in raw.get("closed", [])]
        self.last_scan_hour: int | None = raw.get("last_scan_hour")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"open": [to_dict(s) for s in self.open],
                                   "closed": [to_dict(s) for s in self.closed][-5000:],
                                   "last_scan_hour": self.last_scan_hour}, ensure_ascii=False, indent=1))
        tmp.replace(self.path)

    async def _say(self, text: str) -> None:
        log.info(text)
        if self.notify:
            await self.notify(text)

    # --- раз в час: выходы по закрывшимся свечам, выход по сигналу, поиск новых входов ---
    async def hourly(self, now: float) -> None:
        cfg = self.cfg
        universe = await self.md.universe(cfg.quote, cfg.universe_size, set(cfg.exclude), BTC)
        btc = (await self.md.candles([BTC], cfg.btc_history_h, now)).get(BTC)
        if btc is None or btc.index[-1].timestamp() < now - 3 * 3600:
            await self._say("⚠️ Нет свежих данных по BTC, проверка пропущена")
            return
        need = sorted((set(universe) | {s.symbol for s in self.open}) - {BTC})
        frames = await self.md.candles(need, cfg.history_h, now)
        frames[BTC] = btc.iloc[-cfg.history_h:]
        fresh = {s: f for s, f in frames.items() if f.index[-1] == btc.index[-1]}

        # 1. Стоп и время — по каждой закрывшейся свече после входа (стоп проверяется первым).
        await self._exits_by_candles(frames)
        # 2. Стратегии перестали соглашаться — выход по закрытию часа.
        if cfg.exit_votes:
            for sig in list(self.open):
                df = fresh.get(sig.symbol)
                if df is None or df.index[-1] <= pd.Timestamp(sig.time):
                    continue
                n = agree_count(df, btc.close, sig.side)
                if n < cfg.exit_votes:
                    await self._close(sig, "signal", float(df.close.iloc[-1]), df.index[-1] + pd.Timedelta(hours=1),
                                      extra=f" («за» осталось {n} из 15)")
        # 3. Новые входы.
        ranks = {}
        if cfg.top_k_filter:
            close = pd.DataFrame({s: f.close for s, f in frames.items() if s in universe or s == BTC})
            sc = momentum_vs_btc(close, 168).where(eligible(close, 300)).iloc[-1]
            ranks = sc.rank(ascending=False).dropna().astype(int).to_dict()
        busy = {s.symbol for s in self.open}
        for sym in universe:
            df = fresh.get(sym)
            if df is None or sym in busy:
                continue
            allowed = not cfg.top_k_filter or ranks.get(sym, 999) <= cfg.top_k_filter
            sig = find_signal(sym, df, btc.close, cfg, allowed)
            if sig:
                self.open.append(sig)
                await self._say(format_signal(sig, ranks.get(sym), cfg.exit_votes))

    async def _exits_by_candles(self, frames: dict) -> None:
        for sig in list(self.open):
            f = frames.get(sig.symbol)
            if f is None:
                continue
            for t, row in f[f.index > pd.Timestamp(sig.time)].iterrows():
                hit = check_exit(sig, row.high, row.low, row.close, t + pd.Timedelta(hours=1))
                if hit:
                    await self._close(sig, hit[0], hit[1], t + pd.Timedelta(hours=1))
                    break

    async def _close(self, sig: Signal, kind: str, price: float, when: pd.Timestamp, extra: str = "") -> None:
        close_signal(sig, kind, price, when, self.cfg)
        self.open.remove(sig)
        self.closed.append(sig)
        msg = {"tp": "🎯 Тейк-профит", "sl": "🛑 Стоп-лосс", "time": "⏰ Время вышло — закройте позицию",
               "signal": "⚠️ Стратегии больше не согласны — закройте позицию"}[kind]
        await self._say(f"{msg}: {sig.symbol} {('лонг' if sig.side == 'long' else 'шорт')} по {price:.6g}{extra}\n"
                        f"Результат: {sig.r:+.2f}R ({sig.net_bps:+.0f} б.п. с комиссиями)")

    # --- между часами: только стоп/тейк по текущей цене, чтобы сообщить быстрее ---
    async def quick_check(self, now: float) -> None:
        if not self.open:
            return
        prices = await self.md.prices(sorted({s.symbol for s in self.open}))
        ts = pd.Timestamp(now, unit="s", tz="UTC")
        for sig in list(self.open):
            p = prices.get(sig.symbol)
            if p is None:
                continue
            hit = check_exit(sig, p, p, p, ts)
            if hit and hit[0] != "time":  # выход по времени — только по закрытию часа, как в исследовании
                await self._close(sig, hit[0], hit[1], ts)

    async def step(self, now: float | None = None) -> None:
        now = now or time.time()
        hour = int(now // 3600)
        if self.last_scan_hour != hour:
            await self.hourly(now)
            self.last_scan_hour = hour
        else:
            await self.quick_check(now)
        self.save()

    def status(self) -> str:
        if not self.open:
            return "Открытых сигналов нет."
        return "Открытые сигналы:\n" + "\n".join(
            f"• {s.symbol} {s.side}: вход {s.entry:.6g}, стоп {s.sl:.6g}"
            + (f", тейк {s.tp:.6g}" if s.tp is not None else "")
            + f", закрыть до {pd.Timestamp(s.expires):%d.%m %H:%M} UTC" for s in self.open)

    def stats(self) -> str:
        return stats(self.closed)
