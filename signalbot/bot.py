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
from .logic import Signal, check_exit, close_signal, find_signal, format_signal, from_dict, stats, to_dict

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

    # --- поиск сигналов (раз в час, после закрытия свечи) ---
    async def scan(self, now: float) -> list[Signal]:
        cfg = self.cfg
        universe = await self.md.universe(cfg.quote, cfg.universe_size, set(cfg.exclude), BTC)
        btc = (await self.md.candles([BTC], cfg.btc_history_h, now)).get(BTC)
        if btc is None or btc.index[-1].timestamp() < now - 3 * 3600:
            await self._say("⚠️ Нет свежих данных по BTC, поиск сигналов пропущен")
            return []
        frames = await self.md.candles([s for s in universe if s != BTC], cfg.history_h, now)
        frames[BTC] = btc.iloc[-cfg.history_h:]
        ranks = {}
        if cfg.top_k_filter:
            close = pd.DataFrame({s: f.close for s, f in frames.items()})
            sc = momentum_vs_btc(close, 168).where(eligible(close, 300)).iloc[-1]
            ranks = sc.rank(ascending=False).dropna().astype(int).to_dict()
        busy = {s.symbol for s in self.open}
        found = []
        for sym, df in frames.items():
            if sym in busy or df.index[-1] != btc.index[-1]:
                continue
            allowed = not cfg.top_k_filter or ranks.get(sym, 999) <= cfg.top_k_filter
            sig = find_signal(sym, df, btc.close, cfg, allowed)
            if sig:
                self.open.append(sig)
                found.append(sig)
                await self._say(format_signal(sig, ranks.get(sym)))
        return found

    # --- сопровождение открытых сигналов ---
    async def check_open(self, now: float, use_candles: bool) -> None:
        if not self.open:
            return
        ts = pd.Timestamp(now, unit="s", tz="UTC")
        syms = sorted({s.symbol for s in self.open})
        if use_candles:
            frames = await self.md.candles(syms, self.cfg.max_hold_h + 2, now)
        else:
            prices = await self.md.prices(syms)
        still = []
        for sig in self.open:
            hit = None
            if use_candles and sig.symbol in frames:
                after = frames[sig.symbol][frames[sig.symbol].index > pd.Timestamp(sig.time)]
                for t, row in after.iterrows():
                    hit = check_exit(sig, row.high, row.low, row.close, t + pd.Timedelta(hours=1))
                    if hit:
                        break
            elif not use_candles and sig.symbol in prices:
                p = prices[sig.symbol]
                hit = check_exit(sig, p, p, p, ts)
            if not hit:
                still.append(sig)
                continue
            kind, price = hit
            close_signal(sig, kind, price, ts, self.cfg)
            self.closed.append(sig)
            msg = {"tp": "🎯 Тейк-профит", "sl": "🛑 Стоп-лосс", "time": "⏰ Прошло 24 ч — закройте позицию"}[kind]
            await self._say(f"{msg}: {sig.symbol} {('лонг' if sig.side == 'long' else 'шорт')} "
                            f"по {price:.6g}\nРезультат: {sig.r:+.2f}R ({sig.net_bps:+.0f} б.п. с комиссиями)")
        self.open = still

    async def step(self, now: float | None = None) -> None:
        now = now or time.time()
        hour = int(now // 3600)
        if self.last_scan_hour != hour:
            # Сначала проверяем выходы по закрывшимся свечам, затем ищем новые входы.
            await self.check_open(now, use_candles=True)
            await self.scan(now)
            self.last_scan_hour = hour
        else:
            await self.check_open(now, use_candles=False)
        self.save()

    def status(self) -> str:
        if not self.open:
            return "Открытых сигналов нет."
        return "Открытые сигналы:\n" + "\n".join(
            f"• {s.symbol} {s.side}: вход {s.entry:.6g}, стоп {s.sl:.6g}, тейк {s.tp:.6g}, "
            f"до {pd.Timestamp(s.expires):%d.%m %H:%M} UTC" for s in self.open)

    def stats(self) -> str:
        return stats(self.closed)
