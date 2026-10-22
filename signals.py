"""Бот сигналов для ручной торговли: 15 стратегий, вход при совпадении, стоп сразу, выход по сообщению бота.

  python signals.py run              # искать сигналы и присылать в Telegram
  python signals.py status           # открытые сигналы
  python signals.py stats            # статистика всех закрытых сигналов
  python signals.py replay data/panel_1h.parquet --start 2026-07-01 --end 2026-09-30   # прогон по истории

Ключи API не нужны. Нужны TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID, иначе сообщения пишутся в лог.
Настройки: signals.example.toml → signals.toml (-c signals.toml).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import time

from rotator.market import MarketData
from signalbot.bot import SignalBot
from signalbot.config import SignalConfig, load_signal_config

log = logging.getLogger("signals")


async def cmd_run(cfg: SignalConfig) -> None:
    import ccxt.async_support as ccxt

    from rotate import make_notifier

    ex = getattr(ccxt, cfg.exchange)({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    md = MarketData(ex)
    notifier = make_notifier()
    bot = SignalBot(cfg, md, notifier.send)

    async def status() -> str:
        return bot.status()

    async def stats() -> str:
        return bot.stats()

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass
    try:
        await md.load()
        await notifier.start({"/status": status, "/stats": stats})
        exit_rule = f"тейк {cfg.tp_atr} ATR" if cfg.tp_atr > 0 else f"выход, когда «за» < {cfg.exit_votes}/15"
        await notifier.send(f"▶️ Бот сигналов запущен: вход при {cfg.min_votes}/15"
                            + (f" в топ-{cfg.top_k_filter} по силе" if cfg.top_k_filter else "")
                            + (f", стоп {cfg.sl_pct}%" if cfg.sl_pct > 0 else f", стоп {cfg.sl_atr} ATR")
                            + f", {exit_rule}, максимум {cfg.max_hold_h} ч")
        while not stop.is_set():
            try:
                await bot.step()
            except Exception as e:  # noqa: BLE001
                log.exception("Ошибка шага")
                await notifier.send(f"⚠️ Ошибка: {e}")
            # Просыпаемся каждые check_minutes и сразу после закрытия часовой свечи.
            now = time.time()
            to_hour = 3600 - now % 3600 + 30
            delay = min(cfg.check_minutes * 60, to_hour)
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
    finally:
        await notifier.stop()
        await ex.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Бот сигналов для ручной торговли")
    ap.add_argument("-c", "--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    sub.add_parser("status")
    sub.add_parser("stats")
    rp = sub.add_parser("replay")
    rp.add_argument("panel")
    rp.add_argument("--start", required=True)
    rp.add_argument("--end", required=True)
    args = ap.parse_args()
    cfg = load_signal_config(args.config)
    logging.basicConfig(level=cfg.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "run":
        asyncio.run(cmd_run(cfg))
    elif args.cmd == "status":
        print(SignalBot(cfg, None).status())
    elif args.cmd == "replay":
        from signalbot.replay import load_frames, replay

        bot = replay(load_frames(args.panel), cfg, args.start, args.end)
        for s in bot.closed:
            print(f"{s.time[:16]} {s.symbol:10s} {s.side} → {s.status:6s} {s.exit_time[:16]}  {s.r:+.2f}R")
        print(bot.stats())
    else:
        print(SignalBot(cfg, None).stats())


if __name__ == "__main__":
    main()
