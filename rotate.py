"""Бот ротации: держит монеты с самым сильным моментумом, пока BTC в тренде.

  python rotate.py run                       # paper: реальные цены Bybit, виртуальные деньги
  python rotate.py run --once                # один шаг (например, из cron раз в час)
  python rotate.py run --live                # реальные деньги (нужен mode = "live" в конфиге)
  python rotate.py status
  python rotate.py backtest data/panel_1h.parquet

Настройки: rotation.example.toml → rotation.toml (-c rotation.toml).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time

from rotator.config import RotationConfig, load_rotation_config
from rotator.market import LiveAccount, MarketData, PaperAccount
from rotator.trader import Trader

log = logging.getLogger("rotator")


def make_exchange(cfg: RotationConfig, private: bool):
    import ccxt.async_support as ccxt

    params = {"enableRateLimit": True, "options": {"defaultType": "spot"}}
    if private:
        if not cfg.api_key or not cfg.api_secret:
            sys.exit("Нужны переменные окружения BOT_API_KEY и BOT_API_SECRET")
        params.update(apiKey=cfg.api_key, secret=cfg.api_secret)
    return getattr(ccxt, cfg.exchange)(params)


def make_notifier():
    from scalper.config import Config
    from scalper.telegram import Notifier, TelegramNotifier

    c = Config()
    return TelegramNotifier(c.telegram_token, c.telegram_chat_id) if c.telegram_token and c.telegram_chat_id \
        else Notifier()


async def cmd_run(cfg: RotationConfig, live: bool, once: bool) -> None:
    if live != (cfg.mode == "live"):
        sys.exit('Для реальной торговли нужны и mode = "live" в конфиге, и флаг --live')
    ex = make_exchange(cfg, private=live)
    md = MarketData(ex)
    notifier = make_notifier()
    trader = Trader(cfg, md, None, notifier.send)
    if live:
        trader.account = LiveAccount(ex)
    else:
        bal = trader.state.setdefault("paper_balances", {cfg.quote: cfg.paper_start_usdt})
        trader.account = PaperAccount(bal, cfg.fee_bps)

    async def status() -> str:
        return trader.status()

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass
    try:
        await md.load()
        await notifier.start({"/status": status, "/kill": lambda: trader.kill("команда /kill"),
                              "/resume": trader.resume})
        if not once:
            await notifier.send(f"▶️ Бот ротации запущен ({cfg.mode}), пересмотр каждые {cfg.rebalance_h} ч")
        while True:
            try:
                await trader.step()
            except Exception as e:  # noqa: BLE001
                log.exception("Ошибка шага")
                await notifier.send(f"⚠️ Ошибка: {e}")
            if once:
                break
            # Следующий шаг — через минуту после закрытия часовой свечи.
            delay = 3600 - time.time() % 3600 + 60
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay)
                break
            except asyncio.TimeoutError:
                pass
    finally:
        await notifier.stop()
        await ex.close()


def cmd_backtest(cfg: RotationConfig, panel: str, cost: float | None) -> None:
    from rotator.backtest import load_panel, report, run

    close, qvol = load_panel(panel)
    ret, w = run(close, qvol, cfg, cost)
    print(report(ret[ret.index.year >= 2021], w[w.index.year >= 2021]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Бот ротации монет по моментуму")
    ap.add_argument("-c", "--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--live", action="store_true")
    r.add_argument("--once", action="store_true")
    sub.add_parser("status")
    bt = sub.add_parser("backtest")
    bt.add_argument("panel")
    bt.add_argument("--cost-bps", type=float, default=None)
    args = ap.parse_args()
    cfg = load_rotation_config(args.config)
    logging.basicConfig(level=cfg.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "run":
        asyncio.run(cmd_run(cfg, args.live, args.once))
    elif args.cmd == "status":
        print(Trader(cfg, None, None).status())
    else:
        cmd_backtest(cfg, args.panel, args.cost_bps)


if __name__ == "__main__":
    main()
