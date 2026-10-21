"""Точка входа.

  python main.py backtest --synthetic --hours 6
  python main.py backtest --data data/btc.jsonl.gz
  python main.py record --minutes 60 --out data/btc.jsonl.gz
  python main.py run                      # paper trading на реальных данных
  python main.py run --live               # реальные деньги (нужен mode = "live" в конфиге)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

from scalper.backtest import run_backtest
from scalper.config import Config, load_config
from scalper.engine import Engine
from scalper.gateways.live import LiveGateway
from scalper.gateways.paper import PaperGateway
from scalper.marketdata import CcxtFeed, Recorder, replay, synthetic
from scalper.telegram import Notifier, TelegramNotifier

log = logging.getLogger("scalper")


def make_exchange(cfg: Config, private: bool):
    import ccxt.pro as ccxtpro

    params = {"enableRateLimit": True, "options": {"defaultType": "swap"}}
    if private:
        if not cfg.api_key or not cfg.api_secret:
            sys.exit("Нужны переменные окружения BOT_API_KEY и BOT_API_SECRET")
        params.update(apiKey=cfg.api_key, secret=cfg.api_secret)
    ex = getattr(ccxtpro, cfg.exchange.name)(params)
    # Paper и запись данных работают на настоящем рынке: в стакане testnet нет реальной ликвидности.
    if private and cfg.exchange.testnet:
        ex.set_sandbox_mode(True)
    return ex


async def cmd_run(cfg: Config, live: bool) -> None:
    if live and cfg.mode != "live":
        sys.exit('Для --live в конфиге должно быть mode = "live"')
    if cfg.mode == "live" and not live:
        sys.exit("В конфиге mode = \"live\": запустите с флагом --live, чтобы подтвердить реальную торговлю")

    ex = make_exchange(cfg, private=live)
    gateway = LiveGateway(ex, cfg.exchange) if live else PaperGateway(cfg.exchange, cfg.paper.latency_ms)
    notifier: Notifier = (TelegramNotifier(cfg.telegram_token, cfg.telegram_chat_id)
                          if cfg.telegram_token and cfg.telegram_chat_id else Notifier())
    recorder = Recorder(cfg.record_path) if cfg.record_path else None
    engine = Engine(cfg, gateway, notifier, recorder, realtime=True)
    feed = CcxtFeed(ex, cfg.exchange.symbol, cfg.exchange.depth, engine.emit)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # Windows
            pass

    async def status() -> str:
        return engine.status()

    async def ticker() -> None:
        n = 0
        while True:
            await asyncio.sleep(1)
            await engine.on_tick()
            n += 1
            if n % 60 == 0:
                log.info("STATUS\n%s", engine.status())

    await engine.start()
    if live:
        pos = await gateway.fetch_position()
        if abs(pos) > 0:
            await gateway.stop()
            await ex.close()
            sys.exit(f"На бирже уже открыта позиция {pos}. Закройте её вручную перед запуском.")
    await notifier.start({"/status": status, "/kill": engine.kill, "/resume": engine.resume})
    await feed.start()
    tasks = [asyncio.create_task(engine.run_forever()), asyncio.create_task(ticker())]
    await notifier.send(f"▶️ Бот запущен ({'LIVE' if live else 'paper'}) {cfg.exchange.symbol}")
    try:
        await stop.wait()
    finally:
        log.info("Останавливаюсь…")
        await engine.kill("остановка процесса")
        # Даём время снять ордера и закрыть позицию (если flatten_on_kill).
        for _ in range(20):
            await asyncio.sleep(0.5)
            await engine.on_tick()
            if not engine.oms.working() and abs(engine.portfolio.position) < 1e-12:
                break
        for t in tasks:
            t.cancel()
        await feed.stop()
        await gateway.stop()
        await notifier.send(f"⏹ Бот остановлен\n{engine.status()}")
        await notifier.stop()
        if recorder:
            recorder.close()
        await ex.close()


async def cmd_record(cfg: Config, minutes: float, out: str) -> None:
    ex = make_exchange(cfg, private=False)
    rec = Recorder(out)
    feed = CcxtFeed(ex, cfg.exchange.symbol, cfg.exchange.depth, rec.write)
    await feed.start()
    log.info("Пишу %s в %s на %.0f мин", cfg.exchange.symbol, out, minutes)
    try:
        await asyncio.sleep(minutes * 60)
    finally:
        await feed.stop()
        rec.close()
        await ex.close()


async def cmd_backtest(cfg: Config, args) -> None:
    if args.data:
        events = replay(args.data)
    else:
        events = synthetic(int(args.hours * 3600), seed=args.seed,
                           start_price=60000.0, tick=cfg.exchange.tick_size)
    print(await run_backtest(cfg, events))


def main() -> None:
    p = argparse.ArgumentParser(description="Скальпинг-бот (маркет-мейкинг)")
    p.add_argument("-c", "--config", default=None, help="путь к config.toml")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--live", action="store_true", help="торговать реальными деньгами")
    rec = sub.add_parser("record")
    rec.add_argument("--minutes", type=float, default=60)
    rec.add_argument("--out", default="data/market.jsonl.gz")
    bt = sub.add_parser("backtest")
    bt.add_argument("--data")
    bt.add_argument("--synthetic", action="store_true")
    bt.add_argument("--hours", type=float, default=6)
    bt.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(level=cfg.log_level if args.cmd != "backtest" else "WARNING",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "run":
        asyncio.run(cmd_run(cfg, args.live))
    elif args.cmd == "record":
        asyncio.run(cmd_record(cfg, args.minutes, args.out))
    else:
        if not args.data and not args.synthetic:
            p.error("укажите --data FILE или --synthetic")
        asyncio.run(cmd_backtest(cfg, args))


if __name__ == "__main__":
    main()
