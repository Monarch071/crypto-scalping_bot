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
    ex = getattr(ccxt, cfg.exchange)(params)
    if cfg.testnet:
        ex.set_sandbox_mode(True)
    return ex


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


def cmd_replay(cfg: RotationConfig, panel: str, start: str, end: str | None) -> None:
    from rotator.backtest import load_panel, run
    from rotator.replay import replay

    close, qvol = load_panel(panel)
    eq, trader = replay(close, qvol, cfg, start, end)
    ret, _ = run(close, qvol, cfg, cfg.fee_bps + 5.0)
    bt = (1 + ret.loc[eq.index[0]:eq.index[-1]]).prod() - 1
    trades = trader.state.get("trades", 0)
    print(f"Прогон живого кода {eq.index[0]:%Y-%m-%d} — {eq.index[-1]:%Y-%m-%d}: "
          f"{cfg.paper_start_usdt:.0f} → {eq.iloc[-1]:.2f} {cfg.quote} ({eq.iloc[-1] / cfg.paper_start_usdt - 1:+.1%}), "
          f"макс. просадка {(1 - eq / eq.cummax()).max():.1%}, сделок {trades}")
    print(f"Векторный бэктест того же периода: {bt:+.1%}")
    print(f"Расхождение: {(eq.iloc[-1] / cfg.paper_start_usdt - 1) - bt:+.1%} "
          f"(допустимо несколько % из-за полосы ребаланса и порядка исполнения)")
    print("Последние события:\n" + "\n".join(trader.state["log"][-8:]))


async def cmd_preflight(cfg: RotationConfig) -> None:
    """Проверка перед запуском на реальные деньги. Ничего не покупает и не продаёт."""
    ok = True

    def mark(good: bool, text: str) -> None:
        nonlocal ok
        ok &= good
        print(("✅ " if good else "❌ ") + text)

    mark(cfg.mode == "live", f"mode = {cfg.mode!r} (для реальных денег нужно 'live')")
    mark(cfg.max_capital_usdt > 0, f"потолок капитала max_capital_usdt = {cfg.max_capital_usdt}")
    ex = make_exchange(cfg, private=True)
    try:
        await ex.load_markets()
        server = await ex.fetch_time()
        drift = abs(server / 1000 - time.time())
        mark(drift < 1.0, f"часы компьютера расходятся с биржей на {drift:.2f} с")
        bal = await ex.fetch_balance()
        usdt = float((bal.get("total") or {}).get(cfg.quote, 0.0))
        mark(usdt >= cfg.max_capital_usdt * 0.99, f"на счёте {usdt:.2f} {cfg.quote}, потолок бота {cfg.max_capital_usdt}")
        if cfg.exchange == "binance" and not cfg.testnet:
            r = await ex.sapi_get_account_apirestrictions()
            mark(not r.get("enableWithdrawals"), "у ключа API запрещён вывод средств")
            mark(bool(r.get("enableSpotAndMarginTrading")), "у ключа API разрешена спотовая торговля")
            mark(bool(r.get("ipRestrict")), "ключ API привязан к IP-адресу (рекомендуется)")
        md = MarketData(ex)
        uni = await md.universe(cfg.quote, cfg.universe_size, set(cfg.exclude), "BTC/USDT")
        per_coin = cfg.max_capital_usdt * cfg.capital_fraction / cfg.top_k
        worst = max((ex.markets[s].get("limits", {}).get("cost", {}).get("min") or 0.0) for s in uni)
        mark(per_coin >= max(2 * worst, cfg.min_trade_usdt * 2),
             f"на монету ≈{per_coin:.2f} {cfg.quote}, минимальный ордер биржи до {worst} {cfg.quote}")
        mark(len(uni) >= 2 * cfg.top_k, f"монет в отборе: {len(uni)}")
    except Exception as e:  # noqa: BLE001
        mark(False, f"ошибка обращения к бирже: {e}")
    finally:
        await ex.close()
    n = make_notifier()
    await n.start({})
    await n.send("✅ Проверка связи с ботом ротации")
    await n.stop()
    print("\nИТОГ: " + ("можно запускать" if ok else "есть проблемы — исправьте перед запуском"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Бот ротации монет по моментуму")
    ap.add_argument("-c", "--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--live", action="store_true")
    r.add_argument("--once", action="store_true")
    sub.add_parser("status")
    sub.add_parser("preflight")
    rp = sub.add_parser("replay")
    rp.add_argument("panel")
    rp.add_argument("--start", required=True)
    rp.add_argument("--end", default=None)
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
    elif args.cmd == "preflight":
        asyncio.run(cmd_preflight(cfg))
    elif args.cmd == "replay":
        cmd_replay(cfg, args.panel, args.start, args.end)
    else:
        cmd_backtest(cfg, args.panel, args.cost_bps)


if __name__ == "__main__":
    main()
