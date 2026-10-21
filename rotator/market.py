"""Доступ к бирже: рыночные данные (публичные) и счёт (настоящий или бумажный)."""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger(__name__)

STABLES = {"USDT", "USDC", "FDUSD", "DAI", "TUSD", "USDE", "PYUSD", "USDD", "BUSD", "EUR", "USD1", "RLUSD",
           "USDP", "GUSD", "FRAX", "LUSD", "XAUT", "PAXG", "WBTC", "STETH", "WETH", "BTCB"}
LEVERAGED = re.compile(r"\d+[LS]$|(UP|DOWN|BULL|BEAR)$")
HOUR_MS = 3_600_000


@dataclass
class Fill:
    amount: float      # базовая валюта
    price: float       # средняя цена
    fee_usdt: float


class MarketData:
    """Публичные данные через ccxt (async). Работает и в paper, и в live."""

    def __init__(self, exchange) -> None:
        self.ex = exchange
        self._sem = asyncio.Semaphore(5)

    async def load(self) -> None:
        await self.ex.load_markets()

    def market(self, symbol: str) -> dict:
        return self.ex.markets[symbol]

    async def universe(self, quote: str, n: int, exclude: set[str], btc: str) -> list[str]:
        """Топ-n спотовых пар к quote по объёму за 24 ч, без стейблкоинов и токенов с плечом."""
        tickers = await self.ex.fetch_tickers()
        rows = []
        for sym, t in tickers.items():
            m = self.ex.markets.get(sym)
            if not m or not m.get("spot") or not m.get("active", True) or m.get("quote") != quote:
                continue
            base = m.get("base", "")
            if base in STABLES or LEVERAGED.search(base) or sym in exclude:
                continue
            qv = t.get("quoteVolume") or 0.0
            rows.append((qv, sym))
        top = [s for _, s in sorted(rows, reverse=True)[:n]]
        if btc not in top:
            top.append(btc)
        return top

    async def prices(self, symbols: list[str]) -> dict[str, float]:
        tickers = await self.ex.fetch_tickers(symbols)
        return {s: float(t["last"]) for s, t in tickers.items() if t.get("last")}

    async def _ohlcv(self, symbol: str, since_ms: int, now_ms: int) -> list[list]:
        out: list[list] = []
        async with self._sem:
            while since_ms < now_ms:
                batch = await self.ex.fetch_ohlcv(symbol, "1h", since=since_ms, limit=1000)
                if not batch:
                    break
                out += batch
                nxt = batch[-1][0] + HOUR_MS
                if nxt <= since_ms:
                    break
                since_ms = nxt
        return out

    async def closes(self, symbols: list[str], hours: int, now: float | None = None) -> pd.DataFrame:
        """Цены закрытия последних закрытых часовых свечей: строки — часы UTC, колонки — пары."""
        now_ms = int((now or time.time()) * 1000)
        since = (now_ms // HOUR_MS - hours) * HOUR_MS
        data = await asyncio.gather(*(self._ohlcv(s, since, now_ms) for s in symbols), return_exceptions=True)
        cols = {}
        for sym, rows in zip(symbols, data):
            if isinstance(rows, Exception):
                log.warning("Не удалось загрузить %s: %s", sym, rows)
                continue
            # Текущая незакрытая свеча не используется: иначе сигнал «подсматривает» внутрь часа.
            closed = {r[0]: r[4] for r in rows if r[0] + HOUR_MS <= now_ms}
            cols[sym] = pd.Series(closed, dtype=float)
        df = pd.DataFrame(cols)
        df.index = pd.to_datetime(df.index, unit="ms", utc=True)
        full = pd.date_range(df.index.min(), df.index.max(), freq="h")
        return df.reindex(full).ffill(limit=3).sort_index()


class Account:
    async def balances(self) -> dict[str, float]:
        raise NotImplementedError

    async def trade(self, symbol: str, side: str, amount: float, last: float, limit: float) -> Fill:
        raise NotImplementedError


class LiveAccount(Account):
    """Настоящий спотовый счёт через ccxt. Только лимитные IOC-ордера: цена не хуже limit."""

    def __init__(self, exchange) -> None:
        self.ex = exchange

    async def balances(self) -> dict[str, float]:
        b = await self.ex.fetch_balance()
        return {k: float(v) for k, v in (b.get("total") or {}).items() if v}

    async def trade(self, symbol: str, side: str, amount: float, last: float, limit: float) -> Fill:
        amount = float(self.ex.amount_to_precision(symbol, amount))
        price = float(self.ex.price_to_precision(symbol, limit))
        if amount <= 0:
            return Fill(0.0, last, 0.0)
        o = await self.ex.create_order(symbol, "limit", side, amount, price, {"timeInForce": "IOC"})
        if o.get("filled") is None or o.get("status") not in ("closed", "canceled", "cancelled", "expired"):
            await asyncio.sleep(1)
            o = await self.ex.fetch_order(o["id"], symbol)
        filled = float(o.get("filled") or 0.0)
        avg = float(o.get("average") or o.get("price") or price)
        fee = 0.0
        for f in o.get("fees") or ([o["fee"]] if o.get("fee") else []):
            if not f or f.get("cost") is None:
                continue
            fee += float(f["cost"]) * (1.0 if f.get("currency") == "USDT" else avg)
        return Fill(filled, avg, fee)


class PaperAccount(Account):
    """Бумажный счёт: исполнение по последней цене с проскальзыванием и комиссией, баланс в state."""

    def __init__(self, balances: dict[str, float], fee_bps: float, slippage_bps: float = 5.0) -> None:
        self.bal = balances
        self.fee_bps = fee_bps
        self.slip = slippage_bps

    async def balances(self) -> dict[str, float]:
        return {k: v for k, v in self.bal.items() if v > 1e-12}

    async def trade(self, symbol: str, side: str, amount: float, last: float, limit: float) -> Fill:
        base, quote = symbol.split("/")
        px = last * (1 + self.slip / 1e4) if side == "buy" else last * (1 - self.slip / 1e4)
        if (side == "buy" and px > limit) or (side == "sell" and px < limit):
            return Fill(0.0, px, 0.0)
        if side == "sell":
            amount = min(amount, self.bal.get(base, 0.0))
        else:
            amount = min(amount, self.bal.get(quote, 0.0) / (px * (1 + self.fee_bps / 1e4)))
        if amount <= 0:
            return Fill(0.0, px, 0.0)
        cost = amount * px
        fee = cost * self.fee_bps / 1e4
        sign = 1 if side == "buy" else -1
        self.bal[base] = self.bal.get(base, 0.0) + sign * amount
        self.bal[quote] = self.bal.get(quote, 0.0) - sign * cost - fee
        return Fill(amount, px, fee)
