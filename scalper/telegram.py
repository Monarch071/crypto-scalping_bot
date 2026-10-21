"""Уведомления и управление через Telegram: /status, /kill, /resume.

Команды принимаются только из чата TELEGRAM_CHAT_ID. Токен берётся из
переменной окружения TELEGRAM_BOT_TOKEN и никогда не пишется в код.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

import aiohttp

log = logging.getLogger(__name__)

Handler = Callable[[], Awaitable[str]]


class Notifier:
    async def send(self, text: str) -> None:
        log.info("[notify] %s", text)

    async def start(self, commands: dict[str, Handler]) -> None:
        pass

    async def stop(self) -> None:
        pass


class TelegramNotifier(Notifier):
    def __init__(self, token: str, chat_id: str) -> None:
        self._url = f"https://api.telegram.org/bot{token}"
        self._chat_id = str(chat_id)
        self._session: aiohttp.ClientSession | None = None
        self._task: asyncio.Task | None = None
        self._commands: dict[str, Handler] = {}

    async def start(self, commands: dict[str, Handler]) -> None:
        self._commands = commands
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=40))
        self._task = asyncio.create_task(self._poll())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
        if self._session:
            await self._session.close()

    async def send(self, text: str) -> None:
        await super().send(text)
        if self._session is None:
            return
        try:
            await self._session.post(f"{self._url}/sendMessage",
                                     json={"chat_id": self._chat_id, "text": text})
        except Exception as e:  # noqa: BLE001
            log.warning("Telegram send: %s", e)

    async def _poll(self) -> None:
        assert self._session is not None
        offset = 0
        while True:
            try:
                async with self._session.get(f"{self._url}/getUpdates",
                                             params={"offset": offset, "timeout": 30}) as r:
                    data = await r.json()
                for upd in data.get("result", []):
                    offset = upd["update_id"] + 1
                    msg = upd.get("message") or {}
                    if str(msg.get("chat", {}).get("id")) != self._chat_id:
                        continue  # чужие чаты игнорируем
                    cmd = (msg.get("text") or "").split()[0].split("@")[0] if msg.get("text") else ""
                    handler = self._commands.get(cmd)
                    if handler:
                        await self.send(await handler())
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                log.warning("Telegram poll: %s", e)
                await asyncio.sleep(5)
