"""Telegram entry point built on aiogram 3.x.

Wires together configuration, database, the OpenRouter client, the hybrid
embedder, the orchestrator and the background news monitor. Includes access
control, typing indicators, long message splitting and graceful shutdown.
"""
from __future__ import annotations

import asyncio
import logging
import signal

from aiogram import Bot, Dispatcher, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender

from .config import Settings, get_settings
from .database import dispose_engine, init_db
from .embeddings import EmbeddingRouter
from .logging_config import configure_logging, log_event
from .news_monitor import NewsMonitor
from .openrouter_client import OpenRouterClient
from .orchestrator import Orchestrator

logger = logging.getLogger(__name__)

_TELEGRAM_LIMIT = 4096

_WELCOME = (
    "Привет. Я мультиагентный оркестратор для стратегии стартапа в сфере "
    "постквантовой криптографии и кибербезопасности (ЕС и Польша).\n\n"
    "Просто опиши задачу или мысль. Я сам выберу нужных экспертов:\n"
    "- Финансовый директор (юнит-экономика, runway)\n"
    "- Ученый-криптограф (PQC, NIST, ML-KEM, ML-DSA)\n"
    "- Юрист по комплаенсу ЕС (NIS2, DORA, ENISA)\n"
    "- Эксперт по грантам ЕС и Польши (Horizon, NCBR, PARP, FENG)\n\n"
    "Команды:\n"
    "/start - это сообщение\n"
    "/help - как пользоваться"
)

_HELP = (
    "Сформулируй запрос обычным текстом. Система проанализирует его, подключит "
    "1-3 релевантных агента, при необходимости поднимет тексты законов и "
    "грантов из локальной базы знаний и, если агентов несколько, прогонит "
    "ответ через Главного Критика для финальной проверки.\n\n"
    "Фоновый мониторинг сам уведомит тебя о критических изменениях в законах "
    "и грантовых программах."
)


class BotApplication:
    """Owns the lifecycle of all shared resources."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._bot = Bot(
            token=settings.telegram_bot_token,
            default=DefaultBotProperties(parse_mode=ParseMode.HTML),
        )
        self._dispatcher = Dispatcher()
        self._client = OpenRouterClient(settings)
        self._embedder = EmbeddingRouter(settings)
        self._orchestrator = Orchestrator(self._client, self._embedder, settings)
        self._monitor = NewsMonitor(
            self._client, self._embedder, self._notify_users, settings
        )
        self._monitor_task: asyncio.Task[None] | None = None
        self._register_handlers()

    # --- Access control -----------------------------------------------------

    def _is_allowed(self, user_id: int | None) -> bool:
        if user_id is None:
            return False
        allowed = self._settings.allowed_user_ids
        # An empty whitelist means the bot is locked down, deny everyone.
        return bool(allowed) and user_id in allowed

    # --- Notifications ------------------------------------------------------

    async def _notify_users(self, text: str) -> None:
        """Push a notification to every whitelisted user."""
        for user_id in self._settings.allowed_user_ids:
            try:
                await self._send_long(user_id, text)
            except Exception as exc:  # noqa: BLE001
                log_event(
                    logger,
                    logging.WARNING,
                    "Failed to notify user",
                    user_id=user_id,
                    error=str(exc),
                )

    async def _send_long(self, chat_id: int, text: str) -> None:
        """Send a message, splitting it to respect the Telegram length limit."""
        for part in _split_message(text, _TELEGRAM_LIMIT):
            await self._bot.send_message(chat_id, part)

    # --- Handlers -----------------------------------------------------------

    def _register_handlers(self) -> None:
        router = Router()

        @router.message(Command("start"))
        async def on_start(message: Message) -> None:
            if not self._is_allowed(message.from_user.id if message.from_user else None):
                await message.answer("Доступ запрещен.")
                return
            await message.answer(_WELCOME)

        @router.message(Command("help"))
        async def on_help(message: Message) -> None:
            if not self._is_allowed(message.from_user.id if message.from_user else None):
                await message.answer("Доступ запрещен.")
                return
            await message.answer(_HELP)

        @router.message()
        async def on_message(message: Message) -> None:
            user = message.from_user
            user_id = user.id if user else None
            if not self._is_allowed(user_id):
                await message.answer("Доступ запрещен.")
                return
            text = (message.text or "").strip()
            if not text:
                await message.answer("Пришли текстовый запрос, пожалуйста.")
                return
            await self._handle_query(message, user_id, text)  # type: ignore[arg-type]

        self._dispatcher.include_router(router)

    async def _handle_query(self, message: Message, user_id: int, text: str) -> None:
        """Run the orchestrator while showing a typing indicator."""
        try:
            async with ChatActionSender.typing(
                bot=self._bot, chat_id=message.chat.id
            ):
                result = await self._orchestrator.handle(user_id, text)
        except Exception as exc:  # noqa: BLE001 - never leak a traceback to the user
            log_event(
                logger,
                logging.ERROR,
                "Query handling failed",
                user_id=user_id,
                error=str(exc),
            )
            await message.answer(
                "Произошла ошибка при обработке запроса. Попробуй еще раз чуть позже."
            )
            return

        header = self._format_header(result)
        await self._send_long(message.chat.id, f"{header}\n\n{result.answer}")

    @staticmethod
    def _format_header(result) -> str:
        """Build a short technical header describing the run."""
        agents = ", ".join(result.agents_used) if result.agents_used else "нет"
        critic = "да" if result.critic_used else "нет"
        return (
            f"Агенты: {agents} | Критик: {critic} | "
            f"RAG чанков: {result.rag_chunks_used} | "
            f"Токенов: {result.tokens_total} | {result.duration_ms} мс"
        )

    # --- Lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Initialize resources and start polling plus the background monitor."""
        await init_db()
        self._monitor_task = asyncio.create_task(self._monitor.run_forever())
        log_event(logger, logging.INFO, "Bot started, polling")
        await self._dispatcher.start_polling(self._bot, handle_signals=False)

    async def shutdown(self) -> None:
        """Gracefully tear down all resources."""
        log_event(logger, logging.INFO, "Shutdown initiated")
        self._monitor.stop()
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        await self._dispatcher.stop_polling()
        await self._monitor.aclose()
        await self._client.aclose()
        await self._embedder.aclose()
        await self._bot.session.close()
        await dispose_engine()
        log_event(logger, logging.INFO, "Shutdown complete")


def _split_message(text: str, limit: int) -> list[str]:
    """Split a long message into chunks under the Telegram limit on line breaks."""
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = window.rfind("\n")
        if cut <= 0:
            cut = limit
        parts.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip("\n")
    if remaining:
        parts.append(remaining)
    return parts


async def _amain() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = BotApplication(settings)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _request_stop() -> None:
        stop_event.set()

    # Register signal handlers where supported (POSIX). Windows ignores this.
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except (NotImplementedError, AttributeError):
            pass

    runner = asyncio.create_task(app.start())
    stopper = asyncio.create_task(stop_event.wait())
    done, _pending = await asyncio.wait(
        {runner, stopper}, return_when=asyncio.FIRST_COMPLETED
    )
    await app.shutdown()
    if runner in done:
        # Surface any startup or polling exception.
        runner.result()
    else:
        runner.cancel()
        try:
            await runner
        except asyncio.CancelledError:
            pass


def main() -> None:
    """Synchronous entry point used by the container and CLI."""
    try:
        asyncio.run(_amain())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
