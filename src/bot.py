"""Telegram entry point built on aiogram 3.x.

Full-featured UX: persistent reply keyboard, inline document browser,
command menu registered in BotFather, agent info, status dashboard,
typing indicators, long-message splitting and graceful shutdown.
"""
from __future__ import annotations

import asyncio
import logging
import signal

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import (
    BotCommand,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)
from aiogram.utils.chat_action import ChatActionSender

from .config import Settings, get_settings
from .database import (
    dispose_engine,
    get_chunk_count,
    get_monitored_count,
    init_db,
)
from .embeddings import EmbeddingRouter
from .logging_config import configure_logging, log_event
from .news_monitor import NewsMonitor
from .openrouter_client import OpenRouterClient
from .orchestrator import Orchestrator

logger = logging.getLogger(__name__)

_TELEGRAM_LIMIT = 4096

# ---------------------------------------------------------------------------
# Reply keyboard — persistent bottom panel
# ---------------------------------------------------------------------------

_BTN_DOCS = "📚 База знаний"
_BTN_AGENTS = "🤖 Агенты"
_BTN_NEWS = "📰 Новости"
_BTN_STATUS = "📊 Статус"
_BTN_HELP = "ℹ️ Помощь"

_REPLY_BUTTONS = {_BTN_DOCS, _BTN_AGENTS, _BTN_NEWS, _BTN_STATUS, _BTN_HELP}


def _main_keyboard() -> ReplyKeyboardMarkup:
    """Persistent navigation keyboard shown at the bottom of every chat."""
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text=_BTN_DOCS), KeyboardButton(text=_BTN_AGENTS)],
            [KeyboardButton(text=_BTN_NEWS), KeyboardButton(text=_BTN_STATUS)],
            [KeyboardButton(text=_BTN_HELP)],
        ],
        resize_keyboard=True,
        persistent=True,
        input_field_placeholder="Задай вопрос по PQC стратегии...",
    )


# ---------------------------------------------------------------------------
# Document knowledge-base catalog
# ---------------------------------------------------------------------------

_DOCS: list[dict] = [
    {
        "id": "nist_203",
        "emoji": "🔐",
        "name": "NIST FIPS 203 (ML-KEM)",
        "group": "nist",
        "desc": (
            "<b>Module-Lattice Key Encapsulation Mechanism</b>\n"
            "Стандарт NIST для квантово-устойчивого обмена ключами.\n"
            "Замена RSA-KEM и ECDH в TLS, VPN, e-mail шифровании.\n"
            "Уровни: ML-KEM-512, ML-KEM-768, ML-KEM-1024."
        ),
        "query": (
            "Объясни NIST FIPS 203 ML-KEM: алгоритм, уровни безопасности, "
            "как внедрить в продукт стартапа, практические примеры применения"
        ),
    },
    {
        "id": "nist_204",
        "emoji": "🔐",
        "name": "NIST FIPS 204 (ML-DSA)",
        "group": "nist",
        "desc": (
            "<b>Module-Lattice Digital Signature Algorithm</b>\n"
            "Стандарт NIST для квантово-устойчивых цифровых подписей.\n"
            "Замена RSA-PSS и ECDSA. Для подписания кода, документов, API.\n"
            "Уровни: ML-DSA-44, ML-DSA-65, ML-DSA-87."
        ),
        "query": (
            "Объясни NIST FIPS 204 ML-DSA: как заменяет RSA/ECDSA, "
            "размеры ключей, производительность, применение в финтех продуктах"
        ),
    },
    {
        "id": "nist_205",
        "emoji": "🔐",
        "name": "NIST FIPS 205 (SLH-DSA)",
        "group": "nist",
        "desc": (
            "<b>Stateless Hash-Based Digital Signature Algorithm</b>\n"
            "Консервативный стандарт подписи на хеш-деревьях (бывший SPHINCS+).\n"
            "Используй когда важна максимальная долгосрочная надежность.\n"
            "Медленнее ML-DSA, но не зависит от сложности решеточных задач."
        ),
        "query": (
            "Объясни NIST FIPS 205 SLH-DSA: когда выбирать вместо ML-DSA, "
            "компромиссы производительности и безопасности, практические сценарии"
        ),
    },
    {
        "id": "nis2",
        "emoji": "⚖️",
        "name": "NIS2 Директива 2022/2555",
        "group": "legal",
        "desc": (
            "<b>EU Directive 2022/2555 — Network and Information Security</b>\n"
            "Требования кибербезопасности для критической инфраструктуры ЕС.\n"
            "Статья 21: криптографические меры обязательны.\n"
            "Штрафы: до 10 млн EUR или 2% годового оборота."
        ),
        "query": (
            "Как NIS2 Директива требует применения PQC криптографии? "
            "Что конкретно нужно нашему стартапу для соответствия и какие сроки?"
        ),
    },
    {
        "id": "dora",
        "emoji": "⚖️",
        "name": "DORA Регламент 2022/2554",
        "group": "legal",
        "desc": (
            "<b>Digital Operational Resilience Act</b>\n"
            "Регламент ЕС о цифровой устойчивости финансового сектора.\n"
            "Статья 9: требования к шифрованию в финтех, банках, страховых.\n"
            "В силе с января 2025. Распространяется на ICT-поставщиков для финсектора."
        ),
        "query": (
            "Как DORA регламент влияет на PQC стартап, работающий с финтех клиентами? "
            "Требования к криптографии, что нужно задокументировать?"
        ),
    },
    {
        "id": "enisa",
        "emoji": "🛡️",
        "name": "ENISA PQC Руководство",
        "group": "legal",
        "desc": (
            "<b>ENISA Post-Quantum Cryptography Guidelines</b>\n"
            "Официальные рекомендации Агентства ЕС по кибербезопасности.\n"
            "Временные рамки перехода, приоритеты алгоритмов, угроза HNDL.\n"
            "Рекомендует ML-KEM, ML-DSA, SLH-DSA как приоритетные."
        ),
        "query": (
            "Что рекомендует ENISA по переходу на PQC? "
            "Конкретные временные рамки и приоритеты алгоритмов для нашего стартапа"
        ),
    },
    {
        "id": "horizon",
        "emoji": "💰",
        "name": "Horizon Europe / EIC Accelerator",
        "group": "grants",
        "desc": (
            "<b>EIC Accelerator — Horizon Europe Cluster 3</b>\n"
            "До 2.5 млн EUR невозвратных грантов + до 15 млн EUR equity.\n"
            "Кибербезопасность — приоритетная тема Cluster 3.\n"
            "Конкурс 3 раза в год. Уровень конкуренции высокий."
        ),
        "query": (
            "Как подать заявку на Horizon Europe EIC Accelerator для PQC стартапа? "
            "Требования, что писать в заявке, стратегия победы, типичные ошибки"
        ),
    },
    {
        "id": "dep",
        "emoji": "💰",
        "name": "Digital Europe Programme",
        "group": "grants",
        "desc": (
            "<b>Digital Europe Programme (DEP) / ECCC</b>\n"
            "Программа ЕС для кибербезопасности и цифровизации.\n"
            "Гранты для SME до 2 млн EUR. Возможность стека с Horizon Europe.\n"
            "Управляется через ECCC (Европейский центр кибербезопасности)."
        ),
        "query": (
            "Расскажи о грантах Digital Europe Programme для PQC кибербезопасности. "
            "Как получить, как совместить с Horizon Europe?"
        ),
    },
    {
        "id": "polska",
        "emoji": "💰",
        "name": "Польские гранты (NCBR/PARP/FENG)",
        "group": "grants",
        "desc": (
            "<b>Polska: NCBR, PARP, FENG — Sciezka SMART</b>\n"
            "Sciezka SMART (FENG 2021-2027): до 80% субсидирования R&D.\n"
            "NCBR: исследовательские проекты в кибербезопасности.\n"
            "PARP: развитие компаний, до 3 млн PLN."
        ),
        "query": (
            "Как польскому стартапу получить гранты NCBR, PARP, FENG для PQC? "
            "Пошаговая стратегия, размеры финансирования, сроки подачи"
        ),
    },
]

_DOC_BY_ID: dict[str, dict] = {d["id"]: d for d in _DOCS}

_GROUP_LABELS: dict[str, str] = {
    "nist": "🔐 NIST FIPS Стандарты",
    "legal": "⚖️ Регуляторика ЕС",
    "grants": "💰 Гранты и финансирование",
}


def _docs_keyboard() -> InlineKeyboardMarkup:
    """Inline keyboard grouping all knowledge-base documents by category."""
    rows: list[list[InlineKeyboardButton]] = []
    current_group: str | None = None
    for doc in _DOCS:
        if doc["group"] != current_group:
            current_group = doc["group"]
            rows.append([
                InlineKeyboardButton(
                    text=f"  {_GROUP_LABELS[current_group]}  ",
                    callback_data="noop",
                )
            ])
        rows.append([
            InlineKeyboardButton(
                text=f"{doc['emoji']} {doc['name']}",
                callback_data=f"doc:{doc['id']}",
            )
        ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def _doc_actions_keyboard(doc_id: str) -> InlineKeyboardMarkup:
    """Action buttons shown below a document description."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(
            text="🔍 Спросить всех агентов",
            callback_data=f"ask_doc:{doc_id}",
        )],
        [InlineKeyboardButton(
            text="◀️ Назад к документам",
            callback_data="back_docs",
        )],
    ])


# ---------------------------------------------------------------------------
# Static message texts
# ---------------------------------------------------------------------------

_WELCOME = """\
👋 Привет, Андрей!

Я <b>PQC Strategic Orchestrator</b> — мультиагентный AI-советник по стратегии \
постквантовой криптографии для стартапа в ЕС/Польше.

<b>Мои специалисты:</b>
🔬 <b>PQC Ученый</b> — NIST FIPS 203/204/205, ML-KEM, ML-DSA, SLH-DSA
💼 <b>CFO</b> — финансы, runway, юнит-экономика
⚖️ <b>EU Юрист</b> — NIS2, DORA, ENISA
💰 <b>Грантовый эксперт</b> — Horizon, NCBR, PARP, FENG
🎯 <b>Chief Critic</b> — финальный синтез при 2+ агентах

<b>Как использовать:</b> напиши любой вопрос обычным текстом — я сам выберу нужных агентов.
Или используй <b>меню ниже</b> для навигации по документам и функциям.\
"""

_HELP = """\
📖 <b>Как работать с ботом</b>

<b>Задать вопрос:</b> напиши обычным текстом, например:
- "Какие алгоритмы внедрить первыми в 2026?"
- "Как NIS2 влияет на мой продукт?"
- "Рассчитай runway при burn rate $15k/мес"
- "Как подать на EIC Accelerator?"

<b>Что происходит внутри:</b>
1. Роутер (Gemini Flash) анализирует запрос
2. Выбирает 1–3 профильных агента
3. Агенты ищут контекст в базе знаний (RAG)
4. Параллельно формируют ответы
5. Chief Critic синтезирует финал (если агентов 2+)

<b>Кнопки меню:</b>
📚 <b>База знаний</b> — 9 документов с описанием и прямым запросом к агентам
🤖 <b>Агенты</b> — описание каждого специалиста
📰 <b>Новости</b> — последние PQC-новости из мониторинга 7 RSS-лент
📊 <b>Статус</b> — размер базы знаний, модели, конфигурация

<b>Фоновый мониторинг</b> проверяет RSS каждые 12ч и присылает алерты \
о критических изменениях в законах, стандартах и грантах.\
"""

_AGENTS = """\
🤖 <b>Агенты системы</b>

🔬 <b>PQC Ученый</b>
Алгоритмы ML-KEM, ML-DSA, SLH-DSA, liboqs, NIST FIPS 203/204/205.
Угрозы HNDL, переход с RSA/ECC, архитектура гибридных схем.

💼 <b>CFO (Финансовый директор)</b>
Юнит-экономика, burn rate, runway, ценообразование B2B SaaS.
Стоимость внедрения PQC, ROI модель, финансовые сценарии.

⚖️ <b>EU Legal (Юрист ЕС)</b>
NIS2, DORA, ENISA, eIDAS 2.0, польское право кибербезопасности.
Требования к криптографии, сроки соответствия, риски штрафов.

💰 <b>Grants Expert (Грантовый эксперт)</b>
Horizon Europe EIC, Digital Europe, NCBR, PARP, FENG, Sciezka SMART.
Стратегия стека грантов, подготовка заявок, критерии оценки.

🎯 <b>Chief Critic (Главный критик)</b>
Активируется автоматически только при 2+ агентах.
Синтез ответов, выявление противоречий, финальная проверка.

<i>Роутер автоматически выбирает 1–3 агентов по каждому запросу. \
Ручной выбор не нужен.</i>\
"""


# ---------------------------------------------------------------------------
# Bot application
# ---------------------------------------------------------------------------


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
        return bool(allowed) and user_id in allowed

    def _user_id(self, message: Message) -> int | None:
        return message.from_user.id if message.from_user else None

    # --- Notifications & sending -------------------------------------------

    async def _notify_users(self, text: str) -> None:
        """Push a notification to every whitelisted user."""
        for uid in self._settings.allowed_user_ids:
            try:
                await self._send_long(uid, text)
            except Exception as exc:  # noqa: BLE001
                log_event(
                    logger, logging.WARNING, "Failed to notify user",
                    user_id=uid, error=str(exc),
                )

    async def _send_long(self, chat_id: int, text: str) -> None:
        """Send, splitting on newlines to respect the Telegram 4096-char limit."""
        for part in _split_message(text, _TELEGRAM_LIMIT):
            await self._bot.send_message(chat_id, part)

    # --- Status text builder ------------------------------------------------

    async def _build_status(self) -> str:
        try:
            chunks = await get_chunk_count()
            news = await get_monitored_count()
        except Exception:  # noqa: BLE001
            chunks = news = -1
        embed_mode = (
            "OpenAI + fastembed (гибрид)"
            if self._settings.openai_embeddings_enabled
            else "fastembed локальный (бесплатно)"
        )
        return (
            "📊 <b>Статус системы</b>\n\n"
            f"🗄️ <b>База знаний:</b> {chunks} чанков\n"
            f"📰 <b>Мониторинг новостей:</b> {news} записей\n"
            f"🧠 <b>Эмбеддинги:</b> {embed_mode}\n\n"
            "<b>Модели агентов:</b>\n"
            f"  Роутер: <code>{self._settings.router_model}</code>\n"
            f"  PQC:    <code>{self._settings.pqc_model}</code>\n"
            f"  CFO:    <code>{self._settings.cfo_model}</code>\n"
            f"  Юрист:  <code>{self._settings.legal_model}</code>\n"
            f"  Гранты: <code>{self._settings.grants_model}</code>\n"
            f"  Критик: <code>{self._settings.critic_model}</code>\n\n"
            f"🕐 <b>Мониторинг RSS:</b> каждые {self._settings.news_interval_hours}ч\n"
            f"📡 <b>Источников RSS:</b> {len(self._settings.news_sources)}"
        )

    # --- Handlers -----------------------------------------------------------

    def _register_handlers(self) -> None:  # noqa: PLR0915 — many small handlers
        router = Router()

        # -- Guard helper (inline) -------------------------------------------
        def allowed(uid: int | None) -> bool:
            return self._is_allowed(uid)

        # -- Commands --------------------------------------------------------

        @router.message(Command("start"))
        async def on_start(message: Message) -> None:
            if not allowed(self._user_id(message)):
                await message.answer("Доступ запрещен.")
                return
            await message.answer(_WELCOME, reply_markup=_main_keyboard())

        @router.message(Command("help"))
        async def on_help(message: Message) -> None:
            if not allowed(self._user_id(message)):
                await message.answer("Доступ запрещен.")
                return
            await message.answer(_HELP, reply_markup=_main_keyboard())

        @router.message(Command("docs"))
        async def on_docs_cmd(message: Message) -> None:
            if not allowed(self._user_id(message)):
                await message.answer("Доступ запрещен.")
                return
            await message.answer(
                "📚 <b>База знаний</b>\n\nВыбери документ для просмотра "
                "или быстрого запроса к агентам:",
                reply_markup=_docs_keyboard(),
            )

        @router.message(Command("agents"))
        async def on_agents_cmd(message: Message) -> None:
            if not allowed(self._user_id(message)):
                await message.answer("Доступ запрещен.")
                return
            await message.answer(_AGENTS)

        @router.message(Command("status"))
        async def on_status_cmd(message: Message) -> None:
            if not allowed(self._user_id(message)):
                await message.answer("Доступ запрещен.")
                return
            await message.answer(await self._build_status())

        @router.message(Command("menu"))
        async def on_menu_cmd(message: Message) -> None:
            if not allowed(self._user_id(message)):
                await message.answer("Доступ запрещен.")
                return
            await message.answer(
                "Главное меню - используй кнопки ниже или напиши вопрос:",
                reply_markup=_main_keyboard(),
            )

        # -- Reply keyboard buttons ------------------------------------------

        @router.message(F.text == _BTN_DOCS)
        async def on_btn_docs(message: Message) -> None:
            if not allowed(self._user_id(message)):
                return
            await message.answer(
                "📚 <b>База знаний</b>\n\nВыбери документ для просмотра "
                "или быстрого запроса к агентам:",
                reply_markup=_docs_keyboard(),
            )

        @router.message(F.text == _BTN_AGENTS)
        async def on_btn_agents(message: Message) -> None:
            if not allowed(self._user_id(message)):
                return
            await message.answer(_AGENTS)

        @router.message(F.text == _BTN_NEWS)
        async def on_btn_news(message: Message) -> None:
            if not allowed(self._user_id(message)):
                return
            uid = self._user_id(message)
            await self._handle_query(
                message, uid,  # type: ignore[arg-type]
                "Summarize the latest critical PQC news and developments from the monitored "
                "sources. What are the most important updates in post-quantum cryptography, "
                "new NIST standards, and regulatory changes that affect our startup?",
            )

        @router.message(F.text == _BTN_STATUS)
        async def on_btn_status(message: Message) -> None:
            if not allowed(self._user_id(message)):
                return
            await message.answer(await self._build_status())

        @router.message(F.text == _BTN_HELP)
        async def on_btn_help(message: Message) -> None:
            if not allowed(self._user_id(message)):
                return
            await message.answer(_HELP)

        # -- Inline callbacks ------------------------------------------------

        @router.callback_query(F.data == "noop")
        async def on_noop(query: CallbackQuery) -> None:
            """Section-header buttons — dismiss the spinner silently."""
            await query.answer()

        @router.callback_query(F.data == "back_docs")
        async def on_back_docs(query: CallbackQuery) -> None:
            await query.answer()
            if query.message:
                await query.message.edit_text(
                    "📚 <b>База знаний</b>\n\nВыбери документ для просмотра "
                    "или быстрого запроса к агентам:",
                    reply_markup=_docs_keyboard(),
                )

        @router.callback_query(F.data.startswith("doc:"))
        async def on_doc_select(query: CallbackQuery) -> None:
            await query.answer()
            doc_id = (query.data or "").split(":", 1)[1]
            doc = _DOC_BY_ID.get(doc_id)
            if not doc or not query.message:
                return
            text = (
                f"{doc['emoji']} <b>{doc['name']}</b>\n\n"
                f"{doc['desc']}\n\n"
                "Выбери действие:"
            )
            await query.message.edit_text(
                text, reply_markup=_doc_actions_keyboard(doc_id)
            )

        @router.callback_query(F.data.startswith("ask_doc:"))
        async def on_ask_doc(query: CallbackQuery) -> None:
            if not allowed(query.from_user.id if query.from_user else None):
                await query.answer("Доступ запрещен.")
                return
            await query.answer("Запрашиваю агентов...")
            doc_id = (query.data or "").split(":", 1)[1]
            doc = _DOC_BY_ID.get(doc_id)
            if not doc or not query.message:
                return
            uid = query.from_user.id if query.from_user else 0
            chat_id = query.message.chat.id
            # Send a placeholder so user sees progress
            placeholder = await self._bot.send_message(
                chat_id,
                f"🔍 Запрашиваю агентов по документу <b>{doc['name']}</b>...",
            )
            try:
                async with ChatActionSender.typing(bot=self._bot, chat_id=chat_id):
                    result = await self._orchestrator.handle(uid, doc["query"])
                header = self._format_header(result)
                full = f"{header}\n\n{result.answer}"
                await placeholder.delete()
                await self._send_long(chat_id, full)
            except Exception as exc:  # noqa: BLE001
                log_event(
                    logger, logging.ERROR, "ask_doc failed",
                    doc_id=doc_id, error=str(exc),
                )
                await placeholder.edit_text(
                    "Ошибка при обработке запроса. Попробуй ещё раз."
                )

        # -- Generic text handler (must be last) -----------------------------

        @router.message()
        async def on_message(message: Message) -> None:
            uid = self._user_id(message)
            if not allowed(uid):
                await message.answer("Доступ запрещен.")
                return
            text = (message.text or "").strip()
            if not text:
                await message.answer("Пришли текстовый запрос, пожалуйста.")
                return
            # Ignore if somehow a button label slipped through
            if text in _REPLY_BUTTONS:
                return
            await self._handle_query(message, uid, text)  # type: ignore[arg-type]

        self._dispatcher.include_router(router)

    # --- Orchestrator invocation --------------------------------------------

    async def _handle_query(
        self, message: Message, user_id: int, text: str
    ) -> None:
        """Run the orchestrator with a typing indicator, then send the result."""
        try:
            async with ChatActionSender.typing(
                bot=self._bot, chat_id=message.chat.id
            ):
                result = await self._orchestrator.handle(user_id, text)
        except Exception as exc:  # noqa: BLE001
            log_event(
                logger, logging.ERROR, "Query handling failed",
                user_id=user_id, error=str(exc),
            )
            await message.answer(
                "Произошла ошибка при обработке запроса. Попробуй ещё раз чуть позже."
            )
            return
        header = self._format_header(result)
        await self._send_long(message.chat.id, f"{header}\n\n{result.answer}")

    @staticmethod
    def _format_header(result) -> str:  # type: ignore[no-untyped-def]
        agents = ", ".join(result.agents_used) if result.agents_used else "нет"
        critic = "да" if result.critic_used else "нет"
        return (
            f"<i>Агенты: {agents} | Критик: {critic} | "
            f"RAG: {result.rag_chunks_used} чанков | "
            f"Токены: {result.tokens_total} | {result.duration_ms} мс</i>"
        )

    # --- Lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Initialize DB, register bot commands and start polling."""
        await init_db()
        await self._bot.set_my_commands([
            BotCommand(command="start",   description="Главное меню"),
            BotCommand(command="docs",    description="База знаний — документы"),
            BotCommand(command="agents",  description="Список агентов"),
            BotCommand(command="status",  description="Статус системы"),
            BotCommand(command="help",    description="Как пользоваться"),
            BotCommand(command="menu",    description="Показать меню"),
        ])
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def _amain() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = BotApplication(settings)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _request_stop() -> None:
        stop_event.set()

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
