"""Two layer background monitor for laws and grant programs.

Layer 1: every N hours poll the configured RSS feeds, diff against the database
and download new items.
Layer 2: classify each new item with a fast model. Critical items are chunked,
re-vectorized into the knowledge base and the user is notified in Telegram.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

import feedparser
import httpx

from .config import Settings, get_settings
from .database import (
    add_memory_fact,
    delete_chunks_by_doc,
    get_known_external_ids,
    insert_chunk,
    record_monitored_item,
    session_scope,
)
from .embeddings import EmbeddingRouter
from .logging_config import log_event
from .openrouter_client import ChatMessage, OpenRouterClient
from .prompts import NEWS_CLASSIFIER_SYSTEM_PROMPT, build_news_classifier_prompt
from .text_utils import chunk_text

logger = logging.getLogger(__name__)

# Async callback used to push notifications to the user (set up by bot.py).
NotifyCallback = Callable[[str], Awaitable[None]]

_CRITICAL_MESSAGE = (
    "Обнаружено критическое обновление грантовой программы / закона. "
    "Локальная база данных успешно актуализирована.\n\n"
    "Источник: {source}\n"
    "Заголовок: {title}\n"
    "Категория: {category}\n"
    "Суть: {summary}\n"
    "Ссылка: {url}"
)


@dataclass(slots=True)
class FeedEntry:
    """A normalized RSS entry."""

    source: str
    external_id: str
    title: str
    url: str | None
    summary: str
    published_at: datetime | None


class NewsMonitor:
    """Background worker that keeps the knowledge base current."""

    def __init__(
        self,
        client: OpenRouterClient,
        embedder: EmbeddingRouter,
        notify: NotifyCallback,
        settings: Settings | None = None,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._notify = notify
        self._settings = settings or get_settings()
        self._stopped = asyncio.Event()
        timeout = httpx.Timeout(60.0, connect=self._settings.http_connect_timeout_seconds)
        self._http = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "PQC-Orchestrator-Monitor/1.0"},
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    def stop(self) -> None:
        """Signal the loop to exit on the next iteration."""
        self._stopped.set()

    async def run_forever(self) -> None:
        """Main loop, runs until stop() is called."""
        # Small initial delay so the bot finishes startup first.
        try:
            await asyncio.wait_for(
                self._stopped.wait(), timeout=self._settings.news_initial_delay_seconds
            )
            return
        except asyncio.TimeoutError:
            pass

        interval = self._settings.news_interval_hours * 3600
        while not self._stopped.is_set():
            try:
                await self.run_once()
            except Exception as exc:  # noqa: BLE001 - the loop must survive any error
                log_event(
                    logger, logging.ERROR, "News monitor cycle failed", error=str(exc)
                )
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue

    async def run_once(self) -> None:
        """Run a single polling cycle across all sources."""
        log_event(logger, logging.INFO, "News monitor cycle started")
        for source_url in self._settings.news_sources:
            try:
                await self._process_feed(source_url)
            except Exception as exc:  # noqa: BLE001 - one bad feed must not stop others
                log_event(
                    logger,
                    logging.WARNING,
                    "Feed processing failed",
                    feed=source_url,
                    error=str(exc),
                )
        log_event(logger, logging.INFO, "News monitor cycle finished")

    async def _process_feed(self, source_url: str) -> None:
        """Fetch one feed, find new entries and process the critical ones."""
        entries = await self._fetch_feed(source_url)
        if not entries:
            return

        external_ids = [entry.external_id for entry in entries]
        async with session_scope() as session:
            known = await get_known_external_ids(session, external_ids)
        new_entries = [e for e in entries if e.external_id not in known]

        if not new_entries:
            return

        log_event(
            logger,
            logging.INFO,
            "New feed entries found",
            feed=source_url,
            count=len(new_entries),
        )

        for entry in new_entries:
            await self._process_entry(entry)

    async def _fetch_feed(self, source_url: str) -> list[FeedEntry]:
        """Download and parse a single RSS feed."""
        response = await self._http.get(source_url)
        response.raise_for_status()
        # feedparser is synchronous, run it off the event loop.
        parsed = await asyncio.to_thread(feedparser.parse, response.content)
        feed_title = getattr(parsed.feed, "title", source_url)

        entries: list[FeedEntry] = []
        for raw in parsed.entries:
            link = getattr(raw, "link", None)
            guid = getattr(raw, "id", None) or link or getattr(raw, "title", "")
            if not guid:
                continue
            entries.append(
                FeedEntry(
                    source=str(feed_title)[:128],
                    external_id=str(guid)[:1024],
                    title=str(getattr(raw, "title", "Без заголовка")),
                    url=link,
                    summary=str(getattr(raw, "summary", "")),
                    published_at=self._parse_date(raw),
                )
            )
        return entries

    @staticmethod
    def _parse_date(raw: Any) -> datetime | None:
        parsed = getattr(raw, "published_parsed", None) or getattr(
            raw, "updated_parsed", None
        )
        if parsed is None:
            return None
        try:
            return datetime(*parsed[:6], tzinfo=timezone.utc)
        except (TypeError, ValueError):
            return None

    async def _process_entry(self, entry: FeedEntry) -> None:
        """Classify a single entry and update the knowledge base if critical."""
        body = await self._fetch_full_text(entry)
        classification = await self._classify(entry.title, body)
        is_critical = bool(classification.get("is_critical"))
        category = str(classification.get("category", "other"))
        summary = str(classification.get("summary", "")).strip()

        content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()

        if is_critical:
            await self._reindex(entry, body, category)
            await self._store_news_memory(entry, summary, category)

        async with session_scope() as session:
            await record_monitored_item(
                session,
                source=entry.source,
                external_id=entry.external_id,
                url=entry.url,
                title=entry.title,
                published_at=entry.published_at,
                content_hash=content_hash,
                is_critical=is_critical,
            )

        log_event(
            logger,
            logging.INFO,
            "Entry processed",
            source=entry.source,
            external_id=entry.external_id,
            is_critical=is_critical,
            category=category,
        )

        if is_critical:
            message = _CRITICAL_MESSAGE.format(
                source=entry.source,
                title=entry.title,
                category=category,
                summary=summary or "нет краткого описания",
                url=entry.url or "нет ссылки",
            )
            try:
                await self._notify(message)
            except Exception as exc:  # noqa: BLE001
                log_event(
                    logger,
                    logging.WARNING,
                    "Failed to send critical notification",
                    error=str(exc),
                )

    async def _fetch_full_text(self, entry: FeedEntry) -> str:
        """Try to download the full article text, fall back to the RSS summary."""
        if not entry.url:
            return entry.summary
        try:
            response = await self._http.get(entry.url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "html" in content_type or "xml" in content_type or not content_type:
                return self._strip_html(response.text)
            return entry.summary
        except Exception as exc:  # noqa: BLE001 - summary is an acceptable fallback
            log_event(
                logger,
                logging.DEBUG,
                "Full text fetch failed, using summary",
                url=entry.url,
                error=str(exc),
            )
            return entry.summary

    @staticmethod
    def _strip_html(html: str) -> str:
        """Very small HTML to text reducer without extra dependencies."""
        import re

        text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
        text = re.sub(r"(?s)<[^>]+>", " ", text)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"&lt;", "<", text)
        text = re.sub(r"&gt;", ">", text)
        return re.sub(r"\s+", " ", text).strip()

    async def _classify(self, title: str, body: str) -> dict[str, Any]:
        """Use the fast model to judge whether the item is critical."""
        messages = [
            ChatMessage(role="system", content=NEWS_CLASSIFIER_SYSTEM_PROMPT),
            ChatMessage(role="user", content=build_news_classifier_prompt(title, body)),
        ]
        try:
            data, _tokens = await self._client.chat_json(
                model=self._settings.news_classifier_model,
                messages=messages,
                temperature=0.0,
                max_tokens=400,
            )
            return data
        except Exception as exc:  # noqa: BLE001
            log_event(
                logger,
                logging.WARNING,
                "Classification failed, treating as non critical",
                error=str(exc),
            )
            return {"is_critical": False, "category": "other", "summary": ""}

    async def _store_news_memory(
        self, entry: FeedEntry, summary: str, category: str
    ) -> None:
        """Record a critical news item as a project memory fact (kind=news).

        This keeps the assistant aware of fresh developments in its long term
        memory, not only in the RAG index, so it can reference them proactively.
        """
        gist = (summary or entry.title).strip()
        content = f"Новость ({category}): {entry.title.strip()}. {gist}"[:1000]
        try:
            embedding = (await self._embedder.embed_documents([content]))[0]
            async with session_scope() as session:
                await add_memory_fact(
                    session,
                    kind="news",
                    content=content,
                    source="news",
                    tags=category,
                    embedding_local=embedding.local,
                    embedding_openai=embedding.openai,
                )
            log_event(
                logger, logging.INFO, "News stored in project memory",
                source=entry.source, category=category,
            )
        except Exception as exc:  # noqa: BLE001 - memory write is best effort
            log_event(
                logger, logging.WARNING, "Failed to store news memory",
                error=str(exc),
            )

    async def _reindex(self, entry: FeedEntry, body: str, category: str) -> None:
        """Chunk, embed and store a critical document, replacing stale chunks."""
        chunks = chunk_text(
            body,
            chunk_size=self._settings.chunk_size,
            overlap=self._settings.chunk_overlap,
        )
        if not chunks:
            return
        embeddings = await self._embedder.embed_documents(chunks)
        source_label = f"monitor:{category}"
        async with session_scope() as session:
            await delete_chunks_by_doc(session, entry.external_id)
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
                await insert_chunk(
                    session,
                    source=source_label,
                    doc_id=entry.external_id,
                    url=entry.url,
                    chunk_index=index,
                    content=chunk,
                    embedding_local=embedding.local,
                    embedding_openai=embedding.openai,
                )
        log_event(
            logger,
            logging.INFO,
            "Knowledge base updated from critical item",
            source=entry.source,
            chunks=len(chunks),
        )
