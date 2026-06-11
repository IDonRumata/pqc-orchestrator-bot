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
    delete_chunks_by_doc,
    get_known_external_ids,
    insert_chunk,
    record_monitored_item,
    session_scope,
)
from .embeddings import EmbeddingRouter
from .logging_config import log_event
from .openrouter_client import ChatMessage, OpenRouterClient
from .prompts import (
    NEWS_CLASSIFIER_SYSTEM_PROMPT,
    NEWS_DIGEST_SYSTEM_PROMPT,
    build_news_classifier_prompt,
    build_news_digest_prompt,
)
from .text_utils import chunk_text, to_telegram_html

logger = logging.getLogger(__name__)

# Async callback used to push notifications to the user (set up by bot.py).
NotifyCallback = Callable[[str], Awaitable[None]]


@dataclass(slots=True)
class FeedEntry:
    """A normalized RSS entry."""

    source: str
    external_id: str
    title: str
    url: str | None
    summary: str
    published_at: datetime | None


@dataclass(slots=True)
class CriticalItem:
    """A critical update collected during one monitoring cycle for the digest."""

    source: str
    title: str
    category: str
    summary: str
    url: str | None


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
        """Run a single polling cycle across all sources.

        Every new item is classified and, if critical, ingested into the
        knowledge base and recorded in the ledger so it is never processed
        again. Notifications are NOT sent per item: all critical items found in
        the cycle are collected and delivered as a single consolidated digest,
        so a noisy feed cannot spam the chat.
        """
        log_event(logger, logging.INFO, "News monitor cycle started")
        critical: list[CriticalItem] = []
        for source_url in self._settings.news_sources:
            try:
                await self._process_feed(source_url, critical)
            except Exception as exc:  # noqa: BLE001 - one bad feed must not stop others
                log_event(
                    logger,
                    logging.WARNING,
                    "Feed processing failed",
                    feed=source_url,
                    error=str(exc),
                )
        if critical:
            await self._send_digest(critical)
        log_event(
            logger, logging.INFO, "News monitor cycle finished",
            critical=len(critical),
        )

    async def _process_feed(
        self, source_url: str, critical: list[CriticalItem]
    ) -> None:
        """Fetch one feed, find new entries and collect the critical ones."""
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
            item = await self._process_entry(entry)
            if item is not None:
                critical.append(item)

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

    async def _process_entry(self, entry: FeedEntry) -> CriticalItem | None:
        """Classify a single entry, ingest it if critical, record it in the ledger.

        Returns a CriticalItem when the entry is critical (for the cycle digest),
        otherwise None. Always records the entry so it is never reprocessed.
        """
        body = await self._fetch_full_text(entry)
        classification = await self._classify(entry.title, body)
        is_critical = bool(classification.get("is_critical"))
        category = str(classification.get("category", "other"))
        summary = str(classification.get("summary", "")).strip()

        content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()

        if is_critical:
            await self._reindex(entry, body, category)

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

        if not is_critical:
            return None
        return CriticalItem(
            source=entry.source,
            title=entry.title.strip(),
            category=category,
            summary=summary or "нет краткого описания",
            url=entry.url,
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

    async def _send_digest(self, items: list[CriticalItem]) -> None:
        """Build and send a single consolidated digest for the whole cycle.

        Tries an LLM written analytical summary, falls back to a plain
        structured list if the model call fails. The knowledge base is already
        updated per item, so the digest is purely a notification.
        """
        intro = await self._write_digest_intro(items)
        header = f"🛰 <b>Сводка мониторинга</b> — обновлений: {len(items)}"
        parts: list[str] = [header]
        if intro:
            parts.append(intro)

        # Compact source list, capped so a huge cycle stays readable.
        cap = max(1, self._settings.news_digest_max_items)
        listed = items[:cap]
        source_lines = ["", "<b>Источники:</b>"]
        for item in listed:
            title = to_telegram_html(item.title)[:160]
            line = f"- [{item.category}] {title}"
            if item.url:
                line += f"\n  {item.url}"
            source_lines.append(line)
        if len(items) > cap:
            source_lines.append(f"...и ещё {len(items) - cap}")
        parts.append("\n".join(source_lines))

        message = "\n\n".join(parts)
        try:
            await self._notify(message)
        except Exception as exc:  # noqa: BLE001 - notification is best effort
            log_event(
                logger, logging.WARNING, "Failed to send digest", error=str(exc)
            )

    async def _write_digest_intro(self, items: list[CriticalItem]) -> str | None:
        """Ask the cheap model for a short analytical intro, None on failure."""
        if not self._settings.news_digest_enabled:
            return None
        payload = [
            {"category": item.category, "title": item.title, "summary": item.summary}
            for item in items
        ]
        messages = [
            ChatMessage(role="system", content=NEWS_DIGEST_SYSTEM_PROMPT),
            ChatMessage(role="user", content=build_news_digest_prompt(payload)),
        ]
        try:
            result = await self._client.chat_completion(
                model=self._settings.news_classifier_model,
                messages=messages,
                temperature=0.2,
            )
            return to_telegram_html(result.content.strip()) or None
        except Exception as exc:  # noqa: BLE001 - fall back to the plain list
            log_event(
                logger, logging.WARNING, "Digest intro generation failed",
                error=str(exc),
            )
            return None

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
