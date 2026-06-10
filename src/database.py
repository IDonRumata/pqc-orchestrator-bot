"""Async database layer built on SQLAlchemy 2.0 and pgvector.

Provides engine and session management, schema bootstrap (extension, tables and
vector indexes) and the high level helpers used by the orchestrator, ingest
script and news monitor.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import datetime

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .config import get_settings
from .logging_config import log_event
from .models import AgentRun, Base, KnowledgeChunk, MemoryFact, MonitoredItem

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Return the lazily created async engine."""
    global _engine, _session_factory
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            echo=False,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=5,
        )
        _session_factory = async_sessionmaker(
            bind=_engine, expire_on_commit=False, class_=AsyncSession
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the configured async session factory."""
    if _session_factory is None:
        get_engine()
    assert _session_factory is not None
    return _session_factory


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Provide a transactional scope around a series of operations."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create the pgvector extension, all tables and the vector indexes."""
    settings = get_settings()
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
        # IVFFlat cosine indexes for both embedding columns.
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_chunks_embedding_local "
                "ON knowledge_chunks USING ivfflat (embedding_local vector_cosine_ops) "
                "WITH (lists = 100)"
            )
        )
        await conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_chunks_embedding_openai "
                "ON knowledge_chunks USING ivfflat (embedding_openai vector_cosine_ops) "
                "WITH (lists = 100)"
            )
        )
    log_event(
        logger,
        logging.INFO,
        "Database initialized",
        dimensions=settings.embedding_dimensions,
    )


async def dispose_engine() -> None:
    """Dispose of the engine on graceful shutdown."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None


# --- Knowledge base operations ---------------------------------------------


async def insert_chunk(
    session: AsyncSession,
    *,
    source: str,
    doc_id: str,
    url: str | None,
    chunk_index: int,
    content: str,
    embedding_local: list[float],
    embedding_openai: list[float] | None,
) -> None:
    """Insert a single knowledge chunk."""
    chunk = KnowledgeChunk(
        source=source,
        doc_id=doc_id,
        url=url,
        chunk_index=chunk_index,
        content=content,
        embedding_local=embedding_local,
        embedding_openai=embedding_openai,
    )
    session.add(chunk)


async def delete_chunks_by_doc(session: AsyncSession, doc_id: str) -> None:
    """Remove all chunks of a document so it can be re-vectorized cleanly."""
    await session.execute(
        text("DELETE FROM knowledge_chunks WHERE doc_id = :doc_id"),
        {"doc_id": doc_id},
    )


async def search_chunks(
    session: AsyncSession,
    *,
    query_vector: list[float],
    provider: str,
    limit: int,
) -> list[KnowledgeChunk]:
    """Cosine search over the chosen embedding column.

    If the preferred provider is openai but no openai vectors are present, the
    search transparently falls back to the local column.
    """
    column = (
        KnowledgeChunk.embedding_openai
        if provider == "openai"
        else KnowledgeChunk.embedding_local
    )
    stmt = (
        select(KnowledgeChunk)
        .where(column.is_not(None))
        .order_by(column.cosine_distance(query_vector))
        .limit(limit)
    )
    result = await session.execute(stmt)
    rows = list(result.scalars().all())
    if not rows and provider == "openai":
        # Fallback to the always populated local column.
        return await search_chunks(
            session,
            query_vector=query_vector,
            provider="local",
            limit=limit,
        )
    return rows


# --- News monitor operations ------------------------------------------------


async def get_known_external_ids(
    session: AsyncSession, external_ids: Sequence[str]
) -> set[str]:
    """Return the subset of external ids already stored."""
    if not external_ids:
        return set()
    stmt = select(MonitoredItem.external_id).where(
        MonitoredItem.external_id.in_(list(external_ids))
    )
    result = await session.execute(stmt)
    return set(result.scalars().all())


async def record_monitored_item(
    session: AsyncSession,
    *,
    source: str,
    external_id: str,
    url: str | None,
    title: str,
    published_at: datetime | None,
    content_hash: str | None,
    is_critical: bool,
) -> None:
    """Upsert a monitored item by its external id."""
    stmt = (
        pg_insert(MonitoredItem)
        .values(
            source=source,
            external_id=external_id,
            url=url,
            title=title,
            published_at=published_at,
            content_hash=content_hash,
            is_critical=is_critical,
        )
        .on_conflict_do_nothing(index_elements=["external_id"])
    )
    await session.execute(stmt)


# --- Statistics for status dashboard --------------------------------------


async def get_chunk_count() -> int:
    """Return the total number of indexed knowledge chunks."""
    async with session_scope() as session:
        result = await session.execute(
            select(func.count()).select_from(KnowledgeChunk)
        )
        return result.scalar_one()


async def get_monitored_count() -> int:
    """Return the total number of stored news monitor items."""
    async with session_scope() as session:
        result = await session.execute(
            select(func.count()).select_from(MonitoredItem)
        )
        return result.scalar_one()


async def get_memory_count() -> int:
    """Return the total number of stored project memory facts."""
    async with session_scope() as session:
        result = await session.execute(
            select(func.count()).select_from(MemoryFact)
        )
        return result.scalar_one()


# --- Project memory operations ---------------------------------------------


async def add_memory_fact(
    session: AsyncSession,
    *,
    kind: str,
    content: str,
    source: str,
    embedding_local: list[float],
    embedding_openai: list[float] | None,
    tags: str | None = None,
) -> None:
    """Persist a single durable project memory fact."""
    fact = MemoryFact(
        kind=kind,
        content=content,
        source=source,
        tags=tags,
        embedding_local=embedding_local,
        embedding_openai=embedding_openai,
    )
    session.add(fact)


async def search_memory_facts(
    session: AsyncSession,
    *,
    query_vector: list[float],
    provider: str,
    limit: int,
) -> list[MemoryFact]:
    """Cosine search over project memory, mirrors search_chunks fallback logic."""
    column = (
        MemoryFact.embedding_openai
        if provider == "openai"
        else MemoryFact.embedding_local
    )
    stmt = (
        select(MemoryFact)
        .where(column.is_not(None))
        .order_by(column.cosine_distance(query_vector))
        .limit(limit)
    )
    result = await session.execute(stmt)
    rows = list(result.scalars().all())
    if not rows and provider == "openai":
        return await search_memory_facts(
            session, query_vector=query_vector, provider="local", limit=limit
        )
    return rows


async def nearest_memory_distance(
    session: AsyncSession,
    *,
    query_vector: list[float],
    provider: str,
) -> float | None:
    """Return the smallest cosine distance to any existing fact, or None if empty.

    Used to deduplicate auto captured facts before insertion.
    """
    column = (
        MemoryFact.embedding_openai
        if provider == "openai"
        else MemoryFact.embedding_local
    )
    stmt = (
        select(column.cosine_distance(query_vector))
        .where(column.is_not(None))
        .order_by(column.cosine_distance(query_vector))
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def list_recent_memory_facts(limit: int = 20) -> list[MemoryFact]:
    """Return the most recently stored memory facts, newest first."""
    async with session_scope() as session:
        stmt = (
            select(MemoryFact)
            .order_by(MemoryFact.created_at.desc())
            .limit(limit)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())


# --- Observability operations ----------------------------------------------


async def record_agent_run(
    session: AsyncSession,
    *,
    user_id: int,
    query: str,
    agents_selected: dict,
    rag_chunks_used: int,
    tokens_total: int,
    duration_ms: int,
) -> None:
    """Persist an orchestration run for auditing and cost analysis."""
    run = AgentRun(
        user_id=user_id,
        query=query,
        agents_selected=agents_selected,
        rag_chunks_used=rag_chunks_used,
        tokens_total=tokens_total,
        duration_ms=duration_ms,
    )
    session.add(run)
