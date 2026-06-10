"""SQLAlchemy 2.0 ORM models, including pgvector backed knowledge chunks."""
from __future__ import annotations

from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .config import get_settings

_DIM = get_settings().embedding_dimensions


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class KnowledgeChunk(Base):
    """A single retrievable chunk of a law, regulation or grant document.

    Two embedding columns are stored so the runtime can pick either the local
    or the OpenAI vector space at query time without breaking cosine search
    consistency. Both vectors share the same dimensionality.
    """

    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    doc_id: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_local: Mapped[list[float]] = mapped_column(Vector(_DIM), nullable=False)
    embedding_openai: Mapped[list[float] | None] = mapped_column(
        Vector(_DIM), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )


class MonitoredItem(Base):
    """A news or document item discovered by the background monitor."""

    __tablename__ = "monitored_items"
    __table_args__ = (UniqueConstraint("external_id", name="uq_monitored_external_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    external_id: Mapped[str] = mapped_column(String(1024), nullable=False)
    url: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_critical: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )


class MemoryFact(Base):
    """A durable fact about project state for long term assistant memory.

    Stores what the founder has already done, decided, prefers or discussed, so
    future answers build on prior state instead of repeating it. Carries the same
    two embedding columns as KnowledgeChunk for semantic retrieval, the openai
    vector is optional and the local vector is always present.
    """

    __tablename__ = "memory_facts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # done | decision | preference | discussion | news | note
    kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True, default="note")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tags: Mapped[str | None] = mapped_column(String(512), nullable=True)
    # manual | auto | news
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="manual")
    embedding_local: Mapped[list[float]] = mapped_column(Vector(_DIM), nullable=False)
    embedding_openai: Mapped[list[float] | None] = mapped_column(
        Vector(_DIM), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )


class AgentRun(Base):
    """Audit log of every orchestration run for observability and cost tracking."""

    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    agents_selected: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    rag_chunks_used: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tokens_total: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
