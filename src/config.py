"""Centralized configuration for the PQC orchestrator bot.

All settings are loaded from environment variables (or a local .env file) and
validated through pydantic-settings. Model identifiers are intentionally kept as
configurable strings so they can be swapped without touching the code.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly typed application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Telegram -----------------------------------------------------------
    telegram_bot_token: str = Field(..., description="Telegram Bot API token")
    allowed_user_ids: list[int] = Field(
        default_factory=list,
        description="Whitelist of Telegram user ids allowed to use the bot",
    )

    # --- OpenRouter (LLM provider) -----------------------------------------
    openrouter_api_key: str = Field(..., description="OpenRouter API key")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter compatible base url",
    )
    openrouter_referer: str = Field(
        default="https://localhost",
        description="HTTP-Referer header recommended by OpenRouter",
    )
    openrouter_app_title: str = Field(
        default="PQC Strategic Orchestrator",
        description="X-Title header shown in OpenRouter dashboard",
    )

    # --- OpenAI (optional, only for embeddings) ----------------------------
    openai_api_key: str | None = Field(
        default=None,
        description="Optional OpenAI key, enables high quality embeddings",
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI compatible base url for embeddings",
    )

    # --- Database -----------------------------------------------------------
    database_url: str = Field(
        ...,
        description="Async SQLAlchemy url, e.g. postgresql+asyncpg://user:pass@host/db",
    )

    # --- Model matrix (OpenRouter slugs, 2026 defaults) --------------------
    # Cheap and fast model for routing and news classification.
    router_model: str = Field(default="google/gemini-2.5-flash")
    news_classifier_model: str = Field(default="google/gemini-2.5-flash")
    # Specialist agents.
    cfo_model: str = Field(default="openai/gpt-4o-mini")
    pqc_model: str = Field(default="anthropic/claude-sonnet-4.5")
    legal_model: str = Field(default="anthropic/claude-sonnet-4.5")
    grants_model: str = Field(default="anthropic/claude-sonnet-4.5")
    # Final validator, the most capable model.
    critic_model: str = Field(default="anthropic/claude-opus-4.1")

    # --- Embeddings ---------------------------------------------------------
    embedding_dimensions: int = Field(
        default=384,
        description="Vector size, both local and OpenAI embeddings are reduced to this",
    )
    local_embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="fastembed model name (384 dimensions)",
    )
    openai_embedding_model: str = Field(default="text-embedding-3-small")
    # Queries longer than this many characters prefer the OpenAI embedder when
    # a key is configured, shorter ones use the free local embedder.
    embedding_quality_threshold: int = Field(default=350)

    # --- RAG / chunking -----------------------------------------------------
    chunk_size: int = Field(default=1100, description="Characters per chunk")
    chunk_overlap: int = Field(default=200, description="Character overlap between chunks")
    rag_top_k: int = Field(default=5, description="Chunks retrieved per RAG query")

    # --- Orchestration ------------------------------------------------------
    max_agents_per_request: int = Field(default=3)

    # --- Project memory -----------------------------------------------------
    # Durable facts about what the founder already did / decided, injected into
    # every agent so answers build on prior state instead of repeating it.
    memory_top_k: int = Field(default=6, description="Memory facts injected per query")
    memory_auto_capture: bool = Field(
        default=True,
        description="Extract and store durable facts after each answer",
    )
    memory_dedup_distance: float = Field(
        default=0.12,
        description="Skip auto facts whose cosine distance to an existing fact is below this",
    )

    # --- News monitor -------------------------------------------------------
    news_interval_hours: int = Field(default=12)
    news_initial_delay_seconds: int = Field(default=30)
    news_sources: list[str] = Field(
        default_factory=lambda: [
            # NIST cybersecurity news (verified working, covers FIPS and PQC announcements).
            "https://www.nist.gov/news-events/cybersecurity/rss.xml",
            # NIST publications feed (covers new FIPS standards like FIPS 203/204/205 updates).
            "https://www.nist.gov/news-events/publications/rss.xml",
            # Open Quantum Safe liboqs Python releases (library updates and PQC fixes).
            "https://github.com/open-quantum-safe/liboqs-python/releases.atom",
            # Open Quantum Safe liboqs C library releases (upstream algorithm updates).
            "https://github.com/open-quantum-safe/liboqs/releases.atom",
            # arXiv cs.CR - cryptography research preprints (new attacks, algorithm updates).
            "https://export.arxiv.org/rss/cs.CR",
            # Google Security Blog (TLS PQC deployment, X25519MLKEM768 news).
            "https://security.googleblog.com/feeds/posts/default",
            # Mozilla Security Blog (Firefox TLS, PQC cipher suite changes).
            "https://blog.mozilla.org/security/feed/",
        ],
    )

    # --- HTTP / resilience --------------------------------------------------
    http_timeout_seconds: float = Field(default=120.0)
    http_connect_timeout_seconds: float = Field(default=15.0)
    retry_attempts: int = Field(default=4)
    retry_min_wait_seconds: float = Field(default=2.0)
    retry_max_wait_seconds: float = Field(default=30.0)

    # --- Observability ------------------------------------------------------
    log_level: str = Field(default="INFO")

    @field_validator("allowed_user_ids", "news_sources", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Allow comma separated env strings for list fields.

        Handles three formats pydantic-settings may produce:
        - plain string: "123456789" or "123,456"
        - pre-parsed int/float (when json.loads succeeds on a bare number): 123456789
        - already a list (e.g. when value comes from defaults)
        """
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            return [item.strip() for item in stripped.split(",") if item.strip()]
        if isinstance(value, (int, float)):
            # json.loads("8300011892") returns an int - wrap it back into a list.
            return [value]
        return value

    @property
    def openai_embeddings_enabled(self) -> bool:
        return bool(self.openai_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
