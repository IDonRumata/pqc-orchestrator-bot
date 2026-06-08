"""Hybrid embedding layer.

Two embedding providers are supported and both are reduced to the same vector
size so they remain interchangeable inside pgvector:

* LocalEmbedder  - fastembed, fully offline, zero cost.
* OpenAIEmbedder - text-embedding-3-small reduced to the same dimensions.

At query time an EmbeddingRouter picks the best provider for the given text:
short or simple queries use the free local model, longer and denser queries use
OpenAI when a key is configured. Any OpenAI failure falls back to local so the
bot never breaks. During ingestion both vectors are produced when possible so
the chosen query space always has matching document vectors.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import Settings, get_settings
from .logging_config import log_event

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class QueryEmbedding:
    """A query vector together with the provider that produced it."""

    vector: list[float]
    provider: str


@dataclass(slots=True)
class DocumentEmbedding:
    """Both vectors for a document chunk, openai may be None."""

    local: list[float]
    openai: list[float] | None


class LocalEmbedder:
    """Wraps a fastembed model. The fastembed call is synchronous and CPU bound,
    so it is dispatched to a worker thread to keep the event loop responsive."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None  # Lazily loaded on first use.
        self._lock = asyncio.Lock()

    async def _ensure_model(self) -> None:
        if self._model is not None:
            return
        async with self._lock:
            if self._model is not None:
                return
            # Import here so the heavy dependency is only loaded when needed.
            from fastembed import TextEmbedding

            def _load() -> "TextEmbedding":
                return TextEmbedding(model_name=self._settings.local_embedding_model)

            self._model = await asyncio.to_thread(_load)
            log_event(
                logger,
                logging.INFO,
                "Local embedding model loaded",
                model=self._settings.local_embedding_model,
            )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        await self._ensure_model()
        model = self._model
        assert model is not None

        def _run() -> list[list[float]]:
            return [vector.tolist() for vector in model.embed(texts)]

        return await asyncio.to_thread(_run)


class OpenAIEmbedder:
    """Calls the OpenAI compatible embeddings endpoint, reduced to N dimensions."""

    _RETRYABLE = (httpx.TimeoutException, httpx.TransportError, httpx.NetworkError)

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        timeout = httpx.Timeout(
            settings.http_timeout_seconds,
            connect=settings.http_connect_timeout_seconds,
        )
        self._client = httpx.AsyncClient(
            base_url=settings.openai_base_url,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self._settings.openai_embedding_model,
            "input": texts,
            "dimensions": self._settings.embedding_dimensions,
        }
        retryer = AsyncRetrying(
            stop=stop_after_attempt(self._settings.retry_attempts),
            wait=wait_exponential(
                multiplier=self._settings.retry_min_wait_seconds,
                max=self._settings.retry_max_wait_seconds,
            ),
            retry=retry_if_exception_type(self._RETRYABLE),
            reraise=True,
        )
        async for attempt in retryer:
            with attempt:
                response = await self._client.post("/embeddings", json=payload)
                if response.status_code in (429,) or response.status_code >= 500:
                    raise httpx.TransportError(
                        f"OpenAI embeddings transient error {response.status_code}"
                    )
                response.raise_for_status()
                data = response.json()
                items = sorted(data["data"], key=lambda item: item["index"])
                return [item["embedding"] for item in items]
        raise RuntimeError("OpenAI embeddings retry loop exhausted")


class EmbeddingRouter:
    """Selects the best embedding provider per query and produces both vectors
    for document ingestion."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._local = LocalEmbedder(self._settings)
        self._openai: OpenAIEmbedder | None = (
            OpenAIEmbedder(self._settings)
            if self._settings.openai_embeddings_enabled
            else None
        )

    async def aclose(self) -> None:
        if self._openai is not None:
            await self._openai.aclose()

    def _prefer_openai(self, text: str) -> bool:
        """Heuristic, prefer the higher quality OpenAI embedder for longer or
        denser queries when a key is configured."""
        if self._openai is None:
            return False
        return len(text.strip()) >= self._settings.embedding_quality_threshold

    async def embed_query(self, text: str) -> QueryEmbedding:
        """Embed a single query, choosing the best provider with a safe fallback."""
        if self._prefer_openai(text):
            try:
                vectors = await self._openai.embed([text])  # type: ignore[union-attr]
                log_event(
                    logger,
                    logging.INFO,
                    "Query embedded",
                    provider="openai",
                    chars=len(text),
                )
                return QueryEmbedding(vector=vectors[0], provider="openai")
            except Exception as exc:  # noqa: BLE001 - fallback must be total
                log_event(
                    logger,
                    logging.WARNING,
                    "OpenAI embedding failed, falling back to local",
                    error=str(exc),
                )
        vectors = await self._local.embed([text])
        log_event(
            logger, logging.INFO, "Query embedded", provider="local", chars=len(text)
        )
        return QueryEmbedding(vector=vectors[0], provider="local")

    async def embed_documents(self, texts: list[str]) -> list[DocumentEmbedding]:
        """Embed document chunks for ingestion, computing both vectors when the
        OpenAI provider is available."""
        local_vectors = await self._local.embed(texts)
        openai_vectors: list[list[float]] | None = None
        if self._openai is not None:
            try:
                openai_vectors = await self._openai.embed(texts)
            except Exception as exc:  # noqa: BLE001
                log_event(
                    logger,
                    logging.WARNING,
                    "OpenAI document embedding failed, storing local only",
                    error=str(exc),
                )
                openai_vectors = None
        results: list[DocumentEmbedding] = []
        for index, local_vector in enumerate(local_vectors):
            openai_vector = openai_vectors[index] if openai_vectors is not None else None
            results.append(DocumentEmbedding(local=local_vector, openai=openai_vector))
        return results
