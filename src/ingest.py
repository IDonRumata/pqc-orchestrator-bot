"""Command line ingestion of laws, regulations and grant documents into the RAG store.

Usage:
    python -m src.ingest --path knowledge_base
    python -m src.ingest --path knowledge_base/nis2.pdf --source legal

Supported formats: .txt, .md and .pdf. Each document is chunked with overlap,
embedded with both providers when available and stored in PostgreSQL.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from .config import get_settings
from .database import delete_chunks_by_doc, dispose_engine, init_db, insert_chunk, session_scope
from .embeddings import EmbeddingRouter
from .logging_config import configure_logging, log_event
from .text_utils import chunk_text

logger = logging.getLogger(__name__)

_SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf_file(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf_file(path)
    return _read_text_file(path)


def _collect_files(target: Path) -> list[Path]:
    if target.is_file():
        return [target] if target.suffix.lower() in _SUPPORTED_SUFFIXES else []
    return sorted(
        p
        for p in target.rglob("*")
        if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES
    )


async def ingest_path(target: Path, source: str) -> None:
    """Ingest a single file or a whole directory tree."""
    settings = get_settings()
    files = _collect_files(target)
    if not files:
        log_event(logger, logging.WARNING, "No supported files found", path=str(target))
        return

    embedder = EmbeddingRouter(settings)
    try:
        await init_db()
        total_chunks = 0
        for file_path in files:
            doc_id = str(file_path.as_posix())
            try:
                raw_text = await asyncio.to_thread(_load_document, file_path)
            except Exception as exc:  # noqa: BLE001 - skip unreadable files
                log_event(
                    logger,
                    logging.ERROR,
                    "Failed to read document",
                    file=doc_id,
                    error=str(exc),
                )
                continue

            chunks = chunk_text(
                raw_text,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )
            if not chunks:
                log_event(logger, logging.WARNING, "Empty document skipped", file=doc_id)
                continue

            embeddings = await embedder.embed_documents(chunks)
            async with session_scope() as session:
                # Replace any previous version of this document.
                await delete_chunks_by_doc(session, doc_id)
                for index, (chunk, embedding) in enumerate(
                    zip(chunks, embeddings, strict=True)
                ):
                    await insert_chunk(
                        session,
                        source=source,
                        doc_id=doc_id,
                        url=None,
                        chunk_index=index,
                        content=chunk,
                        embedding_local=embedding.local,
                        embedding_openai=embedding.openai,
                    )
            total_chunks += len(chunks)
            log_event(
                logger,
                logging.INFO,
                "Document ingested",
                file=doc_id,
                chunks=len(chunks),
            )
        log_event(
            logger,
            logging.INFO,
            "Ingestion finished",
            files=len(files),
            chunks=total_chunks,
        )
    finally:
        await embedder.aclose()
        await dispose_engine()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG store")
    parser.add_argument(
        "--path",
        required=True,
        help="File or directory to ingest",
    )
    parser.add_argument(
        "--source",
        default="manual",
        help="Source label stored with each chunk (for example legal or grants)",
    )
    return parser.parse_args()


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    args = _parse_args()
    asyncio.run(ingest_path(Path(args.path), args.source))


if __name__ == "__main__":
    main()
