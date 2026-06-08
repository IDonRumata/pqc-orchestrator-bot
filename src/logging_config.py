"""Structured logging setup.

Logs are emitted as single line JSON so they can be parsed by the Docker log
driver or any log shipper. No long dashes are used anywhere in the output.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    """Render log records as compact JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Attach any structured extra fields placed under record.extra_fields.
        extra_fields = getattr(record, "extra_fields", None)
        if isinstance(extra_fields, dict):
            payload.update(extra_fields)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once with the JSON formatter."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    # Remove pre-existing handlers to avoid duplicated lines on reload.
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    # Reduce noise from third party libraries.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("aiogram.event").setLevel(logging.WARNING)


def log_event(logger: logging.Logger, level: int, message: str, **fields: object) -> None:
    """Emit a log record with structured extra fields."""
    logger.log(level, message, extra={"extra_fields": fields})
