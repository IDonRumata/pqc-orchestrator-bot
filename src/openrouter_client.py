"""Async OpenRouter client with Pydantic v2 validation and tenacity retries."""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from pydantic import BaseModel, Field
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import Settings, get_settings
from .logging_config import log_event

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    """A single chat message."""

    role: str
    content: str


class Usage(BaseModel):
    """Token usage block returned by OpenRouter."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class _ResponseMessage(BaseModel):
    role: str = "assistant"
    content: str | None = None


class _Choice(BaseModel):
    index: int = 0
    message: _ResponseMessage = Field(default_factory=_ResponseMessage)
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    """Validated subset of an OpenRouter chat completion response."""

    id: str | None = None
    model: str | None = None
    choices: list[_Choice] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)

    @property
    def content(self) -> str:
        if not self.choices:
            return ""
        return self.choices[0].message.content or ""


class LLMResult(BaseModel):
    """Normalized result handed back to the orchestrator."""

    content: str
    model: str
    tokens: int


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter returns a non recoverable error."""


# Exceptions that justify a retry.
_RETRYABLE = (
    httpx.TimeoutException,
    httpx.TransportError,
    httpx.NetworkError,
)


class OpenRouterClient:
    """Thin async wrapper around the OpenRouter chat completions endpoint."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        timeout = httpx.Timeout(
            self._settings.http_timeout_seconds,
            connect=self._settings.http_connect_timeout_seconds,
        )
        self._client = httpx.AsyncClient(
            base_url=self._settings.openrouter_base_url,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self._settings.openrouter_api_key}",
                "HTTP-Referer": self._settings.openrouter_referer,
                "X-Title": self._settings.openrouter_app_title,
                "Content-Type": "application/json",
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "OpenRouterClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def chat_completion(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        temperature: float = 0.3,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResult:
        """Perform a chat completion with retries and validation."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        retryer = AsyncRetrying(
            stop=stop_after_attempt(self._settings.retry_attempts),
            wait=wait_exponential(
                multiplier=self._settings.retry_min_wait_seconds,
                max=self._settings.retry_max_wait_seconds,
            ),
            retry=retry_if_exception_type(_RETRYABLE),
            reraise=True,
        )

        async for attempt in retryer:
            with attempt:
                response = await self._client.post("/chat/completions", json=payload)
                self._raise_for_status(response, model)
                parsed = ChatCompletionResponse.model_validate(response.json())
                log_event(
                    logger,
                    logging.INFO,
                    "LLM call completed",
                    model=model,
                    tokens=parsed.usage.total_tokens,
                    json_mode=json_mode,
                )
                return LLMResult(
                    content=parsed.content,
                    model=parsed.model or model,
                    tokens=parsed.usage.total_tokens,
                )
        # AsyncRetrying with reraise=True never reaches this point.
        raise OpenRouterError("Retry loop exhausted without a result")

    async def chat_json(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> tuple[dict[str, Any], int]:
        """Perform a chat completion expecting a JSON object answer."""
        result = await self.chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        data = self._parse_json(result.content)
        return data, result.tokens

    def _raise_for_status(self, response: httpx.Response, model: str) -> None:
        if response.status_code < 400:
            return
        # 5xx and 429 are transient, surface them as retryable timeouts.
        if response.status_code in (429,) or response.status_code >= 500:
            raise httpx.TransportError(
                f"OpenRouter transient error {response.status_code} for {model}"
            )
        # 4xx client errors are not retried.
        raise OpenRouterError(
            f"OpenRouter error {response.status_code} for {model}: {response.text[:500]}"
        )

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        """Parse a JSON object, tolerating code fences and surrounding noise."""
        text = raw.strip()
        if text.startswith("```"):
            # Strip a leading fence such as ```json and the trailing fence.
            text = text.split("```", 2)[1] if text.count("```") >= 2 else text
            if text.lstrip().lower().startswith("json"):
                text = text.lstrip()[4:]
        text = text.strip().strip("`").strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise OpenRouterError(f"Could not parse JSON from model output: {exc}")
        if not isinstance(data, dict):
            raise OpenRouterError("Model returned JSON that is not an object")
        return data
