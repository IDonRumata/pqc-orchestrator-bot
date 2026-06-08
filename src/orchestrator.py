"""The multi-agent orchestrator.

Flow:
1. Router selects 1 to 3 relevant agents and an optional RAG query.
2. For legal and grants agents the relevant knowledge chunks are retrieved.
3. Selected specialist agents run in parallel.
4. The Chief Critic merges and validates the answers when more than one agent
   was used. With a single agent its answer is returned directly.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from pydantic import BaseModel, Field, field_validator

from .config import Settings, get_settings
from .database import record_agent_run, search_chunks, session_scope
from .embeddings import EmbeddingRouter
from .logging_config import log_event
from .openrouter_client import ChatMessage, OpenRouterClient
from .prompts import (
    CFO_SYSTEM_PROMPT,
    CRITIC_SYSTEM_PROMPT,
    GRANTS_SYSTEM_PROMPT,
    LEGAL_SYSTEM_PROMPT,
    PQC_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    build_critic_user_prompt,
    build_specialist_user_prompt,
)

logger = logging.getLogger(__name__)

# Agents that require RAG context retrieval.
_RAG_AGENTS = {"legal", "grants"}
_VALID_AGENTS = {"cfo", "pqc", "legal", "grants"}


@dataclass(slots=True)
class AgentDefinition:
    """Static definition of a specialist agent."""

    key: str
    title: str
    system_prompt: str
    model_attr: str
    needs_rag: bool


def _build_agent_registry(settings: Settings) -> dict[str, AgentDefinition]:
    return {
        "cfo": AgentDefinition(
            key="cfo",
            title="Финансовый директор",
            system_prompt=CFO_SYSTEM_PROMPT,
            model_attr="cfo_model",
            needs_rag=False,
        ),
        "pqc": AgentDefinition(
            key="pqc",
            title="Ученый-криптограф",
            system_prompt=PQC_SYSTEM_PROMPT,
            model_attr="pqc_model",
            needs_rag=False,
        ),
        "legal": AgentDefinition(
            key="legal",
            title="Юрист по комплаенсу ЕС",
            system_prompt=LEGAL_SYSTEM_PROMPT,
            model_attr="legal_model",
            needs_rag=True,
        ),
        "grants": AgentDefinition(
            key="grants",
            title="Эксперт по грантам ЕС и Польши",
            system_prompt=GRANTS_SYSTEM_PROMPT,
            model_attr="grants_model",
            needs_rag=True,
        ),
    }


class RouterDecision(BaseModel):
    """Validated output of the routing step."""

    agents: list[str] = Field(default_factory=list)
    reasoning: str = ""
    rag_query: str = ""

    @field_validator("agents")
    @classmethod
    def _clean_agents(cls, value: list[str]) -> list[str]:
        seen: list[str] = []
        for raw in value:
            key = str(raw).strip().lower()
            if key in _VALID_AGENTS and key not in seen:
                seen.append(key)
        return seen


@dataclass(slots=True)
class AgentOutput:
    """Result produced by one specialist agent."""

    key: str
    title: str
    answer: str
    tokens: int


@dataclass(slots=True)
class OrchestrationResult:
    """Outcome of a full orchestration run."""

    answer: str
    agents_used: list[str] = field(default_factory=list)
    reasoning: str = ""
    rag_chunks_used: int = 0
    tokens_total: int = 0
    duration_ms: int = 0
    critic_used: bool = False


class Orchestrator:
    """Coordinates routing, retrieval, specialist agents and the critic."""

    def __init__(
        self,
        client: OpenRouterClient,
        embedder: EmbeddingRouter,
        settings: Settings | None = None,
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._settings = settings or get_settings()
        self._registry = _build_agent_registry(self._settings)

    async def handle(self, user_id: int, query: str) -> OrchestrationResult:
        """Run the full pipeline for a user query and persist an audit record."""
        started = time.monotonic()
        tokens_total = 0

        decision, router_tokens = await self._route(query)
        tokens_total += router_tokens

        if not decision.agents:
            # Safe default, ask the strongest single specialist when routing is empty.
            decision.agents = ["pqc"]
            decision.reasoning = (
                decision.reasoning
                or "Роутер не выделил агентов, использован агент по умолчанию."
            )

        log_event(
            logger,
            logging.INFO,
            "Routing decision",
            user_id=user_id,
            agents=decision.agents,
            reasoning=decision.reasoning,
        )

        # Retrieve RAG context once if any RAG agent was selected.
        context_block, chunks_used = await self._retrieve_context(decision)

        # Run all selected agents in parallel.
        outputs = await self._run_agents(decision.agents, query, context_block)
        for output in outputs:
            tokens_total += output.tokens

        # Decide whether the critic is needed.
        if len(outputs) > 1:
            final_answer, critic_tokens = await self._run_critic(query, outputs)
            tokens_total += critic_tokens
            critic_used = True
        else:
            final_answer = self._format_single(outputs[0])
            critic_used = False

        duration_ms = int((time.monotonic() - started) * 1000)

        await self._persist_run(
            user_id=user_id,
            query=query,
            decision=decision,
            chunks_used=chunks_used,
            tokens_total=tokens_total,
            duration_ms=duration_ms,
        )

        log_event(
            logger,
            logging.INFO,
            "Orchestration complete",
            user_id=user_id,
            agents=decision.agents,
            critic_used=critic_used,
            rag_chunks_used=chunks_used,
            tokens_total=tokens_total,
            duration_ms=duration_ms,
        )

        return OrchestrationResult(
            answer=final_answer,
            agents_used=decision.agents,
            reasoning=decision.reasoning,
            rag_chunks_used=chunks_used,
            tokens_total=tokens_total,
            duration_ms=duration_ms,
            critic_used=critic_used,
        )

    async def _route(self, query: str) -> tuple[RouterDecision, int]:
        """Ask the router model which agents to engage."""
        messages = [
            ChatMessage(role="system", content=ROUTER_SYSTEM_PROMPT),
            ChatMessage(role="user", content=query),
        ]
        try:
            data, tokens = await self._client.chat_json(
                model=self._settings.router_model,
                messages=messages,
                temperature=0.0,
                max_tokens=600,
            )
            decision = RouterDecision.model_validate(data)
        except Exception as exc:  # noqa: BLE001 - routing must never crash the bot
            log_event(
                logger,
                logging.WARNING,
                "Router failed, using default agent",
                error=str(exc),
            )
            return RouterDecision(agents=["pqc"], reasoning="Сбой роутера, агент по умолчанию."), 0

        # Enforce the configured maximum number of agents.
        decision.agents = decision.agents[: self._settings.max_agents_per_request]
        return decision, tokens

    async def _retrieve_context(
        self, decision: RouterDecision
    ) -> tuple[str | None, int]:
        """Run a vector search if any RAG agent is selected."""
        if not (_RAG_AGENTS & set(decision.agents)):
            return None, 0
        search_text = decision.rag_query.strip() or decision.reasoning.strip()
        if not search_text:
            return None, 0

        query_embedding = await self._embedder.embed_query(search_text)
        async with session_scope() as session:
            chunks = await search_chunks(
                session,
                query_vector=query_embedding.vector,
                provider=query_embedding.provider,
                limit=self._settings.rag_top_k,
            )

        if not chunks:
            log_event(
                logger,
                logging.INFO,
                "RAG search returned no chunks",
                provider=query_embedding.provider,
                query=search_text,
            )
            return None, 0

        formatted: list[str] = []
        for chunk in chunks:
            origin = chunk.url or chunk.doc_id
            formatted.append(f"[Источник: {chunk.source} | {origin}]\n{chunk.content}")
        block = "\n\n---\n\n".join(formatted)

        log_event(
            logger,
            logging.INFO,
            "RAG context retrieved",
            provider=query_embedding.provider,
            chunks=len(chunks),
            query=search_text,
        )
        return block, len(chunks)

    async def _run_agents(
        self, agent_keys: list[str], query: str, context_block: str | None
    ) -> list[AgentOutput]:
        """Run all selected specialist agents concurrently."""
        tasks = [
            self._run_single_agent(key, query, context_block) for key in agent_keys
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        outputs: list[AgentOutput] = []
        for key, result in zip(agent_keys, results, strict=True):
            definition = self._registry[key]
            if isinstance(result, Exception):
                log_event(
                    logger,
                    logging.ERROR,
                    "Agent failed",
                    agent=key,
                    error=str(result),
                )
                outputs.append(
                    AgentOutput(
                        key=key,
                        title=definition.title,
                        answer=(
                            f"Агент {definition.title} временно недоступен. "
                            "Произошла ошибка при обращении к модели."
                        ),
                        tokens=0,
                    )
                )
            else:
                outputs.append(result)
        return outputs

    async def _run_single_agent(
        self, key: str, query: str, context_block: str | None
    ) -> AgentOutput:
        """Run one specialist agent inside its isolated system prompt."""
        definition = self._registry[key]
        model = getattr(self._settings, definition.model_attr)
        # Context is only handed to agents that asked for it.
        agent_context = context_block if definition.needs_rag else None
        user_prompt = build_specialist_user_prompt(query, agent_context)
        messages = [
            ChatMessage(role="system", content=definition.system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]
        result = await self._client.chat_completion(
            model=model,
            messages=messages,
            temperature=0.3,
        )
        return AgentOutput(
            key=key,
            title=definition.title,
            answer=result.content.strip(),
            tokens=result.tokens,
        )

    async def _run_critic(
        self, query: str, outputs: list[AgentOutput]
    ) -> tuple[str, int]:
        """Run the Chief Critic to merge and validate agent answers."""
        agent_answers = {output.title: output.answer for output in outputs}
        user_prompt = build_critic_user_prompt(query, agent_answers)
        messages = [
            ChatMessage(role="system", content=CRITIC_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]
        result = await self._client.chat_completion(
            model=self._settings.critic_model,
            messages=messages,
            temperature=0.2,
        )
        return result.content.strip(), result.tokens

    @staticmethod
    def _format_single(output: AgentOutput) -> str:
        """Format a single agent answer when the critic is skipped."""
        return f"Агент: {output.title}\n\n{output.answer}"

    async def _persist_run(
        self,
        *,
        user_id: int,
        query: str,
        decision: RouterDecision,
        chunks_used: int,
        tokens_total: int,
        duration_ms: int,
    ) -> None:
        """Store the audit record, failures here never break the user response."""
        try:
            async with session_scope() as session:
                await record_agent_run(
                    session,
                    user_id=user_id,
                    query=query,
                    agents_selected={
                        "agents": decision.agents,
                        "reasoning": decision.reasoning,
                    },
                    rag_chunks_used=chunks_used,
                    tokens_total=tokens_total,
                    duration_ms=duration_ms,
                )
        except Exception as exc:  # noqa: BLE001
            log_event(
                logger,
                logging.WARNING,
                "Failed to persist agent run",
                error=str(exc),
            )
