from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, TypeVar

from app.schemas.models import ChatMessage, Episode, Fact, Task, VectorMemorySearchHit, Workflow
from app.services.vector_memory import vector_memory

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "i", "if", "in", "into", "is", "it", "me", "my", "of", "on", "or", "our",
    "that", "the", "their", "them", "then", "this", "to", "up", "use", "we",
    "with", "you", "your",
}

T = TypeVar("T")


@dataclass
class ContextBundle:
    focus: str
    tasks: list[Task]
    facts: list[Fact]
    episodes: list[Episode]
    workflows: list[Workflow]
    semantic_memory: list[VectorMemorySearchHit]
    messages: list[ChatMessage]


def _terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9_]{3,}", text.lower())
        if token not in _STOPWORDS
    }


def _score_text(query_terms: set[str], text: str, recency_rank: int = 0) -> float:
    if not text:
        return float(recency_rank)
    text_terms = _terms(text)
    overlap = len(query_terms & text_terms)
    coverage = overlap / max(len(query_terms), 1)
    return overlap * 3.0 + coverage + recency_rank


def _focus_text(user_input: str | None, messages: list[ChatMessage], fallback: str) -> str:
    if user_input:
        return user_input
    for message in reversed(messages):
        if message.role == "user" and message.content:
            return message.content
    return fallback


def _pick_messages(messages: list[ChatMessage], limit: int = 10) -> list[ChatMessage]:
    if len(messages) <= limit:
        return list(messages)

    omitted = messages[:-limit]
    summary_lines = [f"{len(omitted)} earlier messages omitted to reduce context load."]

    seen_roles: set[str] = set()
    for message in reversed(omitted):
        if message.role in seen_roles or not message.content:
            continue
        excerpt = " ".join(message.content.split())[:220]
        summary_lines.append(f"{message.role}: {excerpt}")
        seen_roles.add(message.role)
        if len(seen_roles) >= 2:
            break

    summary = ChatMessage(
        role="user",
        content="Conversation summary:\n" + "\n".join(f"- {line}" for line in summary_lines),
    )
    return [summary, *messages[-limit:]]


def _pick_ranked(items: Iterable[T], scorer: Callable[[T], float], limit: int) -> list[T]:
    scored = [(scorer(item), item) for item in items]
    scored.sort(key=lambda entry: entry[0], reverse=True)
    return [item for score, item in scored[:limit] if score > 0]


async def build_context_bundle(
    *,
    user_input: str | None,
    fallback_focus: str,
    tasks: list[Task],
    facts: list[Fact],
    episodes: list[Episode],
    workflows: list[Workflow],
    messages: list[ChatMessage],
) -> ContextBundle:
    focus = _focus_text(user_input, messages, fallback_focus)
    query_terms = _terms(focus)
    semantic_hits = await vector_memory.search(focus, limit=14)

    task_ids = [hit.source_id for hit in semantic_hits if hit.source_type == "task"]
    fact_ids = [hit.source_id for hit in semantic_hits if hit.source_type == "fact"]
    episode_ids = [hit.source_id for hit in semantic_hits if hit.source_type == "episode"]
    workflow_ids = [hit.source_id for hit in semantic_hits if hit.source_type == "workflow"]

    active_tasks = [task for task in tasks if task.status.lower() not in {"done", "completed", "cancelled", "canceled"}]

    relevant_tasks = _pick_ranked(
        active_tasks,
        lambda task: _score_text(
            query_terms,
            " ".join(
                filter(
                    None,
                    [
                        task.title,
                        task.notes or "",
                        task.status,
                        task.priority,
                        task.outcome or "",
                        " ".join(task.blocked_by),
                    ],
                )
            ),
            recency_rank=2 if task.progress < 100 else 0,
        ) + (3 if task.blocked_by else 0) + (2 if task.due_at else 0) + (5 if task.id in task_ids else 0),
        8,
    )
    if not relevant_tasks:
        relevant_tasks = active_tasks[:6]

    relevant_facts = _pick_ranked(
        facts,
        lambda fact: _score_text(query_terms, fact.text) + (5 if fact.id in fact_ids else 0),
        10,
    )
    if not relevant_facts:
        relevant_facts = facts[:8]

    relevant_episodes = _pick_ranked(
        episodes,
        lambda episode: _score_text(query_terms, episode.summary) + (5 if episode.id in episode_ids else 0),
        6,
    )
    if not relevant_episodes:
        relevant_episodes = episodes[:5]

    relevant_workflows = _pick_ranked(
        workflows,
        lambda workflow: _score_text(
            query_terms,
            " ".join(
                filter(
                    None,
                    [
                        workflow.name,
                        workflow.description or "",
                        *(step.tool_name for step in workflow.steps),
                    ],
                )
            ),
        ) + (2 if workflow.enabled else 0) + (5 if workflow.id in workflow_ids else 0),
        6,
    )
    if not relevant_workflows:
        relevant_workflows = workflows[:5]

    return ContextBundle(
        focus=focus,
        tasks=relevant_tasks,
        facts=relevant_facts,
        episodes=relevant_episodes,
        workflows=relevant_workflows,
        semantic_memory=semantic_hits,
        messages=_pick_messages(messages),
    )
