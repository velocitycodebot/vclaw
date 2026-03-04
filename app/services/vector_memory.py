from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from app.schemas.models import Preferences, VectorMemorySearchHit, VectorMemoryStatus

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]{2,}")


@dataclass
class _MemoryDocument:
    source_type: str
    source_id: str
    text: str
    metadata: dict[str, Any]


def _token_overlap_score(query: str, text: str) -> float:
    query_terms = {term for term in _TOKEN_RE.findall(query.lower()) if len(term) > 2}
    if not query_terms:
        return 0.0
    text_terms = {term for term in _TOKEN_RE.findall(text.lower()) if len(term) > 2}
    if not text_terms:
        return 0.0
    return len(query_terms & text_terms) / len(query_terms)


def _normalize_dense(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(lv * rv for lv, rv in zip(left, right))


class VectorMemoryService:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._dirty = True
        self._provider = "uninitialized"
        self._dimensions = 0
        self._degraded = False
        self._last_rebuilt_at: Optional[datetime] = None
        self._documents: list[_MemoryDocument] = []
        self._matrix: list[list[float]] = []
        self._vectorizer: Any = None
        self._reducer: Any = None

    async def initialize(self):
        self.mark_dirty()
        await self.ensure_fresh()

    def mark_dirty(self):
        self._dirty = True

    async def status(self) -> VectorMemoryStatus:
        await self.ensure_fresh()
        return VectorMemoryStatus(
            provider=self._provider,
            dimensions=self._dimensions,
            documents=len(self._documents),
            degraded=self._degraded,
            dirty=self._dirty,
            last_rebuilt_at=self._last_rebuilt_at,
        )

    async def search(
        self,
        query: str,
        *,
        limit: int = 8,
        source_types: Optional[list[str]] = None,
    ) -> list[VectorMemorySearchHit]:
        if not query.strip():
            return []

        await self.ensure_fresh()
        if not self._documents or not self._matrix:
            return []

        query_vector = self._embed_query(query)
        allowed = set(source_types or [])

        hits: list[VectorMemorySearchHit] = []
        for doc, vector in zip(self._documents, self._matrix):
            if allowed and doc.source_type not in allowed:
                continue
            semantic = _cosine_similarity(query_vector, vector)
            lexical = _token_overlap_score(query, doc.text)
            score = semantic * 0.85 + lexical * 0.15
            if score <= 0:
                continue
            hits.append(
                VectorMemorySearchHit(
                    source_type=doc.source_type,
                    source_id=doc.source_id,
                    text=doc.text,
                    score=round(score, 6),
                    metadata=dict(doc.metadata),
                )
            )

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[: max(1, limit)]

    async def ensure_fresh(self):
        if not self._dirty and (self._documents or self._provider == "empty"):
            return

        async with self._lock:
            if not self._dirty and (self._documents or self._provider == "empty"):
                return
            docs = await self._load_documents()
            await asyncio.to_thread(self._rebuild_sync, docs)

    def _rebuild_sync(self, docs: list[_MemoryDocument]):
        self._documents = docs
        self._matrix = []
        self._vectorizer = None
        self._reducer = None
        self._provider = "empty"
        self._dimensions = 0
        self._degraded = False
        self._last_rebuilt_at = datetime.now(timezone.utc)

        if not docs:
            self._dirty = False
            return

        texts = [doc.text for doc in docs]
        try:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=4096,
            )
            matrix = vectorizer.fit_transform(texts)
            if matrix.shape[1] == 0:
                raise ValueError("empty vocabulary")

            reducer = None
            dense_vectors: list[list[float]]
            max_components = min(matrix.shape[0] - 1, matrix.shape[1] - 1, 128)
            if max_components >= 2:
                reducer = TruncatedSVD(n_components=max_components, random_state=42)
                dense = reducer.fit_transform(matrix)
                dense_vectors = [_normalize_dense(row.tolist()) for row in dense]
                self._provider = "tfidf-lsa"
            else:
                dense = matrix.toarray()
                dense_vectors = [_normalize_dense(row.tolist()) for row in dense]
                self._provider = "tfidf"

            self._vectorizer = vectorizer
            self._reducer = reducer
            self._matrix = dense_vectors
            self._dimensions = len(dense_vectors[0]) if dense_vectors else 0
            self._degraded = False
        except Exception:
            self._provider = "hashed-fallback"
            self._degraded = True
            self._matrix = [self._hashed_vector(text) for text in texts]
            self._dimensions = len(self._matrix[0]) if self._matrix else 0

        self._dirty = False

    def _embed_query(self, query: str) -> list[float]:
        if self._provider == "hashed-fallback" or self._vectorizer is None:
            return self._hashed_vector(query)

        matrix = self._vectorizer.transform([query])
        if self._reducer is not None:
            dense = self._reducer.transform(matrix)[0].tolist()
        else:
            dense = matrix.toarray()[0].tolist()
        return _normalize_dense(dense)

    def _hashed_vector(self, text: str, dimensions: int = 256) -> list[float]:
        vector = [0.0] * dimensions
        for token in _TOKEN_RE.findall(text.lower()):
            slot = hash(token) % dimensions
            vector[slot] += 1.0
        return _normalize_dense(vector)

    async def _load_documents(self) -> list[_MemoryDocument]:
        import app.storage.database as db
        from app.services.memory_workspace import memory_workspace
        from app.services.session_manager import session_manager

        prefs = await db.get_preferences()
        facts = await db.get_all_facts()
        episodes = await db.get_episodes(100)
        tasks = await db.get_all_tasks()
        workflows = await db.get_all_workflows()
        memory_docs = await memory_workspace.memory_documents()
        sessions = await session_manager.list_sessions()

        docs: list[_MemoryDocument] = []
        docs.extend(self._preference_documents(prefs))
        docs.extend(
            _MemoryDocument(
                source_type="fact",
                source_id=fact.id,
                text=fact.text,
                metadata={"category": fact.category or ""},
            )
            for fact in facts
        )
        docs.extend(
            _MemoryDocument(
                source_type="episode",
                source_id=episode.id,
                text=episode.summary,
                metadata={"created_at": episode.created_at.isoformat()},
            )
            for episode in episodes
        )
        docs.extend(
            _MemoryDocument(
                source_type="task",
                source_id=task.id,
                text=self._task_text(task),
                metadata={
                    "status": task.status,
                    "priority": task.priority,
                    "progress": task.progress,
                },
            )
            for task in tasks
        )
        docs.extend(
            _MemoryDocument(
                source_type="workflow",
                source_id=workflow.id,
                text=self._workflow_text(workflow),
                metadata={"enabled": workflow.enabled},
            )
            for workflow in workflows
        )
        docs.extend(
            _MemoryDocument(
                source_type=item["source_type"],
                source_id=item["source_id"],
                text=item["text"],
                metadata=item["metadata"],
            )
            for item in memory_docs
        )
        for session in sessions:
            messages = await session_manager.load_messages(session.id)
            for index, message in enumerate(messages[-40:]):
                if not message.content.strip():
                    continue
                docs.append(
                    _MemoryDocument(
                        source_type="session",
                        source_id=f"{session.id}:{index}",
                        text=f"Session {session.session_key} {message.role}: {message.content[:1500]}",
                        metadata={"session_id": session.id, "session_key": session.session_key, "role": message.role},
                    )
                )
        return docs

    def _preference_documents(self, prefs: Preferences) -> list[_MemoryDocument]:
        docs = [
            _MemoryDocument(
                source_type="preference",
                source_id="name",
                text=f"User name preference: {prefs.name or 'unknown'}",
                metadata={"field": "name"},
            ),
            _MemoryDocument(
                source_type="preference",
                source_id="tone",
                text=f"User preferred tone: {prefs.tone}",
                metadata={"field": "tone"},
            ),
        ]
        for index, topic in enumerate(prefs.topics):
            docs.append(
                _MemoryDocument(
                    source_type="preference",
                    source_id=f"topic:{index}",
                    text=f"User topic or interest: {topic}",
                    metadata={"field": "topics"},
                )
            )
        for key, value in prefs.custom.items():
            docs.append(
                _MemoryDocument(
                    source_type="preference",
                    source_id=f"custom:{key}",
                    text=f"User preference {key}: {value}",
                    metadata={"field": key},
                )
            )
        return docs

    @staticmethod
    def _task_text(task: Any) -> str:
        parts = [
            task.title,
            task.notes or "",
            f"status {task.status}",
            f"priority {task.priority}",
            f"progress {task.progress}",
            task.outcome or "",
        ]
        if task.assignee:
            parts.append(f"assignee {task.assignee}")
        if task.blocked_by:
            parts.append(f"blocked by {' '.join(task.blocked_by)}")
        if task.depends_on:
            parts.append(f"depends on {' '.join(task.depends_on)}")
        return " ".join(part for part in parts if part)

    @staticmethod
    def _workflow_text(workflow: Any) -> str:
        parts = [workflow.name, workflow.description or ""]
        for step in workflow.steps:
            parts.append(step.tool_name)
        if workflow.schedule and workflow.schedule.label:
            parts.append(workflow.schedule.label)
        return " ".join(part for part in parts if part)


vector_memory = VectorMemoryService()
