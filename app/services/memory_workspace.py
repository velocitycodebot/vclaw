from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.schemas.models import MemoryFileContent, MemoryFileInfo, MemoryWorkspaceSearchHit
from app.storage.database import DATA_DIR

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]{3,}")


class MemoryWorkspaceService:
    def __init__(self):
        self.root = DATA_DIR / "memory"
        self.long_term_path = self.root / "MEMORY.md"
        self._lock = asyncio.Lock()

    async def initialize(self):
        async with self._lock:
            await asyncio.to_thread(self.root.mkdir, parents=True, exist_ok=True)
            if not self.long_term_path.exists():
                await asyncio.to_thread(
                    self.long_term_path.write_text,
                    "# MEMORY\n\nDurable notes for the assistant.\n\n## Durable Notes\n",
                    "utf-8",
                )

    def daily_path(self, day: datetime | None = None) -> Path:
        stamp = (day or datetime.now().astimezone()).strftime("%Y-%m-%d")
        return self.root / f"{stamp}.md"

    async def append_long_term_notes(self, notes: list[str]):
        cleaned = [note.strip() for note in notes if note and note.strip()]
        if not cleaned:
            return
        await self.initialize()
        async with self._lock:
            existing = await asyncio.to_thread(self.long_term_path.read_text, "utf-8")
            if "## Durable Notes" not in existing:
                existing = existing.rstrip() + "\n\n## Durable Notes\n"
            stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
            lines = "\n".join(f"- [{stamp}] {note}" for note in cleaned)
            payload = existing.rstrip() + "\n" + lines + "\n"
            await asyncio.to_thread(self.long_term_path.write_text, payload, "utf-8")
        self._mark_vector_memory_dirty()

    async def append_daily_notes(self, notes: list[str], date_str: str | None = None):
        cleaned = [note.strip() for note in notes if note and note.strip()]
        if not cleaned:
            return
        await self.initialize()
        target = self.daily_path(self._parse_day(date_str))
        async with self._lock:
            if target.exists():
                existing = await asyncio.to_thread(target.read_text, "utf-8")
            else:
                heading = target.stem
                existing = f"# {heading}\n\n## Session Notes\n"
            if "## Session Notes" not in existing:
                existing = existing.rstrip() + "\n\n## Session Notes\n"
            stamp = datetime.now().astimezone().strftime("%H:%M")
            lines = "\n".join(f"- [{stamp}] {note}" for note in cleaned)
            payload = existing.rstrip() + "\n" + lines + "\n"
            await asyncio.to_thread(target.write_text, payload, "utf-8")
        self._mark_vector_memory_dirty()

    async def list_files(self) -> list[MemoryFileInfo]:
        await self.initialize()
        files: list[MemoryFileInfo] = []
        for path in sorted(self.root.glob("*.md"), reverse=True):
            stat = await asyncio.to_thread(path.stat)
            files.append(
                MemoryFileInfo(
                    name=path.name,
                    kind="long_term" if path.name == "MEMORY.md" else "daily",
                    path=str(path),
                    size=stat.st_size,
                    updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                )
            )
        return files

    async def get_file(self, name: str) -> MemoryFileContent:
        await self.initialize()
        path = self._resolve_name(name)
        if not path.exists():
            raise FileNotFoundError(name)
        stat = await asyncio.to_thread(path.stat)
        content = await asyncio.to_thread(path.read_text, "utf-8")
        return MemoryFileContent(
            name=path.name,
            kind="long_term" if path.name == "MEMORY.md" else "daily",
            path=str(path),
            content=content,
            updated_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
        )

    async def write_file(self, name: str, content: str) -> MemoryFileContent:
        await self.initialize()
        path = self._resolve_name(name)
        if path.name != "MEMORY.md" and not path.exists():
            raise FileNotFoundError(name)
        payload = content if content.endswith("\n") else content + "\n"
        async with self._lock:
            await asyncio.to_thread(path.write_text, payload, "utf-8")
        self._mark_vector_memory_dirty()
        return await self.get_file(path.name)

    async def search(
        self,
        query: str,
        *,
        limit: int = 8,
        include_daily: bool = True,
        include_long_term: bool = True,
    ) -> list[MemoryWorkspaceSearchHit]:
        query = query.strip()
        if not query:
            return []
        await self.initialize()
        query_terms = self._terms(query)
        if not query_terms:
            return []

        hits: list[MemoryWorkspaceSearchHit] = []
        for path in sorted(self.root.glob("*.md"), reverse=True):
            kind = "long_term" if path.name == "MEMORY.md" else "daily"
            if kind == "daily" and not include_daily:
                continue
            if kind == "long_term" and not include_long_term:
                continue
            content = await asyncio.to_thread(path.read_text, "utf-8")
            updated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            for chunk in self._iter_chunks(path, content):
                chunk_terms = self._terms(chunk["text"])
                if not chunk_terms:
                    continue
                overlap = len(query_terms & chunk_terms)
                if overlap == 0:
                    continue
                coverage = overlap / max(1, len(query_terms))
                recency = 0.08 if kind == "daily" else 0.0
                score = round(overlap * 1.25 + coverage + recency, 6)
                hits.append(
                    MemoryWorkspaceSearchHit(
                        file_name=path.name,
                        path=str(path),
                        kind=kind,
                        title=chunk["title"],
                        snippet=chunk["text"][:500],
                        score=score,
                        line_start=chunk["line_start"],
                        line_end=chunk["line_end"],
                        updated_at=updated_at,
                    )
                )

        hits.sort(key=lambda hit: (hit.score, hit.updated_at or datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
        return hits[: max(1, limit)]

    async def memory_documents(self) -> list[dict[str, Any]]:
        await self.initialize()
        docs: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("*.md"), reverse=True):
            kind = "memory_long_term" if path.name == "MEMORY.md" else "memory_daily"
            content = await asyncio.to_thread(path.read_text, "utf-8")
            for index, chunk in enumerate(self._iter_chunks(path, content)):
                docs.append({
                    "source_type": kind,
                    "source_id": f"{path.name}:{index}",
                    "text": chunk["text"][:1500],
                    "metadata": {
                        "file_name": path.name,
                        "path": str(path),
                        "title": chunk["title"],
                        "line_start": chunk["line_start"],
                        "line_end": chunk["line_end"],
                    },
                })
        return docs

    def _resolve_name(self, name: str) -> Path:
        candidate = (self.root / name).resolve()
        if self.root.resolve() not in candidate.parents and candidate != self.root.resolve():
            raise FileNotFoundError(name)
        return candidate

    def _parse_day(self, date_str: str | None) -> datetime | None:
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=datetime.now().astimezone().tzinfo)
        except ValueError:
            return None

    def _iter_chunks(self, path: Path, content: str) -> list[dict[str, Any]]:
        lines = content.splitlines()
        chunks: list[dict[str, Any]] = []
        title = path.stem
        start = 1
        buffer: list[str] = []

        def flush(end_line: int):
            text = "\n".join(buffer).strip()
            if not text:
                return
            chunks.append({
                "title": title,
                "text": text,
                "line_start": start,
                "line_end": end_line,
            })

        for idx, line in enumerate(lines, start=1):
            if line.startswith("#"):
                flush(idx - 1)
                title = line.lstrip("#").strip() or path.stem
                buffer = [line]
                start = idx
                continue
            if not line.strip() and buffer and buffer[-1] == "":
                flush(idx - 1)
                buffer = []
                start = idx + 1
                continue
            buffer.append(line)
        flush(len(lines))
        return chunks

    def _terms(self, text: str) -> set[str]:
        return {token.lower() for token in _TOKEN_RE.findall(text)}

    def _mark_vector_memory_dirty(self):
        try:
            from app.services.vector_memory import vector_memory

            vector_memory.mark_dirty()
        except Exception:
            pass


memory_workspace = MemoryWorkspaceService()
