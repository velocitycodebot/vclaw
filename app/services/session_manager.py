from __future__ import annotations

import asyncio
import json
import os
from copy import deepcopy
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import app.storage.database as db
from app.schemas.models import ChatMessage, ChatRequest, SessionRecord, SessionResetRequest, SessionScope, new_id
from app.storage.database import DATA_DIR


class SessionManager:
    def __init__(self):
        self.root = DATA_DIR / "sessions"
        self.index_path = self.root / "sessions.json"
        self._lock = asyncio.Lock()
        self._migrated_legacy = False

    async def initialize(self):
        async with self._lock:
            await asyncio.to_thread(self.root.mkdir, parents=True, exist_ok=True)
            if not self.index_path.exists():
                await asyncio.to_thread(self._write_index_sync, self._default_index())
        if not self._migrated_legacy:
            await self._migrate_legacy_conversations()
            self._migrated_legacy = True

    async def resolve_chat_session(self, req: ChatRequest) -> SessionRecord:
        await self.initialize()
        async with self._lock:
            index = await asyncio.to_thread(self._read_index_sync)
            existing: Optional[SessionRecord] = None

            if req.session_id:
                raw = index["sessions"].get(req.session_id)
                if raw:
                    existing = SessionRecord(**raw)
            if existing is None:
                session_key = req.session_key or self._build_session_key(req)
                active_id = index["active_keys"].get(session_key)
                if active_id:
                    raw = index["sessions"].get(active_id)
                    if raw:
                        existing = SessionRecord(**raw)
                if existing is None and req.reset_session:
                    existing = self._find_latest_by_key(index, session_key)

            if existing and not req.new_session and not req.reset_session and not self._needs_rotation(existing):
                return existing

            previous_id = existing.id if existing else None
            resolved_session_key = req.session_key or (existing.session_key if existing else self._build_session_key(req))
            reason = None
            if req.reset_session:
                reason = "manual_reset"
            elif req.new_session:
                reason = "new_session"
            elif existing and self._needs_rotation(existing):
                reason = "session_expired"

            created = self._create_session(
                req=req,
                session_key=resolved_session_key,
                previous_session_id=previous_id,
                reset_reason=reason,
                existing=existing,
            )
            index["sessions"][created.id] = created.model_dump(mode="json")
            index["active_keys"][created.session_key] = created.id
            await asyncio.to_thread(self._write_index_sync, index)
            return created

    async def list_sessions(self) -> list[SessionRecord]:
        await self.initialize()
        async with self._lock:
            index = await asyncio.to_thread(self._read_index_sync)
        active_ids = set(index["active_keys"].values())
        sessions = [
            SessionRecord(**raw)
            for sid, raw in index["sessions"].items()
            if sid in active_ids
        ]
        return sorted(sessions, key=lambda item: item.updated_at, reverse=True)

    async def get_session(self, session_id: str) -> Optional[SessionRecord]:
        await self.initialize()
        async with self._lock:
            index = await asyncio.to_thread(self._read_index_sync)
        raw = index["sessions"].get(session_id)
        return SessionRecord(**raw) if raw else None

    async def reset_session(self, req: SessionResetRequest) -> SessionRecord:
        dummy = ChatRequest(
            message="",
            session_id=req.session_id,
            session_key=req.session_key,
            reset_session=True,
        )
        session = await self.resolve_chat_session(dummy)
        if req.reason:
            await self._update_record(session.id, reset_reason=req.reason)
            session = await self.get_session(session.id) or session
        return session

    async def load_messages(self, session_id: str) -> list[ChatMessage]:
        session = await self.get_session(session_id)
        if session is None:
            return []
        path = Path(session.transcript_path)
        if not path.exists():
            return []
        raw = await asyncio.to_thread(path.read_text, "utf-8")
        messages: list[ChatMessage] = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            messages.append(ChatMessage(**json.loads(line)))
        return messages

    async def save_messages(self, session_id: str, messages: list[ChatMessage]):
        session = await self.get_session(session_id)
        if session is None:
            raise ValueError(f"Session '{session_id}' not found")
        payload = "\n".join(
            json.dumps(message.model_dump(mode="json"), ensure_ascii=True)
            for message in messages
        )
        if payload:
            payload += "\n"
        path = Path(session.transcript_path)
        await asyncio.to_thread(path.write_text, payload, "utf-8")
        await self._update_record(
            session_id,
            message_count=len(messages),
            updated_at=datetime.now(timezone.utc),
            last_activity_at=datetime.now(timezone.utc),
        )
        self._mark_vector_memory_dirty()

    async def _update_record(self, session_id: str, **updates: Any):
        async with self._lock:
            index = await asyncio.to_thread(self._read_index_sync)
            raw = index["sessions"].get(session_id)
            if not raw:
                return
            updated = dict(raw)
            for key, value in updates.items():
                if isinstance(value, datetime):
                    updated[key] = value.isoformat()
                else:
                    updated[key] = value
            index["sessions"][session_id] = updated
            await asyncio.to_thread(self._write_index_sync, index)

    async def _migrate_legacy_conversations(self):
        conversations = await db.get_all_conversations()
        if not conversations:
            return
        async with self._lock:
            index = await asyncio.to_thread(self._read_index_sync)
            if index["sessions"]:
                return
            for cid, messages in conversations.items():
                req = ChatRequest(message="", conversation_id=cid, session_key=f"agent:main:{cid}")
                session = self._create_session(
                    req=req,
                    session_key=req.session_key or self._build_session_key(req),
                    previous_session_id=None,
                    reset_reason="legacy_import",
                )
                index["sessions"][session.id] = session.model_dump(mode="json")
                index["active_keys"][session.session_key] = session.id
                payload = "\n".join(
                    json.dumps(message.model_dump(mode="json"), ensure_ascii=True)
                    for message in messages
                )
                transcript_path = Path(session.transcript_path)
                if payload:
                    payload += "\n"
                await asyncio.to_thread(transcript_path.write_text, payload, "utf-8")
                index["sessions"][session.id]["message_count"] = len(messages)
            await asyncio.to_thread(self._write_index_sync, index)

    def _create_session(
        self,
        *,
        req: ChatRequest,
        session_key: str,
        previous_session_id: str | None,
        reset_reason: str | None,
        existing: SessionRecord | None = None,
    ) -> SessionRecord:
        now = datetime.now(timezone.utc)
        scope = self._parse_scope(req.dm_scope)
        session_id = f"session_{new_id()}"
        transcript_path = self.root / f"{session_id}.jsonl"
        return SessionRecord(
            id=session_id,
            agent_id="main",
            session_key=session_key,
            dm_scope=scope,
            main_key=req.main_key or req.conversation_id or (existing.main_key if existing else "main"),
            peer_id=req.peer_id or (existing.peer_id if existing else None),
            channel_id=req.channel_id or (existing.channel_id if existing else None),
            account_id=req.account_id or (existing.account_id if existing else None),
            idle_minutes=req.idle_minutes if req.idle_minutes is not None else (existing.idle_minutes if existing else None),
            transcript_path=str(transcript_path),
            started_at=now,
            last_activity_at=now,
            updated_at=now,
            reset_reason=reset_reason,
            previous_session_id=previous_session_id,
        )

    def _build_session_key(self, req: ChatRequest) -> str:
        scope = self._parse_scope(req.dm_scope)
        main_key = req.main_key or req.conversation_id or "main"
        if scope == SessionScope.MAIN:
            return f"agent:main:{main_key}"
        if scope == SessionScope.PER_PEER:
            peer = req.peer_id or "unknown-peer"
            return f"agent:main:{main_key}:peer:{peer}"
        if scope == SessionScope.PER_CHANNEL_PEER:
            channel = req.channel_id or "direct"
            peer = req.peer_id or "unknown-peer"
            return f"agent:main:{main_key}:channel:{channel}:peer:{peer}"
        account = req.account_id or "default-account"
        channel = req.channel_id or "direct"
        peer = req.peer_id or "unknown-peer"
        return f"agent:main:{main_key}:account:{account}:channel:{channel}:peer:{peer}"

    def _parse_scope(self, raw: str | None) -> SessionScope:
        if not raw:
            return SessionScope.MAIN
        try:
            return SessionScope(raw)
        except ValueError:
            return SessionScope.MAIN

    def _needs_rotation(self, session: SessionRecord) -> bool:
        now = datetime.now(timezone.utc)
        if session.idle_minutes:
            idle_delta = now - session.last_activity_at
            if idle_delta >= timedelta(minutes=session.idle_minutes):
                return True
        local_now = now.astimezone()
        local_last = session.last_activity_at.astimezone(local_now.tzinfo)
        cutoff = self._current_reset_cutoff(local_now)
        return local_last < cutoff <= local_now

    def _current_reset_cutoff(self, local_now: datetime) -> datetime:
        cutoff = datetime.combine(local_now.date(), time(hour=4), tzinfo=local_now.tzinfo)
        if local_now < cutoff:
            cutoff -= timedelta(days=1)
        return cutoff

    def _find_latest_by_key(self, index: dict[str, Any], session_key: str) -> Optional[SessionRecord]:
        matches = [
            SessionRecord(**raw)
            for raw in index["sessions"].values()
            if raw.get("session_key") == session_key
        ]
        if not matches:
            return None
        matches.sort(key=lambda item: item.updated_at, reverse=True)
        return matches[0]

    def _default_index(self) -> dict[str, Any]:
        return {
            "version": 1,
            "active_keys": {},
            "sessions": {},
        }

    def _read_index_sync(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return self._default_index()
        with self.index_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        defaults = self._default_index()
        changed = False
        for key, value in defaults.items():
            if key not in raw:
                raw[key] = deepcopy(value)
                changed = True
        if changed:
            self._write_index_sync(raw)
        return raw

    def _write_index_sync(self, payload: dict[str, Any]):
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        os.replace(tmp, self.index_path)

    def _mark_vector_memory_dirty(self):
        try:
            from app.services.vector_memory import vector_memory

            vector_memory.mark_dirty()
        except Exception:
            pass


session_manager = SessionManager()
