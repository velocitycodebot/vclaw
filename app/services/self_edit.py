from __future__ import annotations

import asyncio
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from app.schemas.models import SelfEditRunRequest, SelfEditSession, SelfEditWrite, new_id
from app.services.shell import execute_command

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "self_edits"


class SelfEditService:
    def __init__(self):
        self._lock = asyncio.Lock()

    async def initialize(self):
        await asyncio.to_thread(DATA_DIR.mkdir, parents=True, exist_ok=True)

    async def run(self, req: SelfEditRunRequest) -> SelfEditSession:
        if not req.writes:
            raise ValueError("Self-edit run requires at least one file write")

        async with self._lock:
            workspace = self._resolve_workspace(req.workspace)
            session_id = f"edit_{new_id()}"
            session_dir = DATA_DIR / session_id
            snapshot_dir = session_dir / "snapshot"
            await asyncio.to_thread(snapshot_dir.mkdir, parents=True, exist_ok=True)

            changed_files: list[str] = []
            created_files: list[str] = []
            writes = list(req.writes)

            for write in writes:
                target = self._resolve_target(workspace, write.path)
                relative = str(target.relative_to(workspace))
                if target.exists():
                    backup_path = snapshot_dir / relative
                    await asyncio.to_thread(backup_path.parent.mkdir, parents=True, exist_ok=True)
                    await asyncio.to_thread(shutil.copy2, target, backup_path)
                    changed_files.append(relative)
                else:
                    created_files.append(relative)

            for write in writes:
                target = self._resolve_target(workspace, write.path)
                await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
                await asyncio.to_thread(target.write_text, write.content, "utf-8")

            session = SelfEditSession(
                id=session_id,
                workspace=str(workspace),
                writes=writes,
                eval_command=req.eval_command,
                test_command=req.test_command,
                rollback_on_failure=req.rollback_on_failure,
                status="applied",
                changed_files=changed_files,
                created_files=created_files,
                snapshot_dir=str(snapshot_dir),
                notes=req.notes,
            )

            if req.eval_command:
                session.eval_result = await execute_command(
                    req.eval_command,
                    working_dir=str(workspace),
                    timeout_secs=120,
                )
                if session.eval_result.exit_code != 0:
                    session.status = "eval_failed"
                    session.error = "Evaluation command failed"

            if req.test_command and session.status not in {"eval_failed"}:
                session.test_result = await execute_command(
                    req.test_command,
                    working_dir=str(workspace),
                    timeout_secs=300,
                )
                if session.test_result.exit_code != 0:
                    session.status = "test_failed"
                    session.error = "Test command failed"

            if session.status in {"eval_failed", "test_failed"} and req.rollback_on_failure:
                session = await self._rollback_loaded(session, update_status=False)
                session.status = "rolled_back"
            elif session.status == "applied" and (req.eval_command or req.test_command):
                session.status = "verified"

            session.updated_at = datetime.now(timezone.utc)
            await self._save(session)
            return session

    async def rollback(self, session_id: str) -> SelfEditSession:
        async with self._lock:
            session = await self.get(session_id)
            if session is None:
                raise ValueError(f"Self-edit session '{session_id}' not found")
            session = await self._rollback_loaded(session)
            await self._save(session)
            return session

    async def get(self, session_id: str) -> SelfEditSession | None:
        session_path = DATA_DIR / session_id / "session.json"
        if not session_path.exists():
            return None
        data = await asyncio.to_thread(session_path.read_text, "utf-8")
        return SelfEditSession(**json.loads(data))

    async def list_sessions(self, limit: int = 20) -> list[SelfEditSession]:
        if not DATA_DIR.exists():
            return []
        sessions: list[SelfEditSession] = []
        for entry in sorted(DATA_DIR.iterdir(), reverse=True):
            session_path = entry / "session.json"
            if not session_path.exists():
                continue
            data = await asyncio.to_thread(session_path.read_text, "utf-8")
            sessions.append(SelfEditSession(**json.loads(data)))
            if len(sessions) >= limit:
                break
        sessions.sort(key=lambda item: item.created_at, reverse=True)
        return sessions[:limit]

    async def _rollback_loaded(self, session: SelfEditSession, *, update_status: bool = True) -> SelfEditSession:
        workspace = self._resolve_workspace(session.workspace)
        snapshot_dir = Path(session.snapshot_dir)

        for relative in session.created_files:
            target = self._resolve_target(workspace, relative)
            if target.exists():
                await asyncio.to_thread(target.unlink)

        for relative in session.changed_files:
            target = self._resolve_target(workspace, relative)
            backup = snapshot_dir / relative
            if backup.exists():
                await asyncio.to_thread(target.parent.mkdir, parents=True, exist_ok=True)
                await asyncio.to_thread(shutil.copy2, backup, target)

        session.rolled_back = True
        if update_status:
            session.status = "rolled_back"
        session.updated_at = datetime.now(timezone.utc)
        return session

    async def _save(self, session: SelfEditSession):
        session_dir = DATA_DIR / session.id
        await asyncio.to_thread(session_dir.mkdir, parents=True, exist_ok=True)
        session_path = session_dir / "session.json"
        await asyncio.to_thread(
            session_path.write_text,
            json.dumps(session.model_dump(mode="json"), indent=2, ensure_ascii=True) + "\n",
            "utf-8",
        )

    def _resolve_workspace(self, workspace: str | None) -> Path:
        path = Path(workspace or PROJECT_ROOT).resolve()
        if not self._is_within(path, PROJECT_ROOT):
            raise ValueError(f"Workspace must stay within {PROJECT_ROOT}")
        return path

    def _resolve_target(self, workspace: Path, path_str: str) -> Path:
        target = (workspace / path_str).resolve()
        if not self._is_within(target, workspace):
            raise ValueError(f"Target path must stay within workspace: {path_str}")
        return target

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False


self_edit_service = SelfEditService()
