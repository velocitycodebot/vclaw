import asyncio
import json
import os
import sqlite3
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from app.schemas.models import (
    Preferences, Fact, Episode, Task, CreateTask, UpdateTask,
    ToolDef, Workflow, ExecLogEntry, ShellResult, ChatMessage, new_id,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
STORE_PATH = DATA_DIR / "agent_os.json"
LEGACY_DB_PATH = PROJECT_ROOT / "agent_os.db"

_VALID_TABLES = frozenset({
    "facts", "episodes", "tasks", "tools", "workflows",
    "exec_log", "shell_history", "conversations",
})
_STORAGE_LOCK = asyncio.Lock()


def _mark_vector_memory_dirty() -> None:
    try:
        from app.services.vector_memory import vector_memory

        vector_memory.mark_dirty()
    except Exception:
        pass


def _default_state() -> dict[str, Any]:
    return {
        "preferences": Preferences().model_dump(mode="json"),
        "facts": [],
        "episodes": [],
        "tasks": [],
        "tools": [],
        "workflows": [],
        "exec_log": [],
        "shell_history": [],
        "conversations": {},
    }


def _write_state_sync(state: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = STORE_PATH.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=True)
        f.write("\n")
    os.replace(tmp_path, STORE_PATH)


def _migrate_legacy_sqlite_sync() -> bool:
    if STORE_PATH.exists() or not LEGACY_DB_PATH.exists():
        return False

    try:
        conn = sqlite3.connect(LEGACY_DB_PATH)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        return False

    try:
        state = _default_state()

        pref = conn.execute("SELECT * FROM preferences WHERE id=1").fetchone()
        if pref:
            state["preferences"] = {
                "name": pref["name"] or "",
                "tone": pref["tone"] or "friendly",
                "topics": json.loads(pref["topics"] or "[]"),
                "custom": json.loads(pref["custom"] or "{}"),
            }

        for table in ("facts", "episodes", "tasks", "shell_history"):
            rows = conn.execute(f"SELECT * FROM {table}").fetchall()
            state[table] = [dict(row) for row in rows]

        tool_rows = conn.execute("SELECT * FROM tools").fetchall()
        state["tools"] = []
        for row in tool_rows:
            tool = dict(row)
            tool["params"] = json.loads(tool.get("params") or "[]")
            tool["builtin"] = False
            state["tools"].append(tool)

        workflow_rows = conn.execute("SELECT * FROM workflows").fetchall()
        state["workflows"] = []
        for row in workflow_rows:
            workflow = dict(row)
            workflow["steps"] = json.loads(workflow.get("steps") or "[]")
            workflow["schedule"] = json.loads(workflow["schedule"]) if workflow.get("schedule") else None
            workflow["enabled"] = bool(workflow.get("enabled"))
            state["workflows"].append(workflow)

        log_rows = conn.execute("SELECT * FROM exec_log").fetchall()
        state["exec_log"] = []
        for row in log_rows:
            entry = dict(row)
            entry["success"] = bool(entry.get("success"))
            entry["scheduled"] = bool(entry.get("scheduled"))
            entry["full_result"] = json.loads(entry["full_result"]) if entry.get("full_result") else None
            state["exec_log"].append(entry)

        conv_rows = conn.execute("SELECT * FROM conversations").fetchall()
        state["conversations"] = {}
        for row in conv_rows:
            rec = dict(row)
            rec["messages"] = json.loads(rec.get("messages") or "[]")
            state["conversations"][rec["id"]] = rec

        _write_state_sync(state)
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def _read_state_sync() -> dict[str, Any]:
    if not STORE_PATH.exists():
        if not _migrate_legacy_sqlite_sync():
            _write_state_sync(_default_state())

    with STORE_PATH.open("r", encoding="utf-8") as f:
        state = json.load(f)

    defaults = _default_state()
    changed = False
    for key, default_value in defaults.items():
        if key not in state:
            state[key] = deepcopy(default_value)
            changed = True

    if not isinstance(state["conversations"], dict):
        state["conversations"] = {}
        changed = True

    if changed:
        _write_state_sync(state)

    return state


def _sort_models(items: list[Any], attr: str, reverse: bool = False) -> list[Any]:
    return sorted(items, key=lambda item: getattr(item, attr), reverse=reverse)


def _is_active_task_status(status: str) -> bool:
    return status.lower() not in {"done", "completed", "cancelled", "canceled", "failed"}


def _normalize_progress(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    return max(0, min(100, value))


async def initialize():
    async with _STORAGE_LOCK:
        await asyncio.to_thread(_read_state_sync)


# ── Preferences ──
async def get_preferences() -> Preferences:
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
    return Preferences(**state["preferences"])


async def update_preferences(prefs: Preferences):
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
        state["preferences"] = prefs.model_dump(mode="json")
        await asyncio.to_thread(_write_state_sync, state)
    _mark_vector_memory_dirty()


# ── Facts ──
async def get_all_facts() -> list[Fact]:
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
    facts = [Fact(**item) for item in state["facts"]]
    return _sort_models(facts, "created_at", reverse=True)


async def add_fact(text: str, category: Optional[str] = None) -> Fact:
    fact = Fact(id=new_id(), text=text, category=category)
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
        state["facts"].append(fact.model_dump(mode="json"))
        await asyncio.to_thread(_write_state_sync, state)
    _mark_vector_memory_dirty()
    return fact


async def delete_fact(fid: str) -> bool:
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
        original_len = len(state["facts"])
        state["facts"] = [fact for fact in state["facts"] if fact["id"] != fid]
        changed = len(state["facts"]) != original_len
        if changed:
            await asyncio.to_thread(_write_state_sync, state)
            _mark_vector_memory_dirty()
        return changed


# ── Episodes ──
async def get_episodes(limit: int = 50) -> list[Episode]:
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
    episodes = [Episode(**item) for item in state["episodes"]]
    return _sort_models(episodes, "created_at", reverse=True)[:limit]


async def add_episode(summary: str) -> Episode:
    episode = Episode(id=new_id(), summary=summary)
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
        state["episodes"].append(episode.model_dump(mode="json"))
        await asyncio.to_thread(_write_state_sync, state)
    _mark_vector_memory_dirty()
    return episode


# ── Tasks ──
async def get_all_tasks() -> list[Task]:
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
    tasks = [Task(**item) for item in state["tasks"]]
    return _sort_models(tasks, "created_at", reverse=True)


async def get_task(tid: str) -> Optional[Task]:
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
    for item in state["tasks"]:
        if item["id"] == tid:
            return Task(**item)
    return None


async def create_task(req: CreateTask) -> Task:
    now = datetime.now(timezone.utc)
    next_review_at = req.next_review_at
    if next_review_at is None and req.review_interval_minutes:
        next_review_at = now

    task = Task(
        id=new_id(),
        title=req.title,
        notes=req.notes,
        status=req.status or "todo",
        priority=req.priority or "medium",
        progress=_normalize_progress(req.progress) or 0,
        assignee=req.assignee,
        parent_task_id=req.parent_task_id,
        depends_on=list(req.depends_on or []),
        blocked_by=list(req.blocked_by or []),
        due_at=req.due_at,
        next_review_at=next_review_at,
        review_interval_minutes=req.review_interval_minutes,
        last_activity_at=now,
        outcome=req.outcome,
        metadata=dict(req.metadata or {}),
    )
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
        state["tasks"].append(task.model_dump(mode="json"))
        await asyncio.to_thread(_write_state_sync, state)
    _mark_vector_memory_dirty()
    return task


async def update_task(tid: str, req: UpdateTask) -> bool:
    return await patch_task(tid, req) is not None


async def patch_task(tid: str, req: UpdateTask) -> Optional[Task]:
    updates = req.model_dump(exclude_none=True)
    if not updates:
        return None

    if "progress" in updates:
        updates["progress"] = _normalize_progress(updates["progress"])

    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
        for task in state["tasks"]:
            if task["id"] != tid:
                continue
            if req.review_interval_minutes is not None and req.next_review_at is None:
                updates["next_review_at"] = datetime.now(timezone.utc).isoformat()
            task.update(updates)
            now = datetime.now(timezone.utc)
            task["updated_at"] = now.isoformat()
            task["last_activity_at"] = now.isoformat()
            await asyncio.to_thread(_write_state_sync, state)
            _mark_vector_memory_dirty()
            return Task(**task)
    return None


async def delete_task(tid: str) -> bool:
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
        original_len = len(state["tasks"])
        state["tasks"] = [task for task in state["tasks"] if task["id"] != tid]
        changed = len(state["tasks"]) != original_len
        if changed:
            await asyncio.to_thread(_write_state_sync, state)
            _mark_vector_memory_dirty()
        return changed


async def get_tasks_due_for_review(
    now: Optional[datetime] = None,
    limit: int = 20,
) -> list[Task]:
    reference = now or datetime.now(timezone.utc)
    tasks = await get_all_tasks()
    due: list[Task] = []
    for task in tasks:
        if not _is_active_task_status(task.status):
            continue
        if task.next_review_at and task.next_review_at <= reference:
            due.append(task)
            continue
        if task.review_interval_minutes and task.next_review_at is None:
            due.append(task)
    return sorted(
        due,
        key=lambda task: task.next_review_at or task.last_activity_at or task.created_at,
    )[:limit]


async def reschedule_task_review(tid: str, base_time: Optional[datetime] = None) -> Optional[Task]:
    task = await get_task(tid)
    if not task or not task.review_interval_minutes:
        return task
    next_review = (base_time or datetime.now(timezone.utc)) + timedelta(minutes=task.review_interval_minutes)
    return await patch_task(
        tid,
        UpdateTask(
            next_review_at=next_review,
        ),
    )


# ── Tools ──
async def get_all_tools() -> list[ToolDef]:
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
    tools = [ToolDef(**item) for item in state["tools"]]
    return _sort_models(tools, "created_at")


async def save_tool(tool: ToolDef):
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
        state["tools"] = [
            item for item in state["tools"]
            if item["id"] != tool.id and item["name"] != tool.name
        ]
        state["tools"].append(tool.model_dump(mode="json"))
        await asyncio.to_thread(_write_state_sync, state)


async def delete_tool(tid: str) -> bool:
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
        original_len = len(state["tools"])
        state["tools"] = [tool for tool in state["tools"] if tool["id"] != tid]
        changed = len(state["tools"]) != original_len
        if changed:
            await asyncio.to_thread(_write_state_sync, state)
        return changed


# ── Workflows ──
async def get_all_workflows() -> list[Workflow]:
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
    workflows = [Workflow(**item) for item in state["workflows"]]
    return _sort_models(workflows, "created_at")


async def save_workflow(wf: Workflow):
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
        state["workflows"] = [item for item in state["workflows"] if item["id"] != wf.id]
        state["workflows"].append(wf.model_dump(mode="json"))
        await asyncio.to_thread(_write_state_sync, state)
    _mark_vector_memory_dirty()


async def delete_workflow(wid: str) -> bool:
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
        original_len = len(state["workflows"])
        state["workflows"] = [wf for wf in state["workflows"] if wf["id"] != wid]
        changed = len(state["workflows"]) != original_len
        if changed:
            await asyncio.to_thread(_write_state_sync, state)
            _mark_vector_memory_dirty()
        return changed


async def update_workflow_last_run(wid: str):
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
        for workflow in state["workflows"]:
            if workflow["id"] == wid:
                workflow["last_run"] = datetime.now(timezone.utc).isoformat()
                await asyncio.to_thread(_write_state_sync, state)
                return


async def set_workflow_enabled(wid: str, enabled: bool) -> Optional[Workflow]:
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
        for workflow in state["workflows"]:
            if workflow["id"] != wid:
                continue
            workflow["enabled"] = enabled
            await asyncio.to_thread(_write_state_sync, state)
            _mark_vector_memory_dirty()
            return Workflow(**workflow)
    return None


# ── Exec Log ──
async def add_log_entry(entry: ExecLogEntry):
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
        state["exec_log"].append(entry.model_dump(mode="json"))
        await asyncio.to_thread(_write_state_sync, state)


async def get_log_entries(limit: int = 100) -> list[ExecLogEntry]:
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
    entries = [ExecLogEntry(**item) for item in state["exec_log"]]
    return _sort_models(entries, "created_at", reverse=True)[:limit]


async def clear_logs():
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
        state["exec_log"] = []
        await asyncio.to_thread(_write_state_sync, state)


# ── Shell History ──
async def add_shell_history(result: ShellResult):
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
        state["shell_history"].append(result.model_dump(mode="json"))
        await asyncio.to_thread(_write_state_sync, state)


async def get_shell_history(limit: int = 50) -> list[ShellResult]:
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
    history = [ShellResult(**item) for item in state["shell_history"]]
    return _sort_models(history, "created_at", reverse=True)[:limit]


# ── Conversations ──
async def save_conversation(cid: str, messages: list[ChatMessage]):
    record = {
        "id": cid,
        "messages": [message.model_dump(mode="json") for message in messages],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
        state["conversations"][cid] = record
        await asyncio.to_thread(_write_state_sync, state)
    _mark_vector_memory_dirty()


async def get_conversation(cid: str) -> list[ChatMessage]:
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
    record = state["conversations"].get(cid)
    if not record:
        return []
    return [ChatMessage(**message) for message in record.get("messages", [])]


async def get_all_conversations() -> dict[str, list[ChatMessage]]:
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
    conversations: dict[str, list[ChatMessage]] = {}
    for cid, record in state["conversations"].items():
        conversations[cid] = [ChatMessage(**message) for message in record.get("messages", [])]
    return conversations


async def count_table(table: str) -> int:
    if table not in _VALID_TABLES:
        raise ValueError(f"Invalid table name: {table}")

    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)

    if table == "conversations":
        return len(state["conversations"])
    return len(state[table])


async def reset_all():
    async with _STORAGE_LOCK:
        await asyncio.to_thread(_write_state_sync, _default_state())
    _mark_vector_memory_dirty()
