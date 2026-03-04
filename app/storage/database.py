import asyncio
import json
import os
import sqlite3
from copy import deepcopy
from datetime import datetime, timezone
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
    return fact


async def delete_fact(fid: str) -> bool:
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
        original_len = len(state["facts"])
        state["facts"] = [fact for fact in state["facts"] if fact["id"] != fid]
        changed = len(state["facts"]) != original_len
        if changed:
            await asyncio.to_thread(_write_state_sync, state)
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
    return episode


# ── Tasks ──
async def get_all_tasks() -> list[Task]:
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
    tasks = [Task(**item) for item in state["tasks"]]
    return _sort_models(tasks, "created_at", reverse=True)


async def create_task(req: CreateTask) -> Task:
    task = Task(id=new_id(), title=req.title, notes=req.notes, priority=req.priority or "medium")
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
        state["tasks"].append(task.model_dump(mode="json"))
        await asyncio.to_thread(_write_state_sync, state)
    return task


async def update_task(tid: str, req: UpdateTask) -> bool:
    updates = req.model_dump(exclude_none=True)
    if not updates:
        return False

    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
        for task in state["tasks"]:
            if task["id"] != tid:
                continue
            task.update(updates)
            task["updated_at"] = datetime.now(timezone.utc).isoformat()
            await asyncio.to_thread(_write_state_sync, state)
            return True
    return False


async def delete_task(tid: str) -> bool:
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
        original_len = len(state["tasks"])
        state["tasks"] = [task for task in state["tasks"] if task["id"] != tid]
        changed = len(state["tasks"]) != original_len
        if changed:
            await asyncio.to_thread(_write_state_sync, state)
        return changed


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


async def delete_workflow(wid: str) -> bool:
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
        original_len = len(state["workflows"])
        state["workflows"] = [wf for wf in state["workflows"] if wf["id"] != wid]
        changed = len(state["workflows"]) != original_len
        if changed:
            await asyncio.to_thread(_write_state_sync, state)
        return changed


async def update_workflow_last_run(wid: str):
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
        for workflow in state["workflows"]:
            if workflow["id"] == wid:
                workflow["last_run"] = datetime.now(timezone.utc).isoformat()
                await asyncio.to_thread(_write_state_sync, state)
                return


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


async def get_conversation(cid: str) -> list[ChatMessage]:
    async with _STORAGE_LOCK:
        state = await asyncio.to_thread(_read_state_sync)
    record = state["conversations"].get(cid)
    if not record:
        return []
    return [ChatMessage(**message) for message in record.get("messages", [])]


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
