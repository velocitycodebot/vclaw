from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import app.storage.database as db
from app.schemas.models import AgentStatus, Task
from app.services.agents import AgentRegistry, run_agent_loop

_TERMINAL_STATUSES = {
    AgentStatus.COMPLETED,
    AgentStatus.FAILED,
    AgentStatus.TERMINATED,
}


def _children_done(registry: AgentRegistry, parent_id: str) -> bool:
    children = registry.list_children(parent_id)
    return bool(children) and all(child.status in _TERMINAL_STATUSES for child in children)


def _child_digest(registry: AgentRegistry, parent_id: str) -> str:
    children = [
        {
            "id": child.id,
            "status": child.status.value,
            "updated_at": child.updated_at.isoformat(),
            "result": child.result or "",
        }
        for child in registry.list_children(parent_id)
    ]
    return json.dumps(children, sort_keys=True)


def _child_summary(registry: AgentRegistry, parent_id: str) -> str:
    return "\n".join(
        f"- {child.name} ({child.id}): {child.status.value}; result={child.result or 'no result'}"
        for child in registry.list_children(parent_id)
    )


def _format_review_task(task: Task) -> str:
    parts = [
        f"- {task.title} [{task.status}]",
        f"progress={task.progress}%",
    ]
    if task.assignee:
        parts.append(f"assignee={task.assignee}")
    if task.blocked_by:
        parts.append(f"blocked_by={', '.join(task.blocked_by)}")
    if task.due_at:
        parts.append(f"due={task.due_at.isoformat()}")
    return "; ".join(parts)


async def _resume_waiting_agents(registry: AgentRegistry, tool_registry, workflow_engine):
    for agent in registry.list_agents():
        if agent.status != AgentStatus.WAITING_SUBAGENT:
            continue
        if registry.loop_active(agent.id):
            continue
        if not _children_done(registry, agent.id):
            continue

        digest = _child_digest(registry, agent.id)
        if digest == agent.last_children_digest:
            continue

        agent.last_children_digest = digest
        prompt = (
            "All sub-agents assigned to you have finished.\n"
            f"{_child_summary(registry, agent.id)}\n\n"
            "Synthesize the results, update tasks or memory if needed, and continue the work."
        )
        asyncio.create_task(run_agent_loop(registry, tool_registry, workflow_engine, agent.id, prompt))


async def _review_due_tasks(registry: AgentRegistry, tool_registry, workflow_engine):
    main = registry.get_agent("main")
    if not main or registry.loop_active("main"):
        return
    if main.status not in {AgentStatus.IDLE, AgentStatus.WAITING_INPUT}:
        return

    due_tasks = await db.get_tasks_due_for_review(limit=5)
    if not due_tasks:
        return

    for task in due_tasks:
        await db.reschedule_task_review(task.id)

    prompt = (
        "Scheduled task review triggered. Check these tasks, update their progress, "
        "and decide whether any tool calls, workflows, or sub-agents are needed.\n"
        + "\n".join(_format_review_task(task) for task in due_tasks)
    )
    asyncio.create_task(run_agent_loop(registry, tool_registry, workflow_engine, "main", prompt))


async def run_orchestrator_loop(get_registry, get_tool_registry, get_workflow_engine):
    while True:
        await asyncio.sleep(5)
        try:
            registry = get_registry()
            tool_registry = get_tool_registry()
            workflow_engine = get_workflow_engine()
            await _resume_waiting_agents(registry, tool_registry, workflow_engine)
            await _review_due_tasks(registry, tool_registry, workflow_engine)
        except Exception as exc:
            print(f"Orchestrator supervisor error: {exc}")
