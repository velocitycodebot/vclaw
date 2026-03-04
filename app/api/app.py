import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import app.storage.database as db
from app.schemas.models import (
    ChatRequest, CreateTask, UpdateTask, CreateToolRequest, ToolExecRequest,
    ToolDef, ToolResult, ShellExecRequest, CreateWorkflow, Workflow,
    ExecLogEntry, MemorySearchRequest, Preferences, SelfEditRunRequest, SpawnAgentRequest, AgentConfig,
    AgentMessageRequest, AgentMsgType, AgentStatus, SystemStatus, new_id,
)
from app.services.tools import ToolRegistry
from app.services.workflows import WorkflowEngine, execute_workflow, run_scheduler
from app.services.agents import AgentRegistry, execute_agent_step, run_agent_loop
from app.services.orchestrator import run_orchestrator_loop
from app.services.self_edit import self_edit_service
from app.services.vector_memory import vector_memory
import app.services.shell as shell_mod

# ── Global state ──
tool_registry = ToolRegistry()
workflow_engine = WorkflowEngine()
agent_registry = AgentRegistry()
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await db.initialize()
    print("✓ Database initialized")
    await vector_memory.initialize()
    await self_edit_service.initialize()
    print("✓ Vector memory initialized")
    print("✓ Self-edit pipeline initialized")

    tool_registry.register_builtins()
    for tool in await db.get_all_tools():
        tool_registry.register(tool)
    print(f"✓ Tool registry loaded ({tool_registry.count()} tools)")

    for wf in await db.get_all_workflows():
        workflow_engine.register(wf)
    print(f"✓ Workflow engine loaded ({workflow_engine.count()} workflows)")

    print("✓ Agent registry initialized (main orchestrator ready)")

    # Start scheduler
    asyncio.create_task(
        run_scheduler(lambda: workflow_engine, lambda: tool_registry, None)
    )
    asyncio.create_task(
        run_orchestrator_loop(lambda: agent_registry, lambda: tool_registry, lambda: workflow_engine)
    )
    host = os.environ.get("VCLAW_HOST", "0.0.0.0")
    port = os.environ.get("VCLAW_PORT", "3000")
    print("✓ Workflow scheduler started")
    print("✓ Agent supervisor started")
    print(f"⚡ AgentOS ready on http://{host}:{port}")

    yield
    # Shutdown


app = FastAPI(title="AgentOS", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ═══════════════════════════════════════════
# CHAT — agentic loop
# ═══════════════════════════════════════════

@app.post("/api/chat")
async def chat_handler(req: ChatRequest):
    conv_id = req.conversation_id or "main_conv"

    # Load history into main agent
    existing = await db.get_conversation(conv_id)
    main = agent_registry.get_agent("main")
    if main and not main.messages and existing:
        main.messages = existing

    # Execute main agent step
    step = await execute_agent_step(
        agent_registry, tool_registry, workflow_engine, "main", req.message
    )

    final_text = step.response
    all_tool_calls = list(step.tool_calls)
    all_spawned = list(step.agents_spawned)
    all_tools_created = list(step.tools_created)
    all_workflows_created = list(step.workflows_created)
    all_tasks_changed = list(step.tasks_changed)
    all_memory_updates = list(step.memory_facts_added)
    latest_preferences = step.preferences_updated

    # Continue agentic loop if Claude is waiting on tool results or follow-up actions
    if step.needs_model_followup:
        for _ in range(6):
            next_step = await execute_agent_step(
                agent_registry, tool_registry, workflow_engine, "main", None
            )
            if next_step.tool_calls:
                all_tool_calls.extend(next_step.tool_calls)
            all_spawned.extend(next_step.agents_spawned)
            all_tools_created.extend(next_step.tools_created)
            all_workflows_created.extend(next_step.workflows_created)
            all_tasks_changed.extend(next_step.tasks_changed)
            all_memory_updates.extend(next_step.memory_facts_added)
            if next_step.preferences_updated:
                latest_preferences = next_step.preferences_updated
            if next_step.response:
                final_text = next_step.response
            if next_step.needs_model_followup:
                continue
            break

    # Agent info
    agents_info = []
    for aid in all_spawned:
        a = agent_registry.get_agent(aid)
        if a:
            agents_info.append({
                "id": a.id, "name": a.name,
                "role": a.config.role, "goal": a.config.goal,
                "status": a.status.value,
            })

    # Save conversation
    main = agent_registry.get_agent("main")
    if main:
        await db.save_conversation(conv_id, main.messages)

    summary = f"User: {req.message[:60]}… → Agent responded" if len(req.message) > 60 else f"User: {req.message} → Agent responded"
    await db.add_episode(summary)

    return {
        "message": final_text,
        "conversation_id": conv_id,
        "tool_calls": [tc.model_dump() for tc in all_tool_calls],
        "agents_spawned": agents_info,
        "tools_created": [tool.model_dump() for tool in all_tools_created],
        "workflows_created": [workflow.model_dump() for workflow in all_workflows_created],
        "tasks_changed": [task.model_dump(mode="json") for task in all_tasks_changed],
        "memory_updates": {
            "facts_added": all_memory_updates,
            "preferences": latest_preferences,
        },
    }


# ═══════════════════════════════════════════
# MEMORY
# ═══════════════════════════════════════════

@app.get("/api/memory")
async def get_memory():
    facts = await db.get_all_facts()
    episodes = await db.get_episodes(50)
    return {"facts": [f.model_dump() for f in facts], "episodes": [e.model_dump() for e in episodes]}


@app.post("/api/memory/search")
async def search_memory(req: MemorySearchRequest):
    hits = await vector_memory.search(req.query, limit=req.limit, source_types=req.source_types)
    return {"hits": [hit.model_dump(mode="json") for hit in hits]}


@app.get("/api/memory/vector-status")
async def vector_memory_status():
    return (await vector_memory.status()).model_dump(mode="json")


@app.post("/api/memory/facts")
async def add_fact_route(body: dict):
    text = body.get("text", "")
    if not text:
        raise HTTPException(400, "Missing 'text'")
    f = await db.add_fact(text, body.get("category"))
    return f.model_dump()


@app.delete("/api/memory/facts/{fid}")
async def delete_fact_route(fid: str):
    return {"deleted": await db.delete_fact(fid)}


@app.get("/api/memory/episodes")
async def get_episodes_route():
    eps = await db.get_episodes(50)
    return [e.model_dump() for e in eps]


# ═══════════════════════════════════════════
# PREFERENCES
# ═══════════════════════════════════════════

@app.get("/api/preferences")
async def get_preferences():
    return (await db.get_preferences()).model_dump()


@app.put("/api/preferences")
async def update_preferences(prefs: Preferences):
    await db.update_preferences(prefs)
    return {"ok": True}


# ═══════════════════════════════════════════
# TASKS
# ═══════════════════════════════════════════

@app.get("/api/tasks")
async def get_tasks():
    return [t.model_dump() for t in await db.get_all_tasks()]


@app.post("/api/tasks")
async def create_task_route(req: CreateTask):
    return (await db.create_task(req)).model_dump()


@app.put("/api/tasks/{tid}")
async def update_task_route(tid: str, req: UpdateTask):
    return {"updated": await db.update_task(tid, req)}


@app.delete("/api/tasks/{tid}")
async def delete_task_route(tid: str):
    return {"deleted": await db.delete_task(tid)}


# ═══════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════

@app.get("/api/tools")
async def get_tools():
    return [t.model_dump() for t in tool_registry.list_all()]


@app.post("/api/tools")
async def create_tool_route(req: CreateToolRequest):
    tool = ToolDef(
        id=new_id(), name=req.name, description=req.description,
        params=req.params, builtin=False, handler=req.handler,
    )
    await db.save_tool(tool)
    tool_registry.register(tool)
    return tool.model_dump()


@app.delete("/api/tools/{tid}")
async def delete_tool_route(tid: str):
    t = tool_registry.get_by_id(tid)
    if t:
        tool_registry.remove(t.name)
    await db.delete_tool(tid)
    return {"deleted": True}


@app.post("/api/tools/{tid}/execute")
async def execute_tool_route(tid: str, req: ToolExecRequest):
    t = tool_registry.get_by_id(tid) or tool_registry.get(tid)
    if not t:
        return ToolResult(success=False, error="Tool not found").model_dump()
    result = await tool_registry.execute(t.name, req.params)
    log = ExecLogEntry(
        id=new_id(), source="tool", source_name=t.name,
        input_summary=json.dumps(req.params)[:200],
        result_summary=str(result.result)[:200] if result.result else "",
        success=result.success, execution_ms=result.execution_ms,
    )
    await db.add_log_entry(log)
    return result.model_dump()


# ═══════════════════════════════════════════
# SHELL
# ═══════════════════════════════════════════

@app.post("/api/shell/execute")
async def execute_shell(req: ShellExecRequest):
    result = await shell_mod.execute_command(
        req.command, req.working_dir, req.timeout_secs, req.stdin
    )
    await db.add_shell_history(result)
    log = ExecLogEntry(
        id=new_id(), source="shell", source_name=result.command[:50],
        input_summary=result.command[:200],
        result_summary=(result.stdout if result.exit_code == 0 else result.stderr)[:200],
        success=result.exit_code == 0, execution_ms=result.execution_ms,
    )
    await db.add_log_entry(log)
    return result.model_dump()


@app.get("/api/shell/history")
async def get_shell_history():
    return [h.model_dump() for h in await db.get_shell_history(50)]


# ═══════════════════════════════════════════
# SELF-EDIT
# ═══════════════════════════════════════════

@app.post("/api/self-edit/run")
async def run_self_edit(req: SelfEditRunRequest):
    session = await self_edit_service.run(req)
    return session.model_dump(mode="json")


@app.get("/api/self-edit/sessions")
async def list_self_edit_sessions():
    sessions = await self_edit_service.list_sessions(50)
    return {"sessions": [session.model_dump(mode="json") for session in sessions]}


@app.get("/api/self-edit/sessions/{session_id}")
async def get_self_edit_session(session_id: str):
    session = await self_edit_service.get(session_id)
    if session is None:
        raise HTTPException(404, "Self-edit session not found")
    return session.model_dump(mode="json")


@app.post("/api/self-edit/sessions/{session_id}/rollback")
async def rollback_self_edit_session(session_id: str):
    session = await self_edit_service.rollback(session_id)
    return session.model_dump(mode="json")


# ═══════════════════════════════════════════
# WORKFLOWS
# ═══════════════════════════════════════════

@app.get("/api/workflows")
async def get_workflows():
    return [w.model_dump() for w in workflow_engine.list_all()]


@app.post("/api/workflows")
async def create_workflow_route(req: CreateWorkflow):
    wf = Workflow(
        id=new_id(), name=req.name, description=req.description,
        steps=req.steps, schedule=req.schedule,
        enabled=req.schedule is not None,
    )
    await db.save_workflow(wf)
    workflow_engine.register(wf)
    return wf.model_dump()


@app.delete("/api/workflows/{wid}")
async def delete_workflow_route(wid: str):
    workflow_engine.remove(wid)
    await db.delete_workflow(wid)
    return {"deleted": True}


@app.post("/api/workflows/{wid}/run")
async def run_workflow_route(wid: str):
    wf = workflow_engine.get(wid)
    if not wf:
        raise HTTPException(404, "Workflow not found")
    results = await execute_workflow(wf, tool_registry)
    success = all(r.result.success for r in results)
    workflow_engine.update_last_run(wid)
    await db.update_workflow_last_run(wid)
    log = ExecLogEntry(
        id=new_id(), source="workflow", source_name=wf.name,
        input_summary=f"{len(wf.steps)} steps",
        result_summary=f"{sum(1 for r in results if r.result.success)}/{len(results)} ok",
        success=success, execution_ms=sum(r.result.execution_ms for r in results),
    )
    await db.add_log_entry(log)
    return {"success": success, "results": [r.model_dump() for r in results]}


@app.post("/api/workflows/{wid}/toggle")
async def toggle_workflow_route(wid: str):
    enabled = workflow_engine.toggle(wid)
    if enabled is None:
        raise HTTPException(404, "Workflow not found")
    await db.set_workflow_enabled(wid, enabled)
    return {"enabled": enabled}


# ═══════════════════════════════════════════
# LOGS
# ═══════════════════════════════════════════

@app.get("/api/logs")
async def get_logs():
    return [e.model_dump() for e in await db.get_log_entries(100)]


@app.delete("/api/logs")
async def clear_logs():
    await db.clear_logs()
    return {"cleared": True}


# ═══════════════════════════════════════════
# AGENTS
# ═══════════════════════════════════════════

@app.get("/api/agents")
async def get_agents():
    agents = [{
        "id": a.id, "name": a.name, "role": a.config.role, "goal": a.config.goal,
        "status": a.status.value, "parent_id": a.parent_id, "children": a.children,
        "iterations": a.iterations, "max_iterations": a.config.max_iterations,
        "result": a.result, "created_at": a.created_at.isoformat(),
        "updated_at": a.updated_at.isoformat(),
        "completed_at": a.completed_at.isoformat() if a.completed_at else None,
        "message_count": len(a.messages),
    } for a in agent_registry.list_agents()]
    return {"agents": agents, "total": agent_registry.total_count(), "active": agent_registry.active_count()}


@app.get("/api/agents/tree")
async def get_agent_tree():
    return agent_registry.get_agent_tree()


@app.get("/api/agents/{aid}")
async def get_agent_detail(aid: str):
    a = agent_registry.get_agent(aid)
    if not a:
        raise HTTPException(404, "Agent not found")
    return {
        "id": a.id, "name": a.name, "config": a.config.model_dump(),
        "status": a.status.value, "parent_id": a.parent_id,
        "children": a.children, "iterations": a.iterations,
        "result": a.result,
        "messages": [m.model_dump() for m in a.messages],
        "created_at": a.created_at.isoformat(),
        "updated_at": a.updated_at.isoformat(),
        "completed_at": a.completed_at.isoformat() if a.completed_at else None,
    }


@app.post("/api/agents/spawn")
async def spawn_agent_route(req: SpawnAgentRequest):
    config = AgentConfig(
        role=req.role, goal=req.goal,
        tools=req.tools or [],
        max_iterations=req.max_iterations or 20,
        timeout_secs=req.timeout_secs or 300,
        auto_terminate=True,
        report_to=req.parent_id or "main",
    )
    parent = req.parent_id or "main"
    aid = agent_registry.spawn_agent(req.name, config, parent)

    initial_task = req.initial_task or req.goal
    asyncio.create_task(
        run_agent_loop(agent_registry, tool_registry, workflow_engine, aid, initial_task)
    )
    return {"id": aid, "name": req.name, "status": "running"}


@app.post("/api/agents/{aid}/message")
async def message_agent_route(aid: str, req: AgentMessageRequest):
    mt_map = {"task": AgentMsgType.TASK, "directive": AgentMsgType.DIRECTIVE,
              "query": AgentMsgType.QUERY}
    mt = mt_map.get(req.msg_type or "", AgentMsgType.STATUS)
    mid = agent_registry.send_message("user", aid, req.content, mt)

    agent = agent_registry.get_agent(aid)
    if agent and agent.status == AgentStatus.IDLE:
        asyncio.create_task(
            run_agent_loop(agent_registry, tool_registry, workflow_engine, aid, req.content)
        )

    return {"message_id": mid, "sent": True}


@app.post("/api/agents/{aid}/terminate")
async def terminate_agent_route(aid: str):
    return {"terminated": agent_registry.terminate_agent(aid)}


@app.get("/api/agents/{aid}/messages")
async def get_agent_messages_route(aid: str):
    msgs = agent_registry.get_messages_for(aid)
    return {"messages": [{
        "id": m.id, "from": m.from_agent, "to": m.to_agent,
        "content": m.content, "type": m.msg_type.value,
        "created_at": m.created_at.isoformat(),
    } for m in msgs]}


# ═══════════════════════════════════════════
# SYSTEM
# ═══════════════════════════════════════════

@app.get("/api/status")
async def system_status():
    tools = tool_registry.list_all()
    wfs = workflow_engine.list_all()
    all_tasks = await db.get_all_tasks()
    return SystemStatus(
        total_tools=len(tools),
        builtin_tools=sum(1 for t in tools if t.builtin),
        custom_tools=sum(1 for t in tools if not t.builtin),
        total_workflows=len(wfs),
        active_workflows=sum(1 for w in wfs if w.enabled),
        total_facts=await db.count_table("facts"),
        total_episodes=await db.count_table("episodes"),
        total_tasks=len(all_tasks),
        active_tasks=sum(
            1 for t in all_tasks
            if t.status.lower() not in {"done", "completed", "cancelled", "canceled"}
        ),
        total_logs=await db.count_table("exec_log"),
        total_shell_commands=await db.count_table("shell_history"),
        total_agents=agent_registry.total_count(),
        active_agents=agent_registry.active_count(),
    ).model_dump()


@app.post("/api/reset")
async def reset_all():
    global tool_registry, workflow_engine, agent_registry
    await db.reset_all()
    tool_registry = ToolRegistry()
    tool_registry.register_builtins()
    workflow_engine = WorkflowEngine()
    agent_registry = AgentRegistry()
    return {"reset": True}


# ── Serve frontend ──
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=3000, reload=False)
