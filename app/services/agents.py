import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from app.schemas.models import (
    AgentDef, AgentConfig, AgentStatus, AgentMessage, AgentMsgType,
    ChatMessage, ExecLogEntry, ToolCallResult, ToolDef, ToolParam,
    Workflow, AgentStepResult, new_id,
)
import app.clients.claude as claude_client
import app.storage.database as db


class AgentRegistry:
    def __init__(self):
        self.agents: dict[str, AgentDef] = {}
        self.messages: list[AgentMessage] = []
        self.main_agent_id = "main"

        # Create main orchestrator
        main = AgentDef(
            id="main",
            name="Orchestrator",
            config=AgentConfig(
                role="Main orchestrator agent that manages all sub-agents",
                goal="Serve the user by delegating tasks to specialized sub-agents and coordinating their work",
                max_iterations=100,
                timeout_secs=0,
                auto_terminate=False,
            ),
            status=AgentStatus.IDLE,
        )
        self.agents["main"] = main

    def spawn_agent(self, name: str, config: AgentConfig, parent_id: Optional[str] = None) -> str:
        aid = f"agent_{new_id()}"
        agent = AgentDef(
            id=aid, name=name, config=config,
            status=AgentStatus.IDLE,
            parent_id=parent_id,
        )
        if parent_id and parent_id in self.agents:
            self.agents[parent_id].children.append(aid)
        self.agents[aid] = agent
        print(f"🤖 Spawned agent '{name}' ({aid})")
        return aid

    def get_agent(self, aid: str) -> Optional[AgentDef]:
        return self.agents.get(aid)

    def list_agents(self) -> list[AgentDef]:
        return sorted(self.agents.values(), key=lambda a: a.created_at)

    def list_children(self, parent_id: str) -> list[AgentDef]:
        return [a for a in self.agents.values() if a.parent_id == parent_id]

    def send_message(self, from_id: str, to_id: str, content: str, msg_type: AgentMsgType) -> str:
        mid = new_id()
        msg = AgentMessage(
            id=mid, from_agent=from_id, to_agent=to_id,
            content=content, msg_type=msg_type,
        )
        if to_id in self.agents:
            self.agents[to_id].messages.append(ChatMessage(
                role="user",
                content=f"[Message from agent '{from_id}'] {content}",
                timestamp=datetime.now(timezone.utc),
            ))
        self.messages.append(msg)
        return mid

    def get_messages_for(self, aid: str) -> list[AgentMessage]:
        return [m for m in self.messages if m.to_agent == aid or m.from_agent == aid]

    def terminate_agent(self, aid: str) -> bool:
        if aid == "main":
            return False
        agent = self.agents.get(aid)
        if not agent:
            return False
        agent.status = AgentStatus.TERMINATED
        agent.completed_at = datetime.now(timezone.utc)
        agent.updated_at = datetime.now(timezone.utc)
        for child_id in list(agent.children):
            self.terminate_agent(child_id)
        return True

    def get_agent_tree(self) -> dict:
        return self._build_tree("main")

    def _build_tree(self, aid: str) -> dict:
        agent = self.agents.get(aid)
        if not agent:
            return {"id": aid, "error": "not found"}
        children = [self._build_tree(cid) for cid in agent.children]
        return {
            "id": agent.id, "name": agent.name,
            "status": agent.status.value, "role": agent.config.role,
            "goal": agent.config.goal, "iterations": agent.iterations,
            "result": agent.result, "children": children,
            "created_at": agent.created_at.isoformat(),
        }

    def active_count(self) -> int:
        return sum(1 for a in self.agents.values()
                   if a.status in (AgentStatus.RUNNING, AgentStatus.WAITING_SUBAGENT))

    def total_count(self) -> int:
        return len(self.agents)


def _tool_result_block(tool_use_id: str, payload: Any, *, is_error: bool = False) -> dict[str, Any]:
    if isinstance(payload, str):
        content = payload
    else:
        content = json.dumps(payload, ensure_ascii=True, indent=2)[:50_000]
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }
    if is_error:
        block["is_error"] = True
    return block


async def _apply_preference_updates(updates: dict[str, Any]) -> dict[str, Any]:
    current = await db.get_preferences()
    merged = current.model_copy(deep=True)
    custom = dict(merged.custom)

    for key, value in updates.items():
        if key == "name" and isinstance(value, str):
            merged.name = value
        elif key == "tone" and isinstance(value, str):
            merged.tone = value
        elif key == "topics" and isinstance(value, list):
            merged.topics = [str(item) for item in value]
        elif key == "custom" and isinstance(value, dict):
            custom.update(value)
        else:
            custom[key] = value

    merged.custom = custom
    await db.update_preferences(merged)
    return merged.model_dump(mode="json")


# ═══════════════════════════════════════════
# AGENT STEP EXECUTOR
# ═══════════════════════════════════════════

async def execute_agent_step(
    registry: AgentRegistry,
    tool_registry,
    workflow_engine,
    agent_id: str,
    user_input: Optional[str] = None,
) -> AgentStepResult:
    agent = registry.get_agent(agent_id)
    if not agent:
        raise Exception(f"Agent '{agent_id}' not found")

    if agent.status in (AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.TERMINATED):
        raise Exception(f"Agent '{agent_id}' is {agent.status.value}")

    agent.status = AgentStatus.RUNNING
    agent.iterations += 1
    agent.updated_at = datetime.now(timezone.utc)

    # Build context
    prefs = await db.get_preferences()
    tasks = await db.get_all_tasks()
    facts = await db.get_all_facts()
    episodes = await db.get_episodes(10)

    all_tools = tool_registry.list_all()
    if agent.config.tools:
        filtered_tools = [t for t in all_tools if t.name in agent.config.tools or t.name == "shell"]
    else:
        filtered_tools = all_tools

    workflows = workflow_engine.list_all()

    sibling_info = []
    if agent.parent_id:
        for s in registry.list_children(agent.parent_id):
            if s.id != agent_id:
                sibling_info.append(f"- {s.name} ({s.id}): {s.status.value} — {s.config.role}")

    child_info = []
    for c in registry.list_children(agent_id):
        line = f"- {c.name} ({c.id}): {c.status.value} — {c.config.role}"
        if c.result:
            line += f" | Result: {c.result[:200]}"
        child_info.append(line)

    system_prompt = claude_client.build_agent_system_prompt(
        agent, prefs, tasks, facts, episodes,
        filtered_tools, workflows, sibling_info, child_info,
    )

    messages = list(agent.messages)
    if user_input:
        inbound = ChatMessage(role="user", content=user_input, timestamp=datetime.now(timezone.utc))
        agent.messages.append(inbound)
        messages.append(inbound)

    # Trim context window
    if len(messages) > 24:
        ctx = [messages[0]]
        ctx.append(ChatMessage(role="user", content="[earlier messages summarized — focus on current task]"))
        ctx.extend(messages[-22:])
        messages = ctx

    # Call Claude
    try:
        parsed = await claude_client.chat(
            system_prompt,
            messages,
            filtered_tools,
            agent_id=agent_id,
            allow_agent_complete=bool(agent.parent_id),
        )
    except Exception as e:
        agent.status = AgentStatus.FAILED
        agent.result = f"API error: {e}"
        raise

    step = AgentStepResult(
        agent_id=agent_id,
        response=parsed.clean_text,
        iteration=agent.iterations,
    )

    if parsed.assistant_blocks:
        agent.messages.append(ChatMessage(
            role="assistant",
            content=parsed.clean_text,
            blocks=parsed.assistant_blocks,
            timestamp=datetime.now(timezone.utc),
        ))
    elif parsed.clean_text:
        agent.messages.append(ChatMessage(
            role="assistant",
            content=parsed.clean_text,
            timestamp=datetime.now(timezone.utc),
        ))

    if parsed.tool_use_blocks:
        tool_result_blocks = []

        for block in parsed.tool_use_blocks:
            tool_name = block.get("name", "")
            tool_input = block.get("input") or {}
            tool_use_id = block.get("id", "")

            try:
                if tool_name == claude_client.CONTROL_TOOL_CREATE:
                    create = claude_client.AIToolCreate(**tool_input)
                    tool = ToolDef(
                        id=new_id(),
                        name=create.name,
                        description=create.description,
                        params=create.params,
                        builtin=False,
                        handler=create.handler,
                    )
                    await db.save_tool(tool)
                    tool_registry.register(tool)
                    step.tools_created.append(tool)
                    payload: Any = {"ok": True, "tool_id": tool.id, "name": tool.name}
                    is_error = False

                elif tool_name == claude_client.CONTROL_TOOL_WORKFLOW:
                    workflow_def = claude_client.AIWorkflowDefine(**tool_input)
                    workflow = Workflow(
                        id=new_id(),
                        name=workflow_def.name,
                        description=workflow_def.description,
                        steps=workflow_def.steps,
                        schedule=workflow_def.schedule,
                        enabled=workflow_def.schedule is not None,
                    )
                    await db.save_workflow(workflow)
                    workflow_engine.register(workflow)
                    step.workflows_created.append(workflow)
                    payload = {"ok": True, "workflow_id": workflow.id, "name": workflow.name}
                    is_error = False

                elif tool_name == claude_client.CONTROL_TOOL_MEMORY:
                    memory_update = claude_client.AIMemoryUpdate(**tool_input)
                    if memory_update.facts:
                        for fact_text in memory_update.facts:
                            await db.add_fact(fact_text)
                            step.memory_facts_added.append(fact_text)
                    if memory_update.preferences:
                        step.preferences_updated = await _apply_preference_updates(memory_update.preferences)
                    payload = {
                        "ok": True,
                        "facts_added": len(step.memory_facts_added),
                        "preferences_updated": bool(step.preferences_updated),
                    }
                    is_error = False

                elif tool_name == claude_client.CONTROL_TOOL_SPAWN:
                    spawn = claude_client.AIAgentSpawn(**tool_input)
                    config = AgentConfig(
                        role=spawn.role,
                        goal=spawn.goal,
                        tools=spawn.tools or [],
                        max_iterations=spawn.max_iterations or 20,
                        timeout_secs=spawn.timeout_secs or 300,
                        auto_terminate=spawn.auto_terminate if spawn.auto_terminate is not None else True,
                        report_to=agent_id,
                    )
                    child_id = registry.spawn_agent(spawn.name, config, agent_id)
                    if spawn.initial_task:
                        registry.send_message(agent_id, child_id, spawn.initial_task, AgentMsgType.TASK)
                    asyncio.create_task(
                        run_agent_loop(registry, tool_registry, workflow_engine, child_id, spawn.initial_task)
                    )
                    step.agents_spawned.append(child_id)
                    agent.status = AgentStatus.WAITING_SUBAGENT
                    payload = {"ok": True, "agent_id": child_id}
                    is_error = False

                elif tool_name == claude_client.CONTROL_TOOL_MESSAGE:
                    message = claude_client.AIAgentMessage(**tool_input)
                    mt_map = {"task": AgentMsgType.TASK, "result": AgentMsgType.RESULT,
                              "query": AgentMsgType.QUERY, "response": AgentMsgType.RESPONSE,
                              "directive": AgentMsgType.DIRECTIVE, "alert": AgentMsgType.ALERT}
                    mt = mt_map.get(message.msg_type or "", AgentMsgType.STATUS)
                    registry.send_message(agent_id, message.to_agent, message.content, mt)
                    step.agents_messaged.append(message.to_agent)
                    payload = {"ok": True, "to_agent": message.to_agent}
                    is_error = False

                elif tool_name == claude_client.CONTROL_TOOL_COMPLETE:
                    completion = claude_client.AIAgentComplete(**tool_input)
                    agent.status = AgentStatus.COMPLETED
                    agent.result = completion.result
                    agent.completed_at = datetime.now(timezone.utc)
                    agent.updated_at = datetime.now(timezone.utc)
                    if agent.config.report_to:
                        registry.send_message(
                            agent_id,
                            agent.config.report_to,
                            f"Sub-agent '{agent.name}' completed. Result: {completion.result}",
                            AgentMsgType.RESULT,
                        )
                    step.status_change = "completed"
                    payload = {"ok": True}
                    is_error = False

                else:
                    result = await tool_registry.execute(tool_name, tool_input)
                    log = ExecLogEntry(
                        id=new_id(),
                        source="agent_tool",
                        source_name=f"{agent_id}:{tool_name}",
                        input_summary=json.dumps(tool_input)[:200],
                        result_summary=str(result.result)[:200] if result.result else "",
                        success=result.success,
                        execution_ms=result.execution_ms,
                    )
                    await db.add_log_entry(log)
                    step.tool_calls.append(ToolCallResult(tool=tool_name, params=tool_input, result=result))
                    payload = result.result if result.success else {"error": result.error or "Tool execution failed"}
                    is_error = not result.success

            except Exception as exc:
                payload = {"error": str(exc)}
                is_error = True

            if tool_use_id:
                tool_result_blocks.append(_tool_result_block(tool_use_id, payload, is_error=is_error))

        if tool_result_blocks:
            step.needs_model_followup = True
            agent.messages.append(ChatMessage(
                role="user",
                content="Tool results provided.",
                blocks=tool_result_blocks,
                timestamp=datetime.now(timezone.utc),
            ))

        return step

    # ── Handle agent spawning ──
    for spawn in parsed.agent_spawns:
        config = AgentConfig(
            role=spawn.role, goal=spawn.goal,
            tools=spawn.tools or [],
            max_iterations=spawn.max_iterations or 20,
            timeout_secs=spawn.timeout_secs or 300,
            auto_terminate=spawn.auto_terminate if spawn.auto_terminate is not None else True,
            report_to=agent_id,
        )
        child_id = registry.spawn_agent(spawn.name, config, agent_id)

        if spawn.initial_task:
            registry.send_message(agent_id, child_id, spawn.initial_task, AgentMsgType.TASK)

        # Run child in background
        asyncio.create_task(
            run_agent_loop(registry, tool_registry, workflow_engine, child_id, spawn.initial_task)
        )

        step.agents_spawned.append(child_id)
        agent.status = AgentStatus.WAITING_SUBAGENT

    # ── Handle agent messages ──
    for msg in parsed.agent_messages:
        mt_map = {"task": AgentMsgType.TASK, "result": AgentMsgType.RESULT,
                  "query": AgentMsgType.QUERY, "response": AgentMsgType.RESPONSE,
                  "directive": AgentMsgType.DIRECTIVE, "alert": AgentMsgType.ALERT}
        mt = mt_map.get(msg.msg_type or "", AgentMsgType.STATUS)
        registry.send_message(agent_id, msg.to_agent, msg.content, mt)
        step.agents_messaged.append(msg.to_agent)

    # ── Handle agent completion ──
    if parsed.agent_complete:
        agent.status = AgentStatus.COMPLETED
        agent.result = parsed.agent_complete.result
        agent.completed_at = datetime.now(timezone.utc)
        agent.updated_at = datetime.now(timezone.utc)

        if agent.config.report_to:
            registry.send_message(
                agent_id, agent.config.report_to,
                f"Sub-agent '{agent.name}' completed. Result: {parsed.agent_complete.result}",
                AgentMsgType.RESULT,
            )
        step.status_change = "completed"

    # ── Handle tool calls ──
    if parsed.tool_calls:
        step.needs_model_followup = True
        for tc in parsed.tool_calls:
            result = await tool_registry.execute(tc.tool, tc.params)
            log = ExecLogEntry(
                id=new_id(), source="agent_tool", source_name=f"{agent_id}:{tc.tool}",
                input_summary=json.dumps(tc.params)[:200],
                result_summary=str(result.result)[:200] if result.result else "",
                success=result.success, execution_ms=result.execution_ms,
            )
            await db.add_log_entry(log)
            step.tool_calls.append(ToolCallResult(tool=tc.tool, params=tc.params, result=result))

        # Feed tool results back
        results_text = "\n".join(
            f"[Tool: {tc.tool}] {'OK' if tc.result.success else 'ERR'}: "
            f"{json.dumps(tc.result.result)[:500] if tc.result.result else tc.result.error or ''}"
            for tc in step.tool_calls
        )
        agent.messages.append(ChatMessage(
            role="user", content=f"Tool results:\n{results_text}\n\nContinue with your task.",
            timestamp=datetime.now(timezone.utc),
        ))

    # ── Handle tool creation ──
    for tc in parsed.tool_creates:
        tool = ToolDef(
            id=new_id(), name=tc.name, description=tc.description,
            params=tc.params, builtin=False, handler=tc.handler,
        )
        await db.save_tool(tool)
        tool_registry.register(tool)
        step.tools_created.append(tool)

    # ── Handle workflow definitions ──
    for wd in parsed.workflow_defines:
        wf = Workflow(
            id=new_id(), name=wd.name, description=wd.description,
            steps=wd.steps, schedule=wd.schedule,
            enabled=wd.schedule is not None,
        )
        await db.save_workflow(wf)
        workflow_engine.register(wf)
        step.workflows_created.append(wf)

    # ── Handle memory updates ──
    if parsed.memory_update:
        if parsed.memory_update.facts:
            for fact_text in parsed.memory_update.facts:
                await db.add_fact(fact_text)
                step.memory_facts_added.append(fact_text)
        if parsed.memory_update.preferences:
            step.preferences_updated = await _apply_preference_updates(parsed.memory_update.preferences)

    # Update agent messages with final response
    if not step.status_change and not parsed.tool_calls:
        if agent.status not in (AgentStatus.COMPLETED, AgentStatus.WAITING_SUBAGENT):
            agent.status = AgentStatus.IDLE
        agent.updated_at = datetime.now(timezone.utc)

    return step


# ═══════════════════════════════════════════
# AGENT LOOP — runs a sub-agent to completion
# ═══════════════════════════════════════════

async def run_agent_loop(
    registry: AgentRegistry,
    tool_registry,
    workflow_engine,
    agent_id: str,
    initial_input: Optional[str] = None,
) -> str:
    agent = registry.get_agent(agent_id)
    if not agent:
        return "Agent not found"

    max_iterations = agent.config.max_iterations
    timeout_secs = agent.config.timeout_secs
    deadline = datetime.now(timezone.utc) + timedelta(seconds=timeout_secs) if timeout_secs > 0 else None

    iteration = 0
    last_input = initial_input

    while True:
        iteration += 1

        if iteration > max_iterations:
            agent.status = AgentStatus.FAILED
            agent.result = "Exceeded maximum iterations"
            agent.completed_at = datetime.now(timezone.utc)
            return "Exceeded maximum iterations"

        if deadline and datetime.now(timezone.utc) > deadline:
            agent.status = AgentStatus.FAILED
            agent.result = "Timed out"
            agent.completed_at = datetime.now(timezone.utc)
            return "Timed out"

        agent = registry.get_agent(agent_id)
        if not agent:
            return "Agent removed"
        if agent.status == AgentStatus.TERMINATED:
            return "Terminated"
        if agent.status == AgentStatus.COMPLETED:
            return agent.result or ""

        try:
            result = await execute_agent_step(
                registry, tool_registry, workflow_engine, agent_id, last_input
            )
        except Exception as e:
            print(f"Agent {agent_id} step error: {e}")
            agent.status = AgentStatus.FAILED
            agent.result = str(e)
            return str(e)

        last_input = None

        if result.status_change == "completed":
            return registry.get_agent(agent_id).result or ""

        if result.needs_model_followup:
            continue

        # Wait for sub-agents
        agent = registry.get_agent(agent_id)
        if agent and agent.status == AgentStatus.WAITING_SUBAGENT:
            while True:
                await asyncio.sleep(2)
                agent = registry.get_agent(agent_id)
                if not agent:
                    return "Agent removed"

                all_done = all(
                    registry.get_agent(cid) is None or
                    registry.get_agent(cid).status in (AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.TERMINATED)
                    for cid in agent.children
                )

                if all_done:
                    child_results = "\n".join(
                        f"Agent '{c.name}' ({c.id}): {c.status.value} — Result: {c.result or 'no result'}"
                        for cid in agent.children
                        if (c := registry.get_agent(cid))
                    )
                    last_input = f"All sub-agents have completed. Results:\n{child_results}\n\nProcess these results and continue."
                    agent.status = AgentStatus.RUNNING
                    break

                if deadline and datetime.now(timezone.utc) > deadline:
                    break

            continue

        if agent and agent.status == AgentStatus.IDLE and not result.agents_spawned:
            break

    return "Agent loop ended"
