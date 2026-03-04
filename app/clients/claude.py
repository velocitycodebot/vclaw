from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from app.schemas.models import (
    AIAgentComplete,
    AIAgentMessage,
    AIAgentSpawn,
    AIMemoryUpdate,
    AIToolCall,
    AIToolCreate,
    AIWorkflowDefine,
    AgentDef,
    ChatMessage,
    Episode,
    Fact,
    ParsedAIResponse,
    Preferences,
    Task,
    ToolDef,
    ToolParam,
    Workflow,
)

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
DEFAULT_MAX_TOKENS = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "4096"))
PROMPT_CACHING_ENABLED = os.environ.get("ANTHROPIC_PROMPT_CACHING", "1") not in {"0", "false", "False"}
THINKING_ENABLED = os.environ.get("ANTHROPIC_ENABLE_THINKING", "0") in {"1", "true", "True"}
THINKING_BUDGET = int(os.environ.get("ANTHROPIC_THINKING_BUDGET_TOKENS", "2048"))

CONTROL_TOOL_CREATE = "agentos_create_tool"
CONTROL_TOOL_WORKFLOW = "agentos_define_workflow"
CONTROL_TOOL_MEMORY = "agentos_update_memory"
CONTROL_TOOL_SPAWN = "agentos_spawn_agent"
CONTROL_TOOL_MESSAGE = "agentos_message_agent"
CONTROL_TOOL_COMPLETE = "agentos_complete_agent"


def _schema_type(type_name: str) -> str:
    normalized = type_name.strip().lower()
    mapping = {
        "str": "string",
        "string": "string",
        "text": "string",
        "int": "integer",
        "integer": "integer",
        "float": "number",
        "number": "number",
        "bool": "boolean",
        "boolean": "boolean",
        "array": "array",
        "list": "array",
        "object": "object",
        "dict": "object",
        "json": "object",
    }
    return mapping.get(normalized, "string")


def _tool_param_schema(param: ToolParam) -> dict[str, Any]:
    schema_type = _schema_type(param.type)
    schema: dict[str, Any] = {
        "type": schema_type,
        "description": param.description or param.name,
    }
    if schema_type == "array":
        schema["items"] = {"type": "string"}
    elif schema_type == "object":
        schema["additionalProperties"] = True
    return schema


def _cacheable(payload: dict[str, Any]) -> dict[str, Any]:
    if not PROMPT_CACHING_ENABLED:
        return payload
    cached = dict(payload)
    cached["cache_control"] = {"type": "ephemeral"}
    return cached


def _tool_schema(tool: ToolDef) -> dict[str, Any]:
    properties = {param.name: _tool_param_schema(param) for param in tool.params}
    required = [param.name for param in tool.params if param.required is not False]
    description = tool.description
    if tool.builtin:
        description = f"{description} Use this only when it advances the user's task."
    return _cacheable({
        "name": tool.name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    })


def _control_tools(allow_agent_complete: bool) -> list[dict[str, Any]]:
    tools = [
        {
            "name": CONTROL_TOOL_CREATE,
            "description": "Create a persistent tool in the local registry for a capability that will be reused later.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Unique tool name"},
                    "description": {"type": "string", "description": "What the tool does"},
                    "params": {
                        "type": "array",
                        "description": "Tool parameters",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string"},
                                "description": {"type": "string"},
                                "required": {"type": "boolean"},
                            },
                            "required": ["name"],
                            "additionalProperties": False,
                        },
                    },
                    "handler": {"type": "string", "description": "Handler in shell:, script:, python:, or http:METHOD:URL format"},
                },
                "required": ["name", "description", "handler"],
                "additionalProperties": False,
            },
        },
        {
            "name": CONTROL_TOOL_WORKFLOW,
            "description": "Create a persistent workflow, optionally with a schedule for automation.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tool_name": {"type": "string"},
                                "params": {"type": "object", "additionalProperties": True},
                                "continue_on_error": {"type": "boolean"},
                                "delay_ms": {"type": "integer"},
                            },
                            "required": ["tool_name"],
                            "additionalProperties": False,
                        },
                    },
                    "schedule": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "description": "interval or daily"},
                            "interval_ms": {"type": "integer"},
                            "hour": {"type": "integer"},
                            "minute": {"type": "integer"},
                            "label": {"type": "string"},
                        },
                        "required": ["type"],
                        "additionalProperties": False,
                    },
                },
                "required": ["name", "steps"],
                "additionalProperties": False,
            },
        },
        {
            "name": CONTROL_TOOL_MEMORY,
            "description": "Store durable facts or update remembered user preferences.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "facts": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "preferences": {
                        "type": "object",
                        "additionalProperties": True,
                        "description": "Partial preference updates. Unknown keys go into custom preferences.",
                    },
                },
                "additionalProperties": False,
            },
        },
        {
            "name": CONTROL_TOOL_SPAWN,
            "description": "Spawn a specialized sub-agent for parallel work.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                    "goal": {"type": "string"},
                    "tools": {"type": "array", "items": {"type": "string"}},
                    "initial_task": {"type": "string"},
                    "max_iterations": {"type": "integer"},
                    "timeout_secs": {"type": "integer"},
                    "auto_terminate": {"type": "boolean"},
                },
                "required": ["name", "role", "goal"],
                "additionalProperties": False,
            },
        },
        {
            "name": CONTROL_TOOL_MESSAGE,
            "description": "Send a typed message to another agent.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "to_agent": {"type": "string"},
                    "content": {"type": "string"},
                    "msg_type": {"type": "string"},
                },
                "required": ["to_agent", "content"],
                "additionalProperties": False,
            },
        },
    ]
    if allow_agent_complete:
        tools.append({
            "name": CONTROL_TOOL_COMPLETE,
            "description": "Mark the current sub-agent as complete and return its final result.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "result": {"type": "string"},
                },
                "required": ["result"],
                "additionalProperties": False,
            },
        })
    return [_cacheable(tool) for tool in tools]


def _message_content(message: ChatMessage) -> str | list[dict[str, Any]]:
    if message.blocks:
        return message.blocks
    return message.content


def _usage_dict(raw_usage: Any) -> dict[str, Any]:
    return raw_usage if isinstance(raw_usage, dict) else {}


def _internal_parse_from_text(text: str) -> ParsedAIResponse:
    clean = text
    tool_calls: list[AIToolCall] = []
    tool_creates: list[AIToolCreate] = []
    workflow_defines: list[AIWorkflowDefine] = []
    memory_update = None
    agent_spawns: list[AIAgentSpawn] = []
    agent_messages: list[AIAgentMessage] = []
    agent_complete = None

    def extract(tag: str, model_cls, target: Optional[list] = None):
        nonlocal clean
        pattern = rf"```{tag}\s*([\s\S]*?)```"
        for match in re.finditer(pattern, text):
            try:
                parsed = json.loads(match.group(1).strip())
                obj = model_cls(**parsed)
                if target is not None:
                    target.append(obj)
                else:
                    return obj
            except Exception:
                pass
        clean = re.sub(pattern, "", clean)
        return None

    extract("tool_call", AIToolCall, tool_calls)
    extract("tool_create", AIToolCreate, tool_creates)
    extract("workflow_define", AIWorkflowDefine, workflow_defines)

    memory_pattern = r"```memory_update\s*([\s\S]*?)```"
    for match in re.finditer(memory_pattern, text):
        try:
            memory_update = AIMemoryUpdate(**json.loads(match.group(1).strip()))
        except Exception:
            pass
    clean = re.sub(memory_pattern, "", clean)

    extract("spawn_agent", AIAgentSpawn, agent_spawns)
    extract("agent_message", AIAgentMessage, agent_messages)

    complete_pattern = r"```agent_complete\s*([\s\S]*?)```"
    for match in re.finditer(complete_pattern, text):
        try:
            agent_complete = AIAgentComplete(**json.loads(match.group(1).strip()))
        except Exception:
            pass
    clean = re.sub(complete_pattern, "", clean)

    return ParsedAIResponse(
        clean_text=clean.strip(),
        tool_calls=tool_calls,
        tool_creates=tool_creates,
        workflow_defines=workflow_defines,
        memory_update=memory_update,
        agent_spawns=agent_spawns,
        agent_messages=agent_messages,
        agent_complete=agent_complete,
    )


async def chat(
    system: str | list[dict[str, Any]],
    messages: list[ChatMessage],
    tools: list[ToolDef],
    *,
    agent_id: str,
    allow_agent_complete: bool = False,
) -> ParsedAIResponse:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    msgs = [
        {"role": m.role, "content": _message_content(m)}
        for m in messages
        if m.role in ("user", "assistant")
    ]

    request_tools = [_tool_schema(tool) for tool in tools]
    request_tools.extend(_control_tools(allow_agent_complete))

    body: dict[str, Any] = {
        "model": DEFAULT_MODEL,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "system": system,
        "messages": msgs,
        "metadata": {"user_id": agent_id[:256]},
    }
    if request_tools:
        body["tools"] = request_tools
        body["tool_choice"] = {"type": "auto"}
    if THINKING_ENABLED:
        body["thinking"] = {
            "type": "enabled",
            "budget_tokens": max(1024, THINKING_BUDGET),
        }
    service_tier = os.environ.get("ANTHROPIC_SERVICE_TIER")
    if service_tier:
        body["service_tier"] = service_tier

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            API_URL,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": API_VERSION,
            },
            json=body,
        )

    data = resp.json()
    if resp.is_error:
        err = data.get("error", {}).get("message", resp.text)
        raise RuntimeError(f"Claude API: {err}")

    content = data.get("content", [])
    text_blocks = [block.get("text", "") for block in content if block.get("type") == "text"]
    tool_use_blocks = [block for block in content if block.get("type") == "tool_use"]
    text = "".join(text_blocks).strip()

    parsed = ParsedAIResponse(
        clean_text=text,
        assistant_blocks=content,
        tool_use_blocks=tool_use_blocks,
        stop_reason=data.get("stop_reason"),
        usage=_usage_dict(data.get("usage")),
    )

    for block in tool_use_blocks:
        name = block.get("name")
        tool_input = block.get("input") or {}
        try:
            if name == CONTROL_TOOL_CREATE:
                parsed.tool_creates.append(AIToolCreate(**tool_input))
            elif name == CONTROL_TOOL_WORKFLOW:
                parsed.workflow_defines.append(AIWorkflowDefine(**tool_input))
            elif name == CONTROL_TOOL_MEMORY:
                parsed.memory_update = AIMemoryUpdate(**tool_input)
            elif name == CONTROL_TOOL_SPAWN:
                parsed.agent_spawns.append(AIAgentSpawn(**tool_input))
            elif name == CONTROL_TOOL_MESSAGE:
                parsed.agent_messages.append(AIAgentMessage(**tool_input))
            elif name == CONTROL_TOOL_COMPLETE:
                parsed.agent_complete = AIAgentComplete(**tool_input)
            else:
                parsed.tool_calls.append(AIToolCall(tool=name or "", params=tool_input))
        except Exception:
            # Keep the raw tool_use block for execution even if validation failed.
            continue

    if not tool_use_blocks and "```" in text:
        fallback = _internal_parse_from_text(text)
        parsed.clean_text = fallback.clean_text
        parsed.tool_calls = fallback.tool_calls
        parsed.tool_creates = fallback.tool_creates
        parsed.workflow_defines = fallback.workflow_defines
        parsed.memory_update = fallback.memory_update
        parsed.agent_spawns = fallback.agent_spawns
        parsed.agent_messages = fallback.agent_messages
        parsed.agent_complete = fallback.agent_complete
        parsed.assistant_blocks = [{"type": "text", "text": parsed.clean_text}] if parsed.clean_text else []

    return parsed


def build_system_prompt(
    prefs: Preferences,
    tasks: list[Task],
    facts: list[Fact],
    episodes: list[Episode],
    tools: list[ToolDef],
    workflows: list[Workflow],
) -> list[dict[str, Any]]:
    active_tasks = [t for t in tasks if t.status != "done"]
    instructions = """You are AgentOS, an autonomous assistant running on a local Python/FastAPI backend.

Use Claude tool calling directly. Do not wrap actions in markdown code fences or pseudo-JSON blocks.
Call the provided tools when you need to act, inspect the system, create tools, define workflows, update memory, or coordinate sub-agents.
Never invent tool results. After tool results arrive, continue from them and answer naturally."""

    runtime_context = "\n".join([
        f"Today: {datetime.now(timezone.utc):%A, %B %d, %Y %H:%M UTC}",
        "",
        "## USER PROFILE",
        f"Name: {prefs.name or 'Not set'}",
        f"Tone: {prefs.tone}",
        f"Interests: {', '.join(prefs.topics) if prefs.topics else 'None'}",
        "",
        f"## ACTIVE TASKS ({len(active_tasks)})",
        "\n".join(
            f"- [{t.priority}] \"{t.title}\" - {t.status}{f' ({t.progress}%)' if t.progress > 0 else ''}"
            for t in active_tasks
        ) or "None",
        "",
        f"## REMEMBERED FACTS ({len(facts)})",
        "\n".join(f"- {f.text}" for f in facts[:30]) or "None yet",
        "",
        "## RECENT EPISODES",
        "\n".join(f"- [{e.created_at:%Y-%m-%d}] {e.summary}" for e in episodes[:10]) or "None yet",
        "",
        f"## WORKFLOWS ({len(workflows)})",
        "\n".join(
            f"- {w.name}{f' [SCHED: {w.schedule.label}]' if w.schedule else ''}: {w.description or ''} ({len(w.steps)} steps)"
            for w in workflows
        ) or "None",
        "",
        f"## TOOLS ({len(tools)})",
        ", ".join(tool.name for tool in tools) or "None",
    ])

    return [
        _cacheable({"type": "text", "text": instructions}),
        {"type": "text", "text": runtime_context},
    ]


def build_agent_system_prompt(
    agent: AgentDef,
    prefs: Preferences,
    tasks: list[Task],
    facts: list[Fact],
    episodes: list[Episode],
    tools: list[ToolDef],
    workflows: list[Workflow],
    sibling_info: list[str],
    child_info: list[str],
) -> list[dict[str, Any]]:
    is_main = agent.id == "main"
    instructions = "\n".join([
        f'You are "{agent.name}" - {agent.config.role}.',
        "Use Claude tool calling directly. Do not emit tool instructions in markdown fences.",
        "Use the provided tools for real actions, use agent coordination tools when work should be parallelized, and never fabricate outputs.",
        "Keep responses concise and operational. When there is nothing left to do, answer clearly.",
        "If you are a sub-agent, finish by calling the completion tool with the result.",
    ])

    sections = [
        f"Your ID: {agent.id}",
        f"Parent agent: {agent.parent_id}" if agent.parent_id else "You are the MAIN orchestrator agent.",
        f"Today: {datetime.now(timezone.utc):%A, %B %d, %Y %H:%M UTC}",
        "",
        "## YOUR GOAL",
        agent.config.goal,
        "",
        f"## AVAILABLE TOOLS ({len(tools)})",
        ", ".join(tool.name for tool in tools) or "None",
        "",
        f"## WORKFLOWS ({len(workflows)})",
        "\n".join(
            f"- {w.name}{f' [SCHED: {w.schedule.label}]' if w.schedule else ''}: {w.description or ''}"
            for w in workflows
        ) or "None",
    ]

    if child_info:
        sections.extend(["", "## YOUR SUB-AGENTS", "\n".join(child_info)])
    if sibling_info:
        sections.extend(["", "## SIBLING AGENTS", "\n".join(sibling_info)])

    if is_main:
        active_tasks = [t for t in tasks if t.status != "done"]
        sections.extend([
            "",
            "## USER PROFILE",
            f"Name: {prefs.name or 'Not set'}",
            f"Tone: {prefs.tone}",
            f"Interests: {', '.join(prefs.topics) if prefs.topics else 'None'}",
            "",
            f"## ACTIVE TASKS ({len(active_tasks)})",
            "\n".join(f"- [{t.priority}] \"{t.title}\" - {t.status}" for t in active_tasks) or "None",
            "",
            f"## REMEMBERED FACTS ({len(facts)})",
            "\n".join(f"- {f.text}" for f in facts[:20]) or "None yet",
            "",
            "## RECENT EPISODES",
            "\n".join(f"- [{e.created_at:%Y-%m-%d}] {e.summary}" for e in episodes[:8]) or "None yet",
            "",
            "## ORCHESTRATION STRATEGY",
            "Handle simple tasks directly. Spawn sub-agents only for parallel or specialized work. Synthesize child results before answering the user.",
        ])
    else:
        sections.extend([
            "",
            "## FOCUS",
            f'Stay focused on: "{agent.config.goal}"',
            "Work efficiently, use tools when needed, and complete when the goal is met.",
        ])

    sections.extend(["", f"Iterations: {agent.iterations}/{agent.config.max_iterations}"])

    return [
        _cacheable({"type": "text", "text": instructions}),
        {"type": "text", "text": "\n".join(sections)},
    ]


def parse_ai_response(text: str) -> ParsedAIResponse:
    return _internal_parse_from_text(text)
