from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime, timezone
from enum import Enum
import uuid


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:8]}" if prefix else uuid.uuid4().hex[:12]


# ── Preferences ──
class Preferences(BaseModel):
    name: str = ""
    tone: str = "friendly"
    topics: list[str] = []
    custom: dict[str, Any] = {}


# ── Memory ──
class Fact(BaseModel):
    id: str = ""
    text: str
    category: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Episode(BaseModel):
    id: str = ""
    summary: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Tasks ──
class Task(BaseModel):
    id: str = ""
    title: str
    notes: Optional[str] = None
    status: str = "todo"
    priority: str = "medium"
    progress: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CreateTask(BaseModel):
    title: str
    notes: Optional[str] = None
    priority: Optional[str] = "medium"


class UpdateTask(BaseModel):
    title: Optional[str] = None
    notes: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None
    progress: Optional[int] = None


# ── Tools ──
class ToolParam(BaseModel):
    name: str
    type: str = "string"
    description: str = ""
    required: Optional[bool] = True


class ToolDef(BaseModel):
    id: str = ""
    name: str
    description: str
    params: list[ToolParam] = []
    builtin: bool = False
    handler: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ToolResult(BaseModel):
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_ms: int = 0


class ToolExecRequest(BaseModel):
    params: dict[str, Any] = {}


class CreateToolRequest(BaseModel):
    name: str
    description: str
    params: list[ToolParam] = []
    handler: str


# ── Shell ──
class ShellExecRequest(BaseModel):
    command: str
    working_dir: Optional[str] = None
    timeout_secs: Optional[int] = None
    stdin: Optional[str] = None


class ShellResult(BaseModel):
    id: str = ""
    command: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    execution_ms: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Workflows ──
class WorkflowStep(BaseModel):
    tool_name: str
    params: dict[str, Any] = {}
    continue_on_error: Optional[bool] = False
    delay_ms: Optional[int] = None


class WorkflowSchedule(BaseModel):
    type: str  # interval | daily
    interval_ms: Optional[int] = None
    hour: Optional[int] = None
    minute: Optional[int] = None
    label: str = ""


class Workflow(BaseModel):
    id: str = ""
    name: str
    description: Optional[str] = None
    steps: list[WorkflowStep] = []
    schedule: Optional[WorkflowSchedule] = None
    enabled: bool = False
    last_run: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CreateWorkflow(BaseModel):
    name: str
    description: Optional[str] = None
    steps: list[WorkflowStep] = []
    schedule: Optional[WorkflowSchedule] = None


# ── Execution Log ──
class ExecLogEntry(BaseModel):
    id: str = ""
    source: str = ""
    source_name: str = ""
    input_summary: str = ""
    result_summary: str = ""
    success: bool = False
    execution_ms: int = 0
    scheduled: bool = False
    full_result: Optional[Any] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Chat ──
class ChatMessage(BaseModel):
    role: str
    content: str = ""
    blocks: Optional[list[dict[str, Any]]] = None
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None


# ── Agents ──
class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_INPUT = "waiting_for_input"
    WAITING_SUBAGENT = "waiting_for_subagent"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


class AgentConfig(BaseModel):
    role: str
    goal: str
    tools: list[str] = []
    max_iterations: int = 20
    timeout_secs: int = 300
    auto_terminate: bool = True
    report_to: Optional[str] = None


class AgentMsgType(str, Enum):
    TASK = "task"
    RESULT = "result"
    STATUS = "status"
    QUERY = "query"
    RESPONSE = "response"
    DIRECTIVE = "directive"
    ALERT = "alert"


class AgentMessage(BaseModel):
    id: str = ""
    from_agent: str
    to_agent: str
    content: str
    msg_type: AgentMsgType = AgentMsgType.STATUS
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentDef(BaseModel):
    id: str = ""
    name: str
    config: AgentConfig
    status: AgentStatus = AgentStatus.IDLE
    parent_id: Optional[str] = None
    children: list[str] = []
    messages: list[ChatMessage] = []
    result: Optional[str] = None
    iterations: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None


class SpawnAgentRequest(BaseModel):
    name: str
    role: str
    goal: str
    tools: Optional[list[str]] = None
    initial_task: Optional[str] = None
    max_iterations: Optional[int] = 20
    timeout_secs: Optional[int] = 300
    parent_id: Optional[str] = "main"


class AgentMessageRequest(BaseModel):
    content: str
    msg_type: Optional[str] = "status"


# ── AI Response Parsing ──
class AIToolCall(BaseModel):
    tool: str
    params: dict[str, Any] = {}


class AIToolCreate(BaseModel):
    name: str
    description: str
    params: list[ToolParam] = []
    handler: str


class AIWorkflowDefine(BaseModel):
    name: str
    description: Optional[str] = None
    steps: list[WorkflowStep] = []
    schedule: Optional[WorkflowSchedule] = None


class AIMemoryUpdate(BaseModel):
    facts: Optional[list[str]] = None
    preferences: Optional[dict[str, Any]] = None


class AIAgentSpawn(BaseModel):
    name: str
    role: str
    goal: str
    tools: Optional[list[str]] = None
    initial_task: Optional[str] = None
    max_iterations: Optional[int] = 20
    timeout_secs: Optional[int] = 300
    auto_terminate: Optional[bool] = True


class AIAgentMessage(BaseModel):
    to_agent: str
    content: str
    msg_type: Optional[str] = "status"


class AIAgentComplete(BaseModel):
    result: str


class ParsedAIResponse(BaseModel):
    clean_text: str = ""
    tool_calls: list[AIToolCall] = []
    tool_creates: list[AIToolCreate] = []
    workflow_defines: list[AIWorkflowDefine] = []
    memory_update: Optional[AIMemoryUpdate] = None
    agent_spawns: list[AIAgentSpawn] = []
    agent_messages: list[AIAgentMessage] = []
    agent_complete: Optional[AIAgentComplete] = None
    assistant_blocks: list[dict[str, Any]] = []
    tool_use_blocks: list[dict[str, Any]] = []
    stop_reason: Optional[str] = None
    usage: dict[str, Any] = {}


class ToolCallResult(BaseModel):
    tool: str
    params: dict[str, Any] = {}
    result: ToolResult


class AgentStepResult(BaseModel):
    agent_id: str
    response: str = ""
    tool_calls: list[ToolCallResult] = []
    agents_spawned: list[str] = []
    agents_messaged: list[str] = []
    tools_created: list[ToolDef] = []
    workflows_created: list[Workflow] = []
    memory_facts_added: list[str] = []
    preferences_updated: Optional[dict[str, Any]] = None
    needs_model_followup: bool = False
    status_change: Optional[str] = None
    iteration: int = 0


class SystemStatus(BaseModel):
    total_tools: int = 0
    builtin_tools: int = 0
    custom_tools: int = 0
    total_workflows: int = 0
    active_workflows: int = 0
    total_facts: int = 0
    total_episodes: int = 0
    total_tasks: int = 0
    active_tasks: int = 0
    total_logs: int = 0
    total_shell_commands: int = 0
    total_agents: int = 0
    active_agents: int = 0
