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
    topics: list[str] = Field(default_factory=list)
    custom: dict[str, Any] = Field(default_factory=dict)


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
    assignee: Optional[str] = None
    parent_task_id: Optional[str] = None
    depends_on: list[str] = Field(default_factory=list)
    blocked_by: list[str] = Field(default_factory=list)
    due_at: Optional[datetime] = None
    next_review_at: Optional[datetime] = None
    review_interval_minutes: Optional[int] = None
    last_activity_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    outcome: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CreateTask(BaseModel):
    title: str
    notes: Optional[str] = None
    priority: Optional[str] = "medium"
    status: Optional[str] = "todo"
    progress: Optional[int] = 0
    assignee: Optional[str] = None
    parent_task_id: Optional[str] = None
    depends_on: Optional[list[str]] = None
    blocked_by: Optional[list[str]] = None
    due_at: Optional[datetime] = None
    next_review_at: Optional[datetime] = None
    review_interval_minutes: Optional[int] = None
    outcome: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class UpdateTask(BaseModel):
    title: Optional[str] = None
    notes: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None
    progress: Optional[int] = None
    assignee: Optional[str] = None
    parent_task_id: Optional[str] = None
    depends_on: Optional[list[str]] = None
    blocked_by: Optional[list[str]] = None
    due_at: Optional[datetime] = None
    next_review_at: Optional[datetime] = None
    review_interval_minutes: Optional[int] = None
    outcome: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


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
    params: list[ToolParam] = Field(default_factory=list)
    builtin: bool = False
    handler: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ToolResult(BaseModel):
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_ms: int = 0


class ToolExecRequest(BaseModel):
    params: dict[str, Any] = Field(default_factory=dict)


class CreateToolRequest(BaseModel):
    name: str
    description: str
    params: list[ToolParam] = Field(default_factory=list)
    handler: str


class VectorMemorySearchHit(BaseModel):
    source_type: str
    source_id: str
    text: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorMemoryStatus(BaseModel):
    provider: str = "uninitialized"
    dimensions: int = 0
    documents: int = 0
    degraded: bool = False
    dirty: bool = True
    last_rebuilt_at: Optional[datetime] = None


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


class SelfEditWrite(BaseModel):
    path: str
    content: str


class SelfEditRunRequest(BaseModel):
    workspace: Optional[str] = None
    writes: list[SelfEditWrite] = Field(default_factory=list)
    eval_command: Optional[str] = None
    test_command: Optional[str] = None
    rollback_on_failure: bool = True
    notes: Optional[str] = None


class SelfEditSession(BaseModel):
    id: str = ""
    workspace: str
    writes: list[SelfEditWrite] = Field(default_factory=list)
    eval_command: Optional[str] = None
    test_command: Optional[str] = None
    rollback_on_failure: bool = True
    status: str = "pending"
    changed_files: list[str] = Field(default_factory=list)
    created_files: list[str] = Field(default_factory=list)
    snapshot_dir: str = ""
    eval_result: Optional[ShellResult] = None
    test_result: Optional[ShellResult] = None
    rolled_back: bool = False
    notes: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Workflows ──
class WorkflowStep(BaseModel):
    tool_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    continue_on_error: Optional[bool] = False
    delay_ms: Optional[int] = None


class WorkflowSchedule(BaseModel):
    type: str  # interval | daily | weekly
    interval_ms: Optional[int] = None
    hour: Optional[int] = None
    minute: Optional[int] = None
    days_of_week: list[int] = Field(default_factory=list)
    timezone: Optional[str] = None
    label: str = ""


class Workflow(BaseModel):
    id: str = ""
    name: str
    description: Optional[str] = None
    steps: list[WorkflowStep] = Field(default_factory=list)
    schedule: Optional[WorkflowSchedule] = None
    enabled: bool = False
    last_run: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CreateWorkflow(BaseModel):
    name: str
    description: Optional[str] = None
    steps: list[WorkflowStep] = Field(default_factory=list)
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
    session_id: Optional[str] = None
    session_key: Optional[str] = None
    new_session: bool = False
    reset_session: bool = False
    dm_scope: Optional[str] = None
    peer_id: Optional[str] = None
    channel_id: Optional[str] = None
    account_id: Optional[str] = None
    main_key: Optional[str] = None
    idle_minutes: Optional[int] = None


class MemorySearchRequest(BaseModel):
    query: str
    limit: int = 8
    source_types: Optional[list[str]] = None


class SessionScope(str, Enum):
    MAIN = "main"
    PER_PEER = "per-peer"
    PER_CHANNEL_PEER = "per-channel-peer"
    PER_ACCOUNT_CHANNEL_PEER = "per-account-channel-peer"


class SessionRecord(BaseModel):
    id: str = ""
    agent_id: str = "main"
    session_key: str
    dm_scope: SessionScope = SessionScope.MAIN
    main_key: str = "main"
    peer_id: Optional[str] = None
    channel_id: Optional[str] = None
    account_id: Optional[str] = None
    idle_minutes: Optional[int] = None
    transcript_path: str = ""
    message_count: int = 0
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    reset_reason: Optional[str] = None
    previous_session_id: Optional[str] = None
    archived_at: Optional[datetime] = None


class SessionResetRequest(BaseModel):
    session_id: Optional[str] = None
    session_key: Optional[str] = None
    reason: Optional[str] = None


class MemoryFileInfo(BaseModel):
    name: str
    kind: str
    path: str
    size: int = 0
    updated_at: Optional[datetime] = None


class MemoryFileContent(BaseModel):
    name: str
    kind: str
    path: str
    content: str
    updated_at: Optional[datetime] = None


class MemoryWorkspaceSearchRequest(BaseModel):
    query: str
    limit: int = 8
    include_daily: bool = True
    include_long_term: bool = True


class MemoryWorkspaceSearchHit(BaseModel):
    file_name: str
    path: str
    kind: str
    title: str
    snippet: str
    score: float = 0.0
    line_start: int = 1
    line_end: int = 1
    updated_at: Optional[datetime] = None


class MemoryWriteRequest(BaseModel):
    long_term_notes: list[str] = Field(default_factory=list)
    daily_notes: list[str] = Field(default_factory=list)
    date: Optional[str] = None


class MemoryFileUpdateRequest(BaseModel):
    content: str


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
    tools: list[str] = Field(default_factory=list)
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
    children: list[str] = Field(default_factory=list)
    messages: list[ChatMessage] = Field(default_factory=list)
    result: Optional[str] = None
    iterations: int = 0
    last_children_digest: str = ""
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
    params: dict[str, Any] = Field(default_factory=dict)


class AIToolCreate(BaseModel):
    name: str
    description: str
    params: list[ToolParam] = Field(default_factory=list)
    handler: str


class AIWorkflowDefine(BaseModel):
    name: str
    description: Optional[str] = None
    steps: list[WorkflowStep] = Field(default_factory=list)
    schedule: Optional[WorkflowSchedule] = None


class AIMemoryUpdate(BaseModel):
    facts: Optional[list[str]] = None
    long_term_notes: Optional[list[str]] = None
    daily_notes: Optional[list[str]] = None
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


class AITaskOperation(BaseModel):
    action: str = "update"
    task_id: Optional[str] = None
    title: Optional[str] = None
    notes: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[str] = None
    progress: Optional[int] = None
    assignee: Optional[str] = None
    parent_task_id: Optional[str] = None
    depends_on: Optional[list[str]] = None
    blocked_by: Optional[list[str]] = None
    due_at: Optional[datetime] = None
    next_review_at: Optional[datetime] = None
    review_interval_minutes: Optional[int] = None
    outcome: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class ParsedAIResponse(BaseModel):
    clean_text: str = ""
    tool_calls: list[AIToolCall] = Field(default_factory=list)
    tool_creates: list[AIToolCreate] = Field(default_factory=list)
    workflow_defines: list[AIWorkflowDefine] = Field(default_factory=list)
    memory_update: Optional[AIMemoryUpdate] = None
    task_operations: list[AITaskOperation] = Field(default_factory=list)
    agent_spawns: list[AIAgentSpawn] = Field(default_factory=list)
    agent_messages: list[AIAgentMessage] = Field(default_factory=list)
    agent_complete: Optional[AIAgentComplete] = None
    assistant_blocks: list[dict[str, Any]] = Field(default_factory=list)
    tool_use_blocks: list[dict[str, Any]] = Field(default_factory=list)
    stop_reason: Optional[str] = None
    usage: dict[str, Any] = Field(default_factory=dict)


class ToolCallResult(BaseModel):
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)
    result: ToolResult


class AgentStepResult(BaseModel):
    agent_id: str
    response: str = ""
    tool_calls: list[ToolCallResult] = Field(default_factory=list)
    agents_spawned: list[str] = Field(default_factory=list)
    agents_messaged: list[str] = Field(default_factory=list)
    tools_created: list[ToolDef] = Field(default_factory=list)
    workflows_created: list[Workflow] = Field(default_factory=list)
    tasks_changed: list[Task] = Field(default_factory=list)
    memory_facts_added: list[str] = Field(default_factory=list)
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
