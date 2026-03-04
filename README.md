# AgentOS — Multi-Agent Orchestrator

A self-improving AI assistant with kernel access, sub-agent spawning, tool creation, workflow automation, and persistent memory. Python/FastAPI backend + React frontend.

## Quick Start

```bash
cd /Users/johnwalker/Code/vclaw
uv run vclaw onboard
```

Then open **http://localhost:3000**

If you already saved your startup settings, run:

```bash
cd /Users/johnwalker/Code/vclaw
uv run vclaw start
```

## CLI

`vclaw onboard`
- Prompts for the Anthropic API key and startup host/port
- Prompts for the Anthropic model and saves `claude-sonnet-4-6` by default
- Saves the config locally in `data/runtime_config.json`
- Starts the service immediately after onboarding finishes

`vclaw start`
- Validates that the required startup config is present
- Loads the saved API key into the process environment
- Loads the saved Anthropic model into the process environment
- Starts the FastAPI service with the saved host/port

## Architecture

```
vclaw/
├── main.py                  # Compatibility entrypoint for uvicorn
├── app/
│   ├── api/
│   │   └── app.py           # FastAPI app and REST endpoints
│   ├── clients/
│   │   └── claude.py        # Claude Messages API client, native tools, and prompt shaping
│   ├── schemas/
│   │   └── models.py        # Pydantic data models
│   ├── services/
│   │   ├── agents.py        # Multi-agent orchestration engine
│   │   ├── shell.py         # Kernel command executor + safety filters
│   │   ├── tools.py         # Tool registry (13 built-ins + user tools)
│   │   └── workflows.py     # Workflow engine + background scheduler
│   └── storage/
│       └── database.py      # Local JSON persistence
└── frontend/
    └── index.html           # React SPA (CDN-loaded)
```

## Multi-Agent System

The main **Orchestrator** agent receives user messages. For complex tasks, it spawns specialized sub-agents that run in parallel as async background tasks.

### Agent Hierarchy
- **Orchestrator (main)** — routes user messages, decides to handle directly or decompose
- **Sub-agents** — spawned with role, goal, tool whitelist, iteration/timeout limits
- **Nested agents** — any agent can spawn children, creating arbitrary-depth trees

### How Spawning Works
1. User sends a complex task via chat
2. Orchestrator emits `spawn_agent` blocks in its response
3. Each sub-agent gets its own Claude API conversation, native tool access, and budget
4. Sub-agents run as background `asyncio.create_task()` coroutines
5. Parent waits, then collects results when all children complete
6. Parent synthesizes a unified response

### Agent Communication
- `spawn_agent` — create child agent with custom config
- `agent_message` — send typed messages between agents (task, result, query, directive, alert)
- `agent_complete` — sub-agent reports results back to parent

## Features

### Kernel Access
The `shell` tool gives the AI direct OS command execution with safety filters blocking destructive patterns.

### Self-Improvement
AI creates persistent tools stored in a local JSON file. Handler formats: `shell:`, `script:`, `python:`, `http:`.

### Workflow Automation
Chain tools into scheduled pipelines. Background scheduler checks every 30s for due workflows and now supports `interval`, `daily`, and `weekly` schedules with optional timezone-aware execution.

### Persistent Memory
Facts, episodes, preferences, tasks, tools, workflows, conversations all stored in `data/agent_os.json`.

### Task Orchestration
Tasks now carry progress, blockers, assignee, parent/dependency links, due dates, review cadence, and outcome fields so the assistant can keep work state over time and revisit it automatically.

### Context Management
The backend ranks relevant tasks, facts, workflows, and episodes per request and compresses older conversation history before sending context to Claude, reducing topic overload.

### Semantic Recall
The backend now maintains a vector-style memory index over preferences, facts, tasks, workflows, episodes, and conversation history, then uses semantic retrieval to surface relevant past context before each Claude call.

### Controlled Self-Editing
Source edits can now run through a guarded pipeline that snapshots touched files, applies writes, runs optional eval/test commands, records a session, and rolls changes back automatically on failure.

### Agentic Loop
Single user message can trigger multiple tool executions across multiple iterations (up to 7 continuation rounds).

### Active Supervision
The main orchestrator now has a background supervisor that resumes waiting parent agents when children finish and triggers scheduled task reviews for stale or recurring tasks.

### Claude Integration
- Native Claude Messages API tool use for built-in tools, custom tools, workflows, memory updates, and agent coordination
- Prompt caching on stable system/tool definitions to reduce repeated prompt cost
- Optional extended thinking via `ANTHROPIC_ENABLE_THINKING=1`
- Default model is `claude-sonnet-4-6`, with `claude-opus-4-6` available via saved config, CLI flags, or `ANTHROPIC_MODEL`
- Configurable token budget and service tier via `ANTHROPIC_MAX_TOKENS` and `ANTHROPIC_SERVICE_TIER`

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/chat | Agentic chat loop |
| POST | /api/shell/execute | Run kernel command |
| GET | /api/shell/history | Command history |
| GET/POST/DELETE | /api/tools | Tool CRUD |
| POST | /api/tools/:id/execute | Execute tool |
| GET/POST/DELETE | /api/workflows | Workflow CRUD |
| POST | /api/workflows/:id/run | Manual trigger |
| POST | /api/workflows/:id/toggle | Enable/disable schedule |
| GET/POST/DELETE | /api/memory/facts | Facts CRUD |
| GET | /api/memory/episodes | Episode history |
| POST | /api/memory/search | Semantic memory search |
| GET | /api/memory/vector-status | Vector index status |
| GET/PUT | /api/preferences | User preferences |
| GET/POST/PUT/DELETE | /api/tasks | Task CRUD |
| GET/DELETE | /api/logs | Execution log |
| POST | /api/self-edit/run | Run guarded self-edit session |
| GET | /api/self-edit/sessions | List self-edit sessions |
| GET | /api/self-edit/sessions/:id | Self-edit session detail |
| POST | /api/self-edit/sessions/:id/rollback | Roll back a session |
| GET | /api/agents | List all agents |
| GET | /api/agents/tree | Agent hierarchy tree |
| POST | /api/agents/spawn | Spawn new agent |
| GET | /api/agents/:id | Agent detail |
| POST | /api/agents/:id/message | Send message to agent |
| POST | /api/agents/:id/terminate | Terminate agent + children |
| GET | /api/agents/:id/messages | Inter-agent message log |
| GET | /api/status | System statistics |
| POST | /api/reset | Reset all data |

## Frontend (React)

8-tab interface:
- **Chat** — conversation with tool results and spawn notifications
- **Agents** — tree visualization, spawn form, agent detail, IPC log, real-time polling
- **Shell** — direct kernel command interface
- **Tools** — registry showing built-in + custom tools
- **Workflows** — list with run/pause/enable controls
- **Logs** — execution history
- **Memory** — facts and episodes
- **Settings** — preferences, statistics, reset

## Dependencies

- **fastapi** — web framework
- **uvicorn** — ASGI server
- **httpx** — async HTTP client (Claude API)
- **pydantic** — data validation
