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
- Saves the config locally in `data/runtime_config.json`
- Starts the service immediately after onboarding finishes

`vclaw start`
- Validates that the required startup config is present
- Loads the saved API key into the process environment
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
Chain tools into scheduled pipelines. Background scheduler checks every 30s for due workflows.

### Persistent Memory
Facts, episodes, preferences, tasks, tools, workflows, conversations all stored in `data/agent_os.json`.

### Agentic Loop
Single user message can trigger multiple tool executions across multiple iterations (up to 7 continuation rounds).

### Claude Integration
- Native Claude Messages API tool use for built-in tools, custom tools, workflows, memory updates, and agent coordination
- Prompt caching on stable system/tool definitions to reduce repeated prompt cost
- Optional extended thinking via `ANTHROPIC_ENABLE_THINKING=1`
- Configurable model and service tier via `ANTHROPIC_MODEL`, `ANTHROPIC_MAX_TOKENS`, and `ANTHROPIC_SERVICE_TIER`

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
| GET/PUT | /api/preferences | User preferences |
| GET/POST/PUT/DELETE | /api/tasks | Task CRUD |
| GET/DELETE | /api/logs | Execution log |
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
