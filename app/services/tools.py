import asyncio
import json
import time
import os
import httpx
from datetime import datetime
from pathlib import Path
from app.schemas.models import ToolDef, ToolParam, ToolResult, new_id
from app.services.shell import execute_command
from app.schemas.models import SelfEditRunRequest, SelfEditWrite
from app.services.self_edit import self_edit_service
from app.services.vector_memory import vector_memory

from typing import Optional


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolDef] = {}

    def register(self, tool: ToolDef):
        self._tools[tool.name] = tool

    def remove(self, name: str):
        self._tools.pop(name, None)

    def get(self, name: str) -> Optional[ToolDef]:
        return self._tools.get(name)

    def get_by_id(self, tid: str) -> Optional[ToolDef]:
        return next((t for t in self._tools.values() if t.id == tid), None)

    def list_all(self) -> list[ToolDef]:
        return sorted(self._tools.values(), key=lambda t: (not t.builtin, t.name))

    def count(self) -> int:
        return len(self._tools)

    async def execute(self, name: str, params: dict) -> ToolResult:
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(success=False, error=f"Tool '{name}' not found")

        start = time.time()
        try:
            if tool.builtin:
                result = await _execute_builtin(name, params)
            else:
                result = await _execute_user_tool(tool, params)
            result.execution_ms = int((time.time() - start) * 1000)
            return result
        except Exception as e:
            return ToolResult(
                success=False, error=str(e),
                execution_ms=int((time.time() - start) * 1000),
            )

    def register_builtins(self):
        builtins = [
            ("shell", "Execute a shell command on the host OS. Full kernel access.",
             [ToolParam(name="command", type="string", description="Command to execute"),
              ToolParam(name="working_dir", type="string", description="Optional working directory", required=False),
              ToolParam(name="timeout_secs", type="integer", description="Optional timeout in seconds", required=False),
              ToolParam(name="stdin", type="string", description="Optional stdin payload", required=False)],
             "_builtin_shell"),
            ("run_script", "Execute a multi-line bash script.",
             [ToolParam(name="script", type="string", description="Script content"),
              ToolParam(name="working_dir", type="string", description="Optional working directory", required=False),
              ToolParam(name="timeout_secs", type="integer", description="Optional timeout in seconds", required=False),
              ToolParam(name="stdin", type="string", description="Optional stdin payload", required=False)],
             "_builtin_run_script"),
            ("python_exec", "Execute a Python snippet in a subprocess.",
             [ToolParam(name="code", type="string", description="Python code to execute"),
              ToolParam(name="working_dir", type="string", description="Optional working directory", required=False),
              ToolParam(name="timeout_secs", type="integer", description="Optional timeout in seconds", required=False),
              ToolParam(name="stdin", type="string", description="Optional stdin payload", required=False)],
             "_builtin_python_exec"),
            ("system_info", "Get system information (OS, memory, disk, CPU).", [], "_builtin_system_info"),
            ("kernel_info", "Get kernel version and system details.", [], "_builtin_kernel_info"),
            ("read_file", "Read the contents of a file.",
             [ToolParam(name="path", type="string", description="File path to read")],
             "_builtin_read_file"),
            ("write_file", "Write content to a file.",
             [ToolParam(name="path", type="string", description="File path"),
              ToolParam(name="content", type="string", description="File content")],
             "_builtin_write_file"),
            ("list_files", "List files in a directory.",
             [ToolParam(name="path", type="string", description="Directory path", required=False)],
             "_builtin_list_files"),
            ("http_request", "Make an HTTP request.",
             [ToolParam(name="url", type="string", description="URL"),
              ToolParam(name="method", type="string", description="GET/POST/PUT/DELETE", required=False)],
             "_builtin_http_request"),
            ("calculate", "Evaluate a math expression.",
             [ToolParam(name="expression", type="string", description="Math expression")],
             "_builtin_calculate"),
            ("get_datetime", "Get current date and time.", [], "_builtin_get_datetime"),
            ("list_processes", "List running processes.", [], "_builtin_list_processes"),
            ("get_env", "Get an environment variable.",
             [ToolParam(name="name", type="string", description="Variable name")],
             "_builtin_get_env"),
            ("network_info", "Get network interfaces and connections.", [], "_builtin_network_info"),
            ("semantic_memory_search", "Search persistent memory semantically across preferences, tasks, facts, episodes, workflows, and conversations.",
             [ToolParam(name="query", type="string", description="What to recall"),
              ToolParam(name="limit", type="integer", description="Max results", required=False),
              ToolParam(name="source_types_json", type="string", description="Optional JSON array of source types", required=False)],
             "_builtin_semantic_memory_search"),
            ("self_edit_pipeline", "Apply controlled code edits with snapshot, optional evaluation, optional tests, and automatic rollback on failure.",
             [ToolParam(name="workspace", type="string", description="Workspace root to edit", required=False),
              ToolParam(name="writes_json", type="string", description="JSON array of {path, content} objects"),
              ToolParam(name="eval_command", type="string", description="Optional evaluation command", required=False),
              ToolParam(name="test_command", type="string", description="Optional test command", required=False),
              ToolParam(name="rollback_on_failure", type="boolean", description="Rollback if checks fail", required=False),
              ToolParam(name="notes", type="string", description="Optional session notes", required=False)],
             "_builtin_self_edit_pipeline"),
            ("self_edit_status", "Inspect a prior self-edit session.",
             [ToolParam(name="session_id", type="string", description="Self-edit session id")],
             "_builtin_self_edit_status"),
            ("self_edit_rollback", "Rollback a prior self-edit session.",
             [ToolParam(name="session_id", type="string", description="Self-edit session id")],
             "_builtin_self_edit_rollback"),
        ]
        for name, desc, params, handler in builtins:
            self._tools[name] = ToolDef(
                id=new_id(), name=name, description=desc,
                params=params, builtin=True, handler=handler,
            )


async def _execute_builtin(name: str, params: dict) -> ToolResult:
    if name == "shell":
        cmd = params.get("command", "")
        if not cmd:
            return ToolResult(success=False, error="No command provided")
        r = await execute_command(
            cmd,
            working_dir=params.get("working_dir"),
            timeout_secs=params.get("timeout_secs"),
            stdin_data=params.get("stdin"),
        )
        return ToolResult(
            success=r.exit_code == 0,
            result={"stdout": r.stdout, "stderr": r.stderr, "exit_code": r.exit_code},
        )

    elif name == "run_script":
        script = params.get("script", "")
        r = await execute_command(
            f"bash -c {repr(script)}",
            working_dir=params.get("working_dir"),
            timeout_secs=params.get("timeout_secs"),
            stdin_data=params.get("stdin"),
        )
        return ToolResult(
            success=r.exit_code == 0,
            result={"stdout": r.stdout, "stderr": r.stderr, "exit_code": r.exit_code},
        )

    elif name == "python_exec":
        code = params.get("code", "")
        if not code:
            return ToolResult(success=False, error="No Python code provided")
        r = await execute_command(
            f"python3 -c {repr(code)}",
            working_dir=params.get("working_dir"),
            timeout_secs=params.get("timeout_secs"),
            stdin_data=params.get("stdin"),
        )
        return ToolResult(
            success=r.exit_code == 0,
            result={"stdout": r.stdout, "stderr": r.stderr, "exit_code": r.exit_code},
        )

    elif name == "system_info":
        r = await execute_command("uname -a && echo '---' && free -h 2>/dev/null && echo '---' && df -h / 2>/dev/null && echo '---' && nproc 2>/dev/null")
        return ToolResult(success=True, result=r.stdout)

    elif name == "kernel_info":
        r = await execute_command("uname -a")
        return ToolResult(success=True, result=r.stdout.strip())

    elif name == "read_file":
        path = params.get("path", "")
        try:
            content = Path(path).read_text()[:500_000]
            return ToolResult(success=True, result=content)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    elif name == "write_file":
        path = params.get("path", "")
        content = params.get("content", "")
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(content)
            return ToolResult(success=True, result=f"Wrote {len(content)} bytes to {path}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    elif name == "list_files":
        path = params.get("path", ".")
        try:
            entries = []
            for entry in sorted(Path(path).iterdir()):
                entries.append({
                    "name": entry.name,
                    "type": "dir" if entry.is_dir() else "file",
                    "size": entry.stat().st_size if entry.is_file() else 0,
                })
            return ToolResult(success=True, result=entries)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    elif name == "http_request":
        url = params.get("url", "")
        method = params.get("method", "GET").upper()
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.request(method, url)
                body = resp.text[:50_000]
                return ToolResult(success=True, result={
                    "status": resp.status_code, "body": body,
                    "headers": dict(resp.headers),
                })
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    elif name == "calculate":
        expr = params.get("expression", "")
        try:
            # Safe eval using only math operations
            allowed = set("0123456789+-*/().% ")
            if not all(c in allowed for c in expr.replace("**", "").replace("//", "")):
                return ToolResult(success=False, error="Invalid characters in expression")
            result = eval(expr, {"__builtins__": {}}, {})
            return ToolResult(success=True, result=result)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    elif name == "get_datetime":
        return ToolResult(success=True, result=datetime.now().isoformat())

    elif name == "list_processes":
        r = await execute_command("ps aux --sort=-%mem | head -20")
        return ToolResult(success=True, result=r.stdout)

    elif name == "get_env":
        name_val = params.get("name", "")
        val = os.environ.get(name_val)
        return ToolResult(success=val is not None, result=val, error="Not set" if val is None else None)

    elif name == "network_info":
        r = await execute_command("ip addr show 2>/dev/null || ifconfig 2>/dev/null; echo '---'; ss -tuln 2>/dev/null || netstat -tuln 2>/dev/null")
        return ToolResult(success=True, result=r.stdout)

    elif name == "semantic_memory_search":
        query = params.get("query", "")
        if not query:
            return ToolResult(success=False, error="No query provided")
        source_types = None
        raw_source_types = params.get("source_types_json")
        if isinstance(raw_source_types, str) and raw_source_types.strip():
            try:
                parsed = json.loads(raw_source_types)
                if isinstance(parsed, list):
                    source_types = [str(item) for item in parsed]
            except json.JSONDecodeError as exc:
                return ToolResult(success=False, error=f"Invalid source_types_json: {exc}")
        hits = await vector_memory.search(
            query,
            limit=int(params.get("limit", 8) or 8),
            source_types=source_types,
        )
        return ToolResult(success=True, result=[hit.model_dump(mode="json") for hit in hits])

    elif name == "self_edit_pipeline":
        writes_json = params.get("writes_json", "")
        if not writes_json:
            return ToolResult(success=False, error="No writes_json provided")
        try:
            parsed_writes = json.loads(writes_json)
        except json.JSONDecodeError as exc:
            return ToolResult(success=False, error=f"Invalid writes_json: {exc}")
        if not isinstance(parsed_writes, list):
            return ToolResult(success=False, error="writes_json must decode to a list")
        writes = [SelfEditWrite(**item) for item in parsed_writes]
        session = await self_edit_service.run(
            SelfEditRunRequest(
                workspace=params.get("workspace"),
                writes=writes,
                eval_command=params.get("eval_command"),
                test_command=params.get("test_command"),
                rollback_on_failure=params.get("rollback_on_failure", True),
                notes=params.get("notes"),
            )
        )
        return ToolResult(success=session.status in {"applied", "verified"}, result=session.model_dump(mode="json"))

    elif name == "self_edit_status":
        session_id = params.get("session_id", "")
        if not session_id:
            return ToolResult(success=False, error="No session_id provided")
        session = await self_edit_service.get(session_id)
        if session is None:
            return ToolResult(success=False, error=f"Session '{session_id}' not found")
        return ToolResult(success=True, result=session.model_dump(mode="json"))

    elif name == "self_edit_rollback":
        session_id = params.get("session_id", "")
        if not session_id:
            return ToolResult(success=False, error="No session_id provided")
        session = await self_edit_service.rollback(session_id)
        return ToolResult(success=True, result=session.model_dump(mode="json"))

    return ToolResult(success=False, error=f"Unknown builtin: {name}")


async def _execute_user_tool(tool: ToolDef, params: dict) -> ToolResult:
    handler = tool.handler

    # Interpolate params
    for key, value in params.items():
        handler = handler.replace(f"{{{{{key}}}}}", str(value))

    if handler.startswith("shell:"):
        cmd = handler[6:]
        r = await execute_command(
            cmd,
            working_dir=params.get("working_dir"),
            timeout_secs=params.get("timeout_secs"),
            stdin_data=params.get("stdin"),
        )
        return ToolResult(
            success=r.exit_code == 0,
            result={"stdout": r.stdout, "stderr": r.stderr, "exit_code": r.exit_code},
        )

    elif handler.startswith("script:"):
        script = handler[7:]
        r = await execute_command(
            f"bash -c {repr(script)}",
            working_dir=params.get("working_dir"),
            timeout_secs=params.get("timeout_secs"),
            stdin_data=params.get("stdin"),
        )
        return ToolResult(
            success=r.exit_code == 0,
            result={"stdout": r.stdout, "stderr": r.stderr, "exit_code": r.exit_code},
        )

    elif handler.startswith("python:"):
        code = handler[7:]
        r = await execute_command(
            f"python3 -c {repr(code)}",
            working_dir=params.get("working_dir"),
            timeout_secs=params.get("timeout_secs"),
            stdin_data=params.get("stdin"),
        )
        return ToolResult(
            success=r.exit_code == 0,
            result={"stdout": r.stdout, "stderr": r.stderr, "exit_code": r.exit_code},
        )

    elif handler.startswith("http:"):
        spec = handler[5:]
        method = "GET"
        url = spec
        if spec.startswith("http://") or spec.startswith("https://"):
            url = spec
        elif spec.count(":") >= 1 and spec.split(":", 1)[0].isalpha() and "://" in spec.split(":", 1)[1]:
            method, url = spec.split(":", 1)

        headers = params.get("headers")
        if isinstance(headers, str):
            try:
                headers = json.loads(headers)
            except json.JSONDecodeError:
                headers = {}

        json_body = params.get("json")
        if isinstance(json_body, str):
            try:
                json_body = json.loads(json_body)
            except json.JSONDecodeError:
                pass

        try:
            async with httpx.AsyncClient(timeout=params.get("timeout_secs", 30)) as client:
                resp = await client.request(
                    method.upper(),
                    url,
                    headers=headers if isinstance(headers, dict) else None,
                    json=json_body if isinstance(json_body, (dict, list)) else None,
                    content=params.get("body") if json_body is None else None,
                )
            return ToolResult(
                success=resp.is_success,
                result={
                    "status": resp.status_code,
                    "body": resp.text[:50_000],
                    "headers": dict(resp.headers),
                },
                error=None if resp.is_success else f"HTTP {resp.status_code}",
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    return ToolResult(success=False, error=f"Unsupported tool handler: {tool.handler}")
