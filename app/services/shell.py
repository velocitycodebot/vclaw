import asyncio
import time
import re
from typing import Optional
from app.schemas.models import ShellResult, new_id

BLOCKED_PATTERNS = [
    r"rm\s+-rf\s+/\s*$", r"rm\s+-rf\s+/\*", r"mkfs\.", r"dd\s+if=.+of=/dev/",
    r":\(\)\{.*\|.*\}", r"fork\s*bomb", r"chmod\s+-R\s+777\s+/\s*$",
    r">\s*/dev/sda", r"mv\s+/\s+/dev/null",
]


def is_blocked(command: str) -> Optional[str]:
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return f"Blocked dangerous pattern: {pattern}"
    return None


async def execute_command(
    command: str,
    working_dir: Optional[str] = None,
    timeout_secs: Optional[int] = None,
    stdin_data: Optional[str] = None,
) -> ShellResult:
    blocked = is_blocked(command)
    if blocked:
        return ShellResult(
            id=new_id(), command=command, stderr=blocked, exit_code=1,
        )

    timeout = timeout_secs or 30
    start = time.time()

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE if stdin_data else None,
            cwd=working_dir,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(input=stdin_data.encode() if stdin_data else None),
            timeout=timeout,
        )
        elapsed = int((time.time() - start) * 1000)

        stdout = stdout_bytes.decode("utf-8", errors="replace")[:1_000_000]
        stderr = stderr_bytes.decode("utf-8", errors="replace")[:500_000]

        return ShellResult(
            id=new_id(), command=command,
            stdout=stdout, stderr=stderr,
            exit_code=proc.returncode or 0,
            execution_ms=elapsed,
        )
    except asyncio.TimeoutError:
        elapsed = int((time.time() - start) * 1000)
        return ShellResult(
            id=new_id(), command=command,
            stderr=f"Timed out after {timeout}s", exit_code=124,
            execution_ms=elapsed,
        )
    except Exception as e:
        elapsed = int((time.time() - start) * 1000)
        return ShellResult(
            id=new_id(), command=command,
            stderr=str(e), exit_code=1, execution_ms=elapsed,
        )
