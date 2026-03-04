import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Optional
from app.schemas.models import Workflow, ToolCallResult, ToolResult, new_id
from app.services.tools import ToolRegistry


class WorkflowEngine:
    def __init__(self):
        self._workflows: dict[str, Workflow] = {}

    def register(self, wf: Workflow):
        self._workflows[wf.id] = wf

    def remove(self, wid: str):
        self._workflows.pop(wid, None)

    def get(self, wid: str) -> Optional[Workflow]:
        return self._workflows.get(wid)

    def list_all(self) -> list[Workflow]:
        return sorted(self._workflows.values(), key=lambda w: w.created_at)

    def count(self) -> int:
        return len(self._workflows)

    def toggle(self, wid: str) -> Optional[bool]:
        wf = self._workflows.get(wid)
        if not wf:
            return None
        wf.enabled = not wf.enabled
        return wf.enabled

    def update_last_run(self, wid: str):
        wf = self._workflows.get(wid)
        if wf:
            wf.last_run = datetime.now(timezone.utc)


async def execute_workflow(wf: Workflow, registry: ToolRegistry) -> list[ToolCallResult]:
    results = []
    last_result = ""

    for i, step in enumerate(wf.steps):
        # Delay if specified
        if step.delay_ms:
            await asyncio.sleep(step.delay_ms / 1000)

        # Interpolate variables
        params = {}
        for k, v in step.params.items():
            s = str(v)
            s = s.replace("{{last}}", last_result)
            for j, prev in enumerate(results):
                s = s.replace(f"{{{{step{j}}}}}", json.dumps(prev.result.result) if prev.result.result else "")
            params[k] = s

        result = await registry.execute(step.tool_name, params)

        results.append(ToolCallResult(
            tool=step.tool_name, params=params, result=result,
        ))

        if result.result:
            last_result = str(result.result)[:2000]

        if not result.success and not step.continue_on_error:
            break

    return results


async def run_scheduler(get_engine, get_registry, get_db_funcs):
    """Background scheduler that checks for due workflows every 30s."""
    import app.storage.database as db

    while True:
        await asyncio.sleep(30)
        try:
            engine = get_engine()
            registry = get_registry()
            for wf in engine.list_all():
                if not wf.enabled or not wf.schedule:
                    continue

                now = datetime.now(timezone.utc)
                should_run = False

                if wf.schedule.type == "interval" and wf.schedule.interval_ms:
                    if wf.last_run is None:
                        should_run = True
                    else:
                        elapsed = (now - wf.last_run).total_seconds() * 1000
                        should_run = elapsed >= wf.schedule.interval_ms

                elif wf.schedule.type == "daily":
                    h = wf.schedule.hour or 0
                    m = wf.schedule.minute or 0
                    if now.hour == h and now.minute == m:
                        if wf.last_run is None or wf.last_run.date() < now.date():
                            should_run = True

                if should_run:
                    results = await execute_workflow(wf, registry)
                    engine.update_last_run(wf.id)
                    await db.update_workflow_last_run(wf.id)

                    from app.schemas.models import ExecLogEntry
                    log = ExecLogEntry(
                        id=new_id(), source="workflow", source_name=wf.name,
                        input_summary=f"{len(wf.steps)} steps",
                        result_summary=f"{sum(1 for r in results if r.result.success)}/{len(results)} ok",
                        success=all(r.result.success for r in results),
                        execution_ms=sum(r.result.execution_ms for r in results),
                        scheduled=True,
                    )
                    await db.add_log_entry(log)
        except Exception as e:
            print(f"Scheduler error: {e}")
