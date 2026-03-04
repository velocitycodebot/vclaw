import asyncio
import json
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo
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

    def set_enabled(self, wid: str, enabled: bool) -> Optional[Workflow]:
        wf = self._workflows.get(wid)
        if not wf:
            return None
        wf.enabled = enabled
        return wf


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


def _schedule_timezone_name(wf: Workflow) -> str:
    if wf.schedule and wf.schedule.timezone:
        return wf.schedule.timezone
    local_tz = datetime.now().astimezone().tzinfo
    if hasattr(local_tz, "key"):
        return getattr(local_tz, "key")
    return "UTC"


def _schedule_now(wf: Workflow) -> datetime:
    return datetime.now(ZoneInfo(_schedule_timezone_name(wf)))


def _should_run_workflow(wf: Workflow, now_utc: datetime) -> bool:
    if not wf.enabled or not wf.schedule:
        return False

    schedule = wf.schedule

    if schedule.type == "interval" and schedule.interval_ms:
        if wf.last_run is None:
            return True
        elapsed = (now_utc - wf.last_run).total_seconds() * 1000
        return elapsed >= schedule.interval_ms

    now_local = _schedule_now(wf)
    last_run_local = wf.last_run.astimezone(now_local.tzinfo) if wf.last_run else None

    if schedule.type == "daily":
        hour = schedule.hour or 0
        minute = schedule.minute or 0
        return (
            now_local.hour == hour
            and now_local.minute == minute
            and (last_run_local is None or last_run_local.date() < now_local.date())
        )

    if schedule.type == "weekly":
        hour = schedule.hour or 0
        minute = schedule.minute or 0
        days = set(schedule.days_of_week or [])
        if days and now_local.weekday() not in days:
            return False
        if now_local.hour != hour or now_local.minute != minute:
            return False
        if last_run_local is None:
            return True
        year_week = now_local.isocalendar()[:2]
        last_year_week = last_run_local.isocalendar()[:2]
        return year_week != last_year_week

    return False


async def run_scheduler(get_engine, get_registry, get_db_funcs):
    """Background scheduler that checks for due workflows every 30s."""
    import app.storage.database as db

    while True:
        await asyncio.sleep(30)
        try:
            engine = get_engine()
            registry = get_registry()
            for wf in engine.list_all():
                now = datetime.now(timezone.utc)
                if _should_run_workflow(wf, now):
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
