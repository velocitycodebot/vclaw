import json
import os
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_PATH = DATA_DIR / "runtime_config.json"
SUPPORTED_ANTHROPIC_MODELS = (
    "claude-sonnet-4-6",
    "claude-opus-4-6",
)
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"


class RuntimeConfigError(Exception):
    pass


class RuntimeConfig(BaseModel):
    anthropic_api_key: str = Field(min_length=1)
    anthropic_model: Literal["claude-sonnet-4-6", "claude-opus-4-6"] = DEFAULT_ANTHROPIC_MODEL
    host: str = "0.0.0.0"
    port: int = Field(default=3000, ge=1, le=65535)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.write("\n")
    os.replace(tmp_path, path)

    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def load_runtime_config() -> Optional[RuntimeConfig]:
    if not CONFIG_PATH.exists():
        return None

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeConfigError(f"Unable to read runtime config at {CONFIG_PATH}: {exc}") from exc

    try:
        return RuntimeConfig(**raw)
    except ValidationError as exc:
        raise RuntimeConfigError(f"Invalid runtime config at {CONFIG_PATH}: {exc}") from exc


def save_runtime_config(config: RuntimeConfig) -> None:
    _write_json(CONFIG_PATH, config.model_dump())


def resolve_runtime_config(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> RuntimeConfig:
    saved = load_runtime_config()

    resolved_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not resolved_api_key and saved is not None:
        resolved_api_key = saved.anthropic_api_key

    resolved_model = model or os.environ.get("ANTHROPIC_MODEL", "")
    if not resolved_model and saved is not None:
        resolved_model = saved.anthropic_model
    if not resolved_model:
        resolved_model = DEFAULT_ANTHROPIC_MODEL

    resolved_host = host if host is not None else (saved.host if saved is not None else "0.0.0.0")
    resolved_port = port if port is not None else (saved.port if saved is not None else 3000)

    try:
        return RuntimeConfig(
            anthropic_api_key=resolved_api_key,
            anthropic_model=resolved_model,
            host=resolved_host,
            port=resolved_port,
        )
    except ValidationError as exc:
        raise RuntimeConfigError(
            "Missing required startup config. Run `vclaw onboard` to save your API key."
        ) from exc
