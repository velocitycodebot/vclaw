import argparse
import sys
from getpass import getpass
from typing import Optional

import uvicorn

from app.runtime_config import (
    DEFAULT_ANTHROPIC_MODEL,
    RuntimeConfig,
    RuntimeConfigError,
    SUPPORTED_ANTHROPIC_MODELS,
    load_runtime_config,
    resolve_runtime_config,
    save_runtime_config,
)


def _prompt(prompt: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value or (default or "")


def _prompt_api_key(existing: str = "") -> str:
    while True:
        prompt = "Anthropic API key"
        if existing:
            prompt += " [press enter to keep saved value]"
        value = getpass(f"{prompt}: ").strip()
        if value:
            return value
        if existing:
            return existing
        print("Anthropic API key is required.", file=sys.stderr)


def _prompt_port(default: int) -> int:
    while True:
        value = _prompt("Server port", str(default))
        try:
            port = int(value)
        except ValueError:
            print("Port must be an integer.", file=sys.stderr)
            continue
        if 1 <= port <= 65535:
            return port
        print("Port must be between 1 and 65535.", file=sys.stderr)


def _prompt_model(default: str) -> str:
    options = ", ".join(SUPPORTED_ANTHROPIC_MODELS)
    while True:
        value = _prompt(f"Anthropic model ({options})", default).strip()
        if value in SUPPORTED_ANTHROPIC_MODELS:
            return value
        print(f"Model must be one of: {options}.", file=sys.stderr)


def _build_onboard_config(args: argparse.Namespace) -> RuntimeConfig:
    existing = load_runtime_config()

    api_key = args.api_key or _prompt_api_key(existing.anthropic_api_key if existing else "")
    model_default = args.model if args.model is not None else (existing.anthropic_model if existing else DEFAULT_ANTHROPIC_MODEL)
    host_default = args.host if args.host is not None else (existing.host if existing else "0.0.0.0")
    port_default = args.port if args.port is not None else (existing.port if existing else 3000)

    model = args.model if args.model is not None else _prompt_model(model_default)
    host = args.host if args.host is not None else _prompt("Server host", host_default)
    port = args.port if args.port is not None else _prompt_port(port_default)

    return RuntimeConfig(
        anthropic_api_key=api_key,
        anthropic_model=model,
        host=host,
        port=port,
    )


def _start_service(config: RuntimeConfig) -> int:
    # Keep the API client contract unchanged by exporting the saved key at process start.
    import os
    from app.api.app import app

    os.environ["ANTHROPIC_API_KEY"] = config.anthropic_api_key
    os.environ["ANTHROPIC_MODEL"] = config.anthropic_model
    os.environ["VCLAW_HOST"] = config.host
    os.environ["VCLAW_PORT"] = str(config.port)
    uvicorn.run(app, host=config.host, port=config.port, reload=False)
    return 0


def onboard_command(args: argparse.Namespace) -> int:
    config = _build_onboard_config(args)
    save_runtime_config(config)
    print("Saved startup configuration to data/runtime_config.json")

    if args.no_start:
        return 0

    print(f"Starting vclaw on http://{config.host}:{config.port} using {config.anthropic_model}")
    return _start_service(config)


def start_command(args: argparse.Namespace) -> int:
    config = resolve_runtime_config(
        api_key=args.api_key,
        model=args.model,
        host=args.host,
        port=args.port,
    )
    print(f"Starting vclaw on http://{config.host}:{config.port} using {config.anthropic_model}")
    return _start_service(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vclaw")
    subparsers = parser.add_subparsers(dest="command")

    onboard = subparsers.add_parser("onboard", help="Save required startup settings and start the service")
    onboard.add_argument("--api-key", help="Anthropic API key to save")
    onboard.add_argument("--model", choices=SUPPORTED_ANTHROPIC_MODELS, help="Anthropic model to save")
    onboard.add_argument("--host", help="Host to bind the service to")
    onboard.add_argument("--port", type=int, help="Port to bind the service to")
    onboard.add_argument("--no-start", action="store_true", help="Save config without starting the service")
    onboard.set_defaults(handler=onboard_command)

    start = subparsers.add_parser("start", help="Start the service using saved startup settings")
    start.add_argument("--api-key", help="Override the saved Anthropic API key for this run")
    start.add_argument("--model", choices=SUPPORTED_ANTHROPIC_MODELS, help="Override the saved Anthropic model for this run")
    start.add_argument("--host", help="Override the saved host for this run")
    start.add_argument("--port", type=int, help="Override the saved port for this run")
    start.set_defaults(handler=start_command)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1

    try:
        return handler(args)
    except RuntimeConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nShutting down.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
