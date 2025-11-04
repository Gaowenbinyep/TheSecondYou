"""
Minimal entrypoint to launch the LangGraph-powered agent.

Usage:

    python run_agent.py

This script ensures required environment variables are set and then delegates to
`Agent/main.py`, which starts an interactive console session.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the interactive agent.")
    parser.add_argument(
        "--model-base",
        default="http://localhost:8888/v1",
        help="Base URL for the OpenAI-compatible endpoint (default: local vLLM).",
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key for the endpoint. Use 'EMPTY' for local deployments.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Override model checkpoint used by the local agent.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    agent_entry = project_root / "Agent" / "main.py"
    if not agent_entry.exists():
        raise FileNotFoundError(f"Agent entry script not found: {agent_entry}")

    env = os.environ.copy()
    env.setdefault("OPENAI_API_BASE", args.model_base)
    env.setdefault("OPENAI_API_KEY", args.api_key)
    if args.model_path:
        env["WECHATROBOT_MODEL_PATH"] = args.model_path

    cmd = [
        sys.executable,
        str(agent_entry),
    ]

    print("Starting interactive agent...")
    print(f"Using API base: {env['OPENAI_API_BASE']}")
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
