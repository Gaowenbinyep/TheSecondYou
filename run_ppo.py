"""
Wrapper script to launch PPO training with sensible defaults.

Usage:

    python run_ppo.py

You can pass optional arguments to override a few key settings:

    python run_ppo.py --max-steps 50 --log-file logs/custom_ppo.log
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch PPO training loop.")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("logs") / "ppo_train.log",
        help="Location to stream PPO logs.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional override for RunCfg.max_training_steps.",
    )
    parser.add_argument(
        "--cuda-devices",
        type=str,
        default=None,
        help="Comma separated GPU device ids to expose (e.g. '0,1').",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    ppo_root = project_root / "PPO"
    train_script = ppo_root / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(f"PPO training script not found: {train_script}")

    # Ensure logs directory exists.
    args.log_file.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.cuda_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    if args.max_steps is not None:
        env["PPO_MAX_STEPS"] = str(args.max_steps)

    cmd = [
        sys.executable,
        str(train_script),
    ]

    log_path = project_root / args.log_file
    print("Launching PPO training:")
    print(" ".join(cmd))
    print(f"Streaming logs to: {log_path}")

    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(cmd, check=True, env=env, stdout=log_file, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    main()
