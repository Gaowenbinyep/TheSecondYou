"""
Quick-start script for supervised fine-tuning (SFT).

This wrapper keeps the defaults that ship with the repository so newcomers can
simply run:

    python run_finetune.py

and rely on relative paths that already exist in the project tree.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_command(
    project_root: Path,
    base_model: Path,
    output_dir: Path,
    max_epochs: int,
) -> list[str]:
    train_script = project_root / "LLaMA-Factory" / "src" / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(
            f"Could not find training entrypoint: {train_script}"
        )

    cmd = [
        sys.executable,
        str(train_script),
        "--stage",
        "sft",
        "--finetuning_type",
        "full",
        "--do_train",
        "--model_name_or_path",
        str(base_model),
        "--dataset",
        "wechat_robot_sft_V1",
        "--dataset_dir",
        str(project_root / "data"),
        "--template",
        "qwen3",
        "--per_device_train_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "3",
        "--learning_rate",
        "5e-6",
        "--num_train_epochs",
        str(max_epochs),
        "--lr_scheduler_type",
        "cosine",
        "--warmup_ratio",
        "0.1",
        "--cutoff_len",
        "4096",
        "--output_dir",
        str(output_dir),
        "--bf16",
        "True",
        "--logging_steps",
        "10",
        "--save_steps",
        "2000",
        "--overwrite_cache",
        "--preprocessing_num_workers",
        "8",
        "--dataloader_num_workers",
        "4",
        "--trust_remote_code",
    ]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch supervised fine-tuning.")
    parser.add_argument(
        "--base-model",
        type=Path,
        default=Path("Base_models") / "Qwen3-1.7B",
        help="Path to the base model checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data") / "v1.0" / "Single_train.json",
        help="Training dataset in ShareGPT JSON format.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Saved_models") / "sft" / "demo_output",
        help="Directory to save the fine-tuned model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of fine-tuning epochs.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    base_model = project_root / args.base_model
    dataset_path = project_root / args.dataset
    output_dir = project_root / args.output_dir

    if not base_model.exists():
        raise FileNotFoundError(
            f"Base model checkpoint not found: {base_model}\n"
            "Please place your model under Base_models/."
        )
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {dataset_path}\n"
            "You can generate one with data_process.py or adjust --dataset."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_command(
        project_root=project_root,
        base_model=base_model,
        output_dir=output_dir,
        max_epochs=args.epochs,
    )

    print("Launching supervised fine-tuning:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
