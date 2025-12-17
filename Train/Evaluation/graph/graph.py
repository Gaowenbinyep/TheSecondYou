#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd


def parse_train_steps_from_log(log_path: str) -> pd.DataFrame:
    """
    从训练日志中解析出所有 TRAIN_STEP 的 JSON 指标，返回 DataFrame。

    预期日志格式类似：
    2025-11-21 19:20:19,416 - __main__ - INFO - TRAIN_STEP | {"step": 152, "progress": 51.0, ...}
    """
    records: List[Dict] = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if "TRAIN_STEP |" not in line:
                continue

            # 取出 "TRAIN_STEP |" 后面的 JSON 字符串
            try:
                _, json_part = line.split("TRAIN_STEP |", 1)
            except ValueError:
                continue

            json_part = json_part.strip()

            # 只截取第一个 { 到最后一个 } 之间的内容
            if "{" in json_part:
                json_part = json_part[json_part.find("{"):]
            if "}" in json_part:
                json_part = json_part[: json_part.rfind("}") + 1]

            try:
                data = json.loads(json_part)
                records.append(data)
            except json.JSONDecodeError:
                # 如果这一行 JSON 解析失败就跳过
                continue

    if not records:
        raise RuntimeError(
            f"在日志 {log_path} 中没有解析到任何 TRAIN_STEP 记录，请检查日志格式/关键字。"
        )

    df = pd.DataFrame(records)

    # 如果有 step 字段，按 step 排序；没有就按行号顺序
    if "step" in df.columns:
        df = df.sort_values("step").reset_index(drop=True)

    return df


def plot_compare(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    metrics: List[str],
    output_path: str,
    title_prefix: str = "",
    label1: str = "run1",
    label2: str = "run2",
):
    """
    将两个 DataFrame 在指定 metrics 上画成多子图对比曲线。

    - df1 用蓝色
    - df2 用红色
    - x 轴使用 step（如果存在），否则用索引
    """
    # 选择 x 轴
    if "step" in df1.columns and "step" in df2.columns:
        x1 = df1["step"]
        x2 = df2["step"]
        x_label = "step"
    else:
        x1 = df1.index
        x2 = df2.index
        x_label = "index"

    # 过滤掉两个 df 中都不存在的列
    valid_metrics = []
    for m in metrics:
        if m in df1.columns or m in df2.columns:
            valid_metrics.append(m)
        else:
            print(f"[WARN] 指标 {m} 在两个日志中都不存在，已跳过。")

    if not valid_metrics:
        print(f"[WARN] 没有可用的指标用于绘制：{metrics}")
        return

    n = len(valid_metrics)
    # 简单按一列多行排布
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]  # 保证可迭代

    for ax, metric in zip(axes, valid_metrics):
        if metric in df1.columns:
            ax.plot(x1, df1[metric], label=label1, color="blue")
        if metric in df2.columns:
            ax.plot(x2, df2[metric], label=label2, color="red")

        ax.set_ylabel(metric)
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel(x_label)

    if title_prefix:
        fig.suptitle(title_prefix, fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95] if title_prefix else None)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="对比两个训练日志中的 TRAIN_STEP 指标，并画出对比曲线"
    )
    parser.add_argument(
        "--log_path_1",
        type=str,
        required=True,
        help="第一个日志文件路径，如: train_v1.log",
    )
    parser.add_argument(
        "--log_path_2",
        type=str,
        required=True,
        help="第二个日志文件路径，如: train_v2.log",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots_compare",
        help="输出图片目录，默认: plots_compare",
    )
    parser.add_argument(
        "--label_1",
        type=str,
        default="run1",
        help="第一条曲线的图例名称，默认: run1",
    )
    parser.add_argument(
        "--label_2",
        type=str,
        default="run2",
        help="第二条曲线的图例名称，默认: run2",
    )

    args = parser.parse_args()

    df1 = parse_train_steps_from_log(args.log_path_1)
    df2 = parse_train_steps_from_log(args.log_path_2)

    print("log1 解析到的列：", list(df1.columns))
    print("log2 解析到的列：", list(df2.columns))

    # 奖励相关指标
    reward_metrics = [
        "style_alignment",
        "relevance",
        "conciseness",
        "role_cognition_clarity",
        "rep_penalty",
        "mean_reward",
    ]
    reward_path = os.path.join(args.output_dir, "reward_metrics_compare.png")
    plot_compare(
        df1,
        df2,
        reward_metrics,
        reward_path,
        title_prefix="多维度奖励曲线对比",
        label1=args.label_1,
        label2=args.label_2,
    )

    # 训练相关指标
    train_metrics = ["value_loss", "entropy", "kl"]
    train_path = os.path.join(args.output_dir, "train_metrics_compare.png")
    plot_compare(
        df1,
        df2,
        train_metrics,
        train_path,
        title_prefix="训练指标曲线对比",
        label1=args.label_1,
        label2=args.label_2,
    )


if __name__ == "__main__":
    main()
