#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re
import sys

# URL 正则（过滤含链接）
URL_PATTERN = re.compile(r'(https?://\S+|www\.\S+)', re.IGNORECASE)

# 匹配方括号内恰好两个字符，例如 [白眼]、[OK]
BRACKET_2CHAR_PATTERN = re.compile(r'\[[^\]]{2}\]')

def has_url(text: str) -> bool:
    if not text:
        return False
    return bool(URL_PATTERN.search(text))

def has_bracket_2chars(text: str) -> bool:
    """判断是否包含形如 [xx] 的内容（方括号内恰好两个字符）"""
    if not text:
        return False
    return bool(BRACKET_2CHAR_PATTERN.search(text))

def conv_too_long(conversations, max_len: int = 50) -> bool:
    """判断 user + assistant 内容字数是否超过 max_len"""
    total_len = 0
    for msg in conversations:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        if role in ("user", "assistant"):
            total_len += len(content)
    return total_len > max_len

def conv_has_url(conversations) -> bool:
    """判断对话是否包含 URL"""
    return any(has_url(msg.get("content", "")) for msg in conversations)

def conv_has_bracket_2chars(conversations) -> bool:
    """判断对话是否包含形如 [xx] 的表达"""
    return any(has_bracket_2chars(msg.get("content", "")) for msg in conversations)

def filter_file(input_path: str, output_path: str,
                max_len: int = 50):
    keep_cnt = 0
    drop_cnt = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                drop_cnt += 1
                continue

            conversations = obj.get("conversations", [])

            # 规则过滤
            if conv_too_long(conversations, max_len=max_len):
                drop_cnt += 1
                continue

            if conv_has_url(conversations):
                drop_cnt += 1
                continue

            if conv_has_bracket_2chars(conversations):
                drop_cnt += 1
                continue

            # 写入保留样本
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            keep_cnt += 1

    print(f"保留样本: {keep_cnt} 条，丢弃样本: {drop_cnt} 条")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python data_process.py input.jsonl output.jsonl")
        sys.exit(1)

    filter_file(sys.argv[1], sys.argv[2], max_len=50)
