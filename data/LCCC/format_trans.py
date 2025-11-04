import json
from pathlib import Path


def convert_to_sharegpt(input_path, output_path):
    # 读取原始JSON数据
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    sharegpt_data = []

    for conversation in raw_data:
        # 初始化对话列表
        formatted_conv = []
        length = len(conversation)
        for i, message in enumerate(conversation):
            if i == length - 1 and i % 2 == 0:
                continue
            role = "user" if i % 2 == 0 else "assistant"
            formatted_conv.append(
                {
                    "role": role,
                    "content": message.strip(),
                }
            )

        # 添加完整对话结构
        sharegpt_data.append({"conversations": formatted_conv})

    # 保存为ShareGPT格式
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sharegpt_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent
    # 输入输出路径
    input_json = data_dir / "LCCC-base_test.json"
    output_json = data_dir / "sharegpt_format.json"

    # 执行转换
    convert_to_sharegpt(input_json, output_json)
    print(f"转换完成！已保存于 {output_json}")
