import argparse
from datasets import load_dataset
import pandas as pd

def load_template(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def extract_between_blank_lines(text: str) -> str:
    """
    从文本中提取第一个和第二个双换行之间的内容。
    如果双换行不足，则返回原文本并去除首尾空白。
    """
    parts = text.split("\n\n")
    if len(parts) >= 3:
        return parts[1].strip()
    return text.strip()

def format_full_prompts(template, problem):
    return template.format(Problem=problem)

def process_split(dataset_name: str, template: str, reward_approach: str, split: str) -> pd.DataFrame:
    # 加载指定 split
    ds = load_dataset(dataset_name, split=split)
    
    # 只处理 prompt 字段，其他字段保持不动
    def transform(example):
        orig = example.get("prompt", [])
        if isinstance(orig, list) and orig:
            content = orig[0].get("content", "")
            core = content
            core = format_full_prompts(template, core)
            example["prompt"] = [{"role": orig[0].get("role", "user"), "content": core}]
            example['data_source'] = example['source']
            if "extra_info" not in example.keys():
                example["extra_info"] = {}
                example["extra_info"]['reward_method'] = reward_approach
            else:
                example["extra_info"]['reward_method'] = reward_approach
        return example
    
    processed = ds.map(transform)
    # 转成 pandas DataFrame
    return processed.to_pandas()

def main():
    parser = argparse.ArgumentParser(
        description="从 HuggingFace 数据集 prompt 中提取双换行间问题并保存为 Parquet 文件"
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True,
        help="HuggingFace 上的数据集名称，例如 'Leo-Dai/APO_AIME25'"
    )
    parser.add_argument(
        "--prompt_path", type=str, required=True,
        help="HuggingFace 上的数据集名称，例如 'Leo-Dai/APO_AIME25'"
    )
    parser.add_argument(
        "--split", type=str, default="train",
        help="要加载的数据集 split，默认为 'train'"
    )
    parser.add_argument(
        "--reward_approach", type=str, default="",
        help=""
    )
    parser.add_argument(
        "--output_file", type=str, required=True,
        help="输出 Parquet 文件路径，例如 'output.parquet'"
    )
    
    args = parser.parse_args()
    template = load_template(args.prompt_path)
    df = process_split(args.dataset_name, template, args.reward_approach, args.split)
    df.to_parquet(args.output_file, index=False)
    print(f"Processed split '{args.split}' of dataset '{args.dataset_name}', saved to {args.output_file}")

if __name__ == "__main__":
    main()


