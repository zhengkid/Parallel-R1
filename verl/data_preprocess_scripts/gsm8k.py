# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import re
import random
import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

def load_template(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def format_full_prompts(template, problem):
    return template.format(Problem=problem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/gsm8k")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--data_source",
        type=str,
        default="TongZheng1999/Deepseek-Qwen-3-8B-GSM8k-Parallel-data-Adaptive-Thinking-v2",
        help="HuggingFace dataset path or identifier."
    )
    parser.add_argument(
        "--prompt_dir",
        type=str,
        default="./prompts/adaptive_parallel_thinking_v2.txt",
        help="Path to the prompt template file."
    )
    parser.add_argument(
        "--reward_approach",
        type=str,
        default="accuracy_add_special_token_reward",
        help="Reward approach."
    )

    args = parser.parse_args()

    dataset = datasets.load_dataset("openai/gsm8k", "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    prompt_path = args.prompt_dir 
    prompt = load_template(prompt_path)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            question = format_full_prompts(prompt, question_raw)
            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            data = {
                "data_source": args.data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question,
                    "reward_method": args.reward_approach,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
