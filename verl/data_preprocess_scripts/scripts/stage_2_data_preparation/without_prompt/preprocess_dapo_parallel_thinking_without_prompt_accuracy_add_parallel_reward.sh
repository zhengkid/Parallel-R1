#!/usr/bin/env bash
# run_process_prompts.sh
# 用法：bash run_process_prompts.sh DATASET TEMPLATE_FILE OUTPUT_FILE [SPLIT]

set -euo pipefail

DATASET_NAME=Leo-Dai/APO_combine #Leo-Dai/dapo-math-17k_dedup # Leo-Dai/APO_combine #Leo-Dai/dapo-math-17k_dedup 
TEMPLATE_FILE="../prompts/adaptive_parallel_thinking-no-prompt.txt"
OUTPUT_dir="data/APO_combine/adaptive_parallel_thinking_final_no_prompt/rl_all_accuracy_add_parallel_reward"
SPLIT="train"            # 可选，默认 train
REWARD_METHOD="accuracy_add_parallel_reward" #"accuracy_times_special_token_reward_times_format"

# 如果目录不存在就创建
if [ ! -d "$OUTPUT_dir" ]; then
  mkdir -p "$OUTPUT_dir"
  echo "Created directory: $OUTPUT_dir"
fi



python dapo_parallel_thinking.py \
  --dataset_name "$DATASET_NAME" \
  --prompt_path "$TEMPLATE_FILE" \
  --split "$SPLIT" \
  --output_file "$OUTPUT_dir/test.parquet" \
  --reward_approach "$REWARD_METHOD"


DATASET_NAME=Leo-Dai/dapo-math-17k_dedup #Leo-Dai/dapo-math-17k_dedup # Leo-Dai/APO_combine #Leo-Dai/dapo-math-17k_dedup 
TEMPLATE_FILE="../prompts/adaptive_parallel_thinking-no-prompt.txt"
OUTPUT_dir="data/dapo/adaptive_parallel_thinking_final_no_prompt/rl_all_accuracy_add_parallel_reward"
SPLIT="train"            # 可选，默认 train
REWARD_METHOD="accuracy_add_parallel_reward" #"accuracy_times_special_token_reward_times_format"
# 如果目录不存在就创建
if [ ! -d "$OUTPUT_dir" ]; then
  mkdir -p "$OUTPUT_dir"
  echo "Created directory: $OUTPUT_dir"
fi


python dapo_parallel_thinking.py \
  --dataset_name "$DATASET_NAME" \
  --prompt_path "$TEMPLATE_FILE" \
  --split "$SPLIT" \
  --output_file "$OUTPUT_dir/train.parquet" \
  --reward_approach "$REWARD_METHOD"