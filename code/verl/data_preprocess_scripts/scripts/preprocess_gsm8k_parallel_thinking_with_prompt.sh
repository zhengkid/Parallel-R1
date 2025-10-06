#!/bin/bash

# 设置变量
LOCAL_DIR="./data/gsm8k/adaptive_parallel_thinking_with_prompt_v2/sft_all"
LOCAL_DIR_RL="./data/gsm8k/adaptive_parallel_thinking_with_prompt_v2/rl_all"
MAX_LENGTH=4096
PROMPT_DIR="../prompts/adaptive_parallel_thinking_v2.txt"
DATA_SOURCE="TongZheng1999/Deepseek-Qwen-3-8B-GSM8k-Parallel-data" 
MODEL_SOURCE="TongZheng1999/Qwen3-4B-Base-add-special-token"
REWARD_APPROACH="accuracy_times_special_token_reward" # "accuracy_times_special_token_reward"
# 构造命令

# 如果目录不存在就创建
if [ ! -d "$LOCAL_DIR" ]; then
  mkdir -p "$LOCAL_DIR"
  echo "Created directory: $LOCAL_DIR"
fi

if [ ! -d "$LOCAL_DIR_RL" ]; then
  mkdir -p "$LOCAL_DIR_RL"
  echo "Created directory: $LOCAL_DIR_RL"
fi

CMD="python gsm8k_parallel_thinking.py \
  --local_dir \"$LOCAL_DIR\" \
  --max_length $MAX_LENGTH \
  --prompt_dir \"$PROMPT_DIR\" \
  --data_source \"$DATA_SOURCE\" \
  --model_source \"$MODEL_SOURCE\" \
  --reward_approach \"$REWARD_APPROACH\""

# 打印命令（可选）
echo "Running command:"
echo $CMD

# 执行命令
eval $CMD


# 构造命令
CMD_RL="python gsm8k.py \
  --local_dir \"$LOCAL_DIR_RL\" \
  --prompt_dir \"$PROMPT_DIR\" \
  --data_source \"$DATA_SOURCE\" \
  --reward_approach \"$REWARD_APPROACH\""

# 打印命令（可选）
echo "Running command:"
echo $CMD_RL

# 执行命令
eval $CMD_RL