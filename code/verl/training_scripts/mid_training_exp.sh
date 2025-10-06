#!/usr/bin/env bash
pkill python
sleep 5

set -x

DAPO_train_path=./data_preprocess_scripts/data/dapo/adaptive_parallel_thinking_final_with_prompt_v3/rl_all_accuracy_reward/train.parquet
APO_combiene_test_path=./data_preprocess_scripts/data/APO_combine/adaptive_parallel_thinking_final_with_prompt_v3/rl_all_accuracy_reward/test.parquet
train_files="['$DAPO_train_path']"
test_files="['$APO_combiene_test_path']"

project_name=Parallel-R1
experiment_name=Mid_Training_Exp
default_local_dir=./$project_name/$experiment_name
validation_data_dir=$default_local_dir/val-log
base_model=Parallel-R1/Parallel-R1-Unseen_Step_200

use_dynamic_bsz=False
offload=False
gradient_ckpt=True
infer_micro_batch_size=4

max_length=3000
max_prompt_len=2000

train_rollout_n=8

# sampling parameters
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
export VLLM_USE_V1=1
# 
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=1 \
    algorithm.lam=1 \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=256 \
    data.return_raw_chat=True \
    data.max_prompt_length=${max_prompt_len} \
    data.max_response_length=${max_length} \
    data.filter_overlong_prompts=False \
    data.truncation='left' \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_length + max_prompt_len)) \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((max_length + max_prompt_len)) \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((max_length + max_prompt_len)) \
    actor_rollout_ref.model.path="${base_model}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=${gradient_ckpt} \
    actor_rollout_ref.rollout.n=${train_rollout_n} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${infer_micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.agent.max_path_response_length=4096 \
    actor_rollout_ref.rollout.agent.num_workers=32 \
    actor_rollout_ref.rollout.agent.max_iterations_for_parallel_thinking=4 \
    actor_rollout_ref.rollout.agent.num_paths=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=2 \
    trainer.total_epochs=100 \
    trainer.validation_data_dir="$default_local_dir/validation-log" \
    trainer.rollout_data_dir="$default_local_dir/rollout-log" \
    trainer.default_local_dir="$default_local_dir" \
    trainer.val_before_train=False \
    trainer.resume_mode=auto $@
