set -x

# if [ "$#" -lt 2 ]; then
#     echo "Usage: run_gemma_7b.sh <nproc_per_node> <save_path> [other_configs...]"
#     exit 1
# fi

nproc_per_node=8
project_name=Parallel-R1
experiment_name=Parallel-SFT-Unseen
save_path=./$project_name/$experiment_name

# # Shift the arguments so $@ refers to the rest
# shift 2
#  -m verl.trainer.fsdp_sft_parallel_thinking_trainer \
# -m verl.trainer.fsdp_sft_trainer \
torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_parallel_sft_trainer \
    data.train_files=./data_preprocess_scripts/data/gsm8k/adaptive_parallel_thinking_final_with_prompt_v3/sft_all_accuracy_times_parallel_reward/train.parquet \
    data.val_files=./data_preprocess_scripts/data/gsm8k/adaptive_parallel_thinking_final_with_prompt_v3/rl_all_accuracy_times_parallel_reward/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.max_length=4096 \
    +data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=1 \
    data.train_batch_size=128 \
    model.partial_pretrain=Parallel-R1/Qwen3-4B-Base-add-special-token \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.total_epochs=5 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null $@
