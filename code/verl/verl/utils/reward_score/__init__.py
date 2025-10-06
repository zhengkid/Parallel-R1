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
# from . import gsm8k, math, prime_math, prime_code

from verl.utils.import_utils import deprecated


def default_compute_score(
    data_source,
    solution_str,
    solution_str_with_special_tokens,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if "GSM8k" in data_source or data_source == "openai/gsm8k":
        if extra_info['reward_method'] == 'accuracy_reward':
            from . import gsm8k
            res = gsm8k.compute_score(solution_str, ground_truth)
        elif extra_info['reward_method'] == 'emphase_acc':
            from . import gsm8k_emphase_acc
            print("reward_func:", "emphase_acc reward")
            res = gsm8k_emphase_acc.compute_score(solution_str, solution_str_with_special_tokens, ground_truth)
        elif extra_info['reward_method'] == 'emphase_parallel':
            from . import gsm8k_emphase_parallel
            print("reward_func:", "emphase_parallel reward")
            res = gsm8k_emphase_parallel.compute_score(solution_str, solution_str_with_special_tokens, ground_truth)
        elif extra_info['reward_method'] == 'emphase_parallel_add_diversity':
            from . import gsm8k_emphase_parallel_add_diversity
            print("reward_func:", "emphase_parallel_add_diversity reward")
            res = gsm8k_emphase_parallel_add_diversity.compute_score(solution_str, solution_str_with_special_tokens, ground_truth)
        elif extra_info['reward_method'] == 'accuracy_add_parallel_reward':
            from . import gsm8k_accuracy_add_parallel
            res = gsm8k_accuracy_add_parallel.compute_score(solution_str, solution_str_with_special_tokens, ground_truth)
        elif extra_info['reward_method'] == 'accuracy_times_special_token_reward':
            from . import gsm8k_add_special_token_reward
            res = gsm8k_add_special_token_reward.compute_score(solution_str, solution_str_with_special_tokens, ground_truth)
        elif extra_info['reward_method'] == 'format_times_accuracy_times_parallel_reward':
            from . import gsm8k_accuracy_times_format_times_parallel
            res = gsm8k_accuracy_times_format_times_parallel.compute_score(
                solution_str, solution_str_with_special_tokens, ground_truth
            )
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
        from . import math

        res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source == "math_dapo" or data_source.startswith("aime") or "APO" in data_source or data_source == "HuggingFaceH4/MATH":
        if extra_info['reward_method'] == 'accuracy_reward':
            from . import math_dapo
            res = math_dapo.compute_score(solution_str, ground_truth)
        elif extra_info['reward_method'] == 'accuracy_add_efficiency_reward':
            from . import math_dapo_efficiency
            res = math_dapo_efficiency.compute_score(solution_str, solution_str_with_special_tokens, ground_truth)
        elif extra_info['reward_method'] == 'accuracy_add_efficiency_reward_decay':
            from . import math_dapo_efficiency_decay
            res = math_dapo_efficiency_decay.compute_score(solution_str, solution_str_with_special_tokens, ground_truth, extra_info['global_steps'])
        elif extra_info['reward_method'] == 'accuracy_parallel_interv_reward':
            from . import math_dapo_acc_parallel_interved
            res = math_dapo_acc_parallel_interved.compute_score(solution_str, solution_str_with_special_tokens, ground_truth, extra_info['global_steps'])
        # elif extra_info['reward_method'] == 'accuracy_add_diversity_reward':
        #     from . import math_dapo_efficiency
        #     res = math_dapo_diversity.compute_score(solution_str, solution_str_with_special_tokens, ground_truth)
        # elif extra_info['reward_method'] == 'accuracy_add_efficiency_diversity_reward':
        #     from . import math_dapo_efficiency_diversity
        #     res = math_dapo_efficiency_diversity.compute_score(solution_str, solution_str_with_special_tokens, ground_truth)
        elif extra_info['reward_method'] == 'format_times_accuracy_times_parallel_reward':
            from . import math_dapo_accuracy_times_format_times_parallel
            res = math_dapo_accuracy_times_format_times_parallel.compute_score(solution_str, solution_str_with_special_tokens, ground_truth)
        elif extra_info['reward_method'] == 'accuracy_parallel_reward':
            from . import math_dapo_accuracy_parallel
            res = math_dapo_accuracy_parallel.compute_score(solution_str, solution_str_with_special_tokens, ground_truth)
        elif extra_info['reward_method'] == 'accuracy_times_parallel_reward':
            from . import math_dapo_accuracy_times_parallel
            res = math_dapo_accuracy_times_parallel.compute_score(solution_str, solution_str_with_special_tokens, ground_truth)
        elif extra_info['reward_method'] == 'accuracy_add_parallel_reward':
            from . import math_dapo_accuracy_add_parallel
            res = math_dapo_accuracy_add_parallel.compute_score(solution_str, solution_str_with_special_tokens, ground_truth)
        elif extra_info['reward_method'] == 'accuracy_add_parallel_add_diversity_reward':
            from . import math_dapo_accuracy_add_parallel_add_diversity
            res = math_dapo_accuracy_add_parallel_add_diversity.compute_score(solution_str, solution_str_with_special_tokens, ground_truth)
        elif extra_info['reward_method'] == 'accuracy_add_diversity_reward':
            from . import math_dapo_accuracy_add_diversity
            res = math_dapo_accuracy_add_diversity.compute_score(solution_str, solution_str_with_special_tokens, ground_truth)
        elif extra_info['reward_method'] == 'emphase_acc':
            from . import math_dapo_emphase_acc
            print("reward_func:", "emphase_acc reward")
            res = math_dapo_emphase_acc.compute_score(solution_str, solution_str_with_special_tokens, ground_truth)
        elif extra_info['reward_method'] == 'emphase_parallel':
            from . import math_dapo_emphase_parallel
            print("reward_func:", "emphase_parallel reward")
            res = math_dapo_emphase_parallel.compute_score(solution_str, solution_str_with_special_tokens, ground_truth)
        elif extra_info['reward_method'] == 'emphase_parallel_enhance':
            from . import math_dapo_emphase_parallel_enhance
            print("reward_func:", "emphase_parallel_enhance reward")
            res = math_dapo_emphase_parallel_enhance.compute_score(solution_str, solution_str_with_special_tokens, ground_truth)
        elif extra_info['reward_method'] == 'emphase_parallel_add_diversity':
            from . import math_dapo_emphase_parallel_add_diversity
            print("reward_func:", "emphase_parallel_add_diversity reward")
            res = math_dapo_emphase_parallel_add_diversity.compute_score(solution_str, solution_str_with_special_tokens, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
    )


__all__ = ["default_compute_score"]
