# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import re
from typing import Optional
import math


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string.

    Args:
        s: String with format "\\boxed{content}"

    Returns:
        The content inside the boxed command
    """
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]


# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.

    Args:
        final_answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    final_answer = final_answer.split("=")[-1]

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def is_correct_minerva(
    solution_str: str, gt: str, gt_need_extract: bool = False, answer_pattern: str = r"(?i)Final Answer\s*:\s*([^\n]+)"
) -> tuple[bool, str]:
    """Check if the solution is correct according to Minerva criteria.

    Args:
        solution_str: The solution string to check
        gt: The ground truth answer
        gt_need_extract: Whether the ground truth needs extraction
        answer_pattern: Regex pattern to extract the answer

    Returns:
        Tuple of (is_correct, normalized_prediction)
    """
    # Extract answer from solution
    match = re.findall(answer_pattern, solution_str)
    extracted_answer = match[-1] if match else "[INVALID]"
    pred = normalize_final_answer(extracted_answer)

    # Process ground truth
    if gt_need_extract:
        gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    else:
        gt = normalize_final_answer(gt)

    return (pred == gt), pred



def check_parallel_thinking_format(text):
    tag_pattern = r'</?Parallel>|</?Path>|</?Summary>'
    tokens = re.findall(tag_pattern, text)
    
    stack = []
    errors = []
    summary_expected = False
    summary_count = 0
    parallel_path_counter = []  # stack aligned, count paths per parallel block
    
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        if token == '<Parallel>':
            stack.append('Parallel')
            parallel_path_counter.append(0)  # init path counter for this block
            summary_expected = False

        elif token == '<Path>':
            stack.append('Path')

        elif token == '</Path>':
            if not stack or stack[-1] != 'Path':
                errors.append(f'Unexpected </Path> at token #{i}')
            else:
                stack.pop()
                # increment path count for current parallel
                if parallel_path_counter:
                    parallel_path_counter[-1] += 1

        elif token == '</Parallel>':
            if not stack or stack[-1] != 'Parallel':
                errors.append(f'Unexpected </Parallel> at token #{i}')
            else:
                stack.pop()
                count = parallel_path_counter.pop()
                if count < 2:
                    errors.append(f'<Parallel> block closed at #{i} with only {count} <Path> blocks')
                summary_expected = True

        elif token == '<Summary>':
            if not summary_expected:
                errors.append(f'<Summary> at token #{i} not following a </Parallel>')
            summary_expected = False
            summary_count += 1

        elif token == '</Summary>':
            pass  # no stack logic

        i += 1

    # Final checks
    if stack:
        errors.append(f'Stack not empty at end: {stack}')
    if summary_expected:
        errors.append(f'Missing <Summary> after last </Parallel>')
    if summary_count != text.count('</Parallel>'):
        errors.append(f'Mismatch: {summary_count} <Summary>, {text.count("</Parallel>")} </Parallel>')

    return errors


def is_correct_strict_box(
    pred: str, gt: str, pause_tokens_index: Optional[list[int]] = None
) -> tuple[int, Optional[str]]:
    """Check if the prediction is correct using strict boxed answer criteria.

    Args:
        pred: The prediction string
        gt: The ground truth answer
        pause_tokens_index: Indices of pause tokens

    Returns:
        Tuple of (score, extracted_prediction)
    """
    # Extract the relevant part of the prediction
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[pause_tokens_index[-1] - 100 :]
    else:
        pred = pred[-100:]

    # Extract and check the boxed answer
    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    return 1 if (extracted_pred == gt) else -1, extracted_pred


def verify(
    solution_str: str, answer: str, strict_box_verify: bool = False, pause_tokens_index: Optional[list[int]] = None
) -> bool:
    """Verify if the solution is correct.

    Args:
        solution_str: The solution string to verify
        answer: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        True if the solution is correct, False otherwise
    """
    if strict_box_verify:
        correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
        return correct == 1, pred

    correct, pred = is_correct_minerva(solution_str, answer)
    return correct, pred


def _strip_tags(text: str) -> str:
    return re.sub(r'</?Parallel>|</?Path>|</?Summary>', '', text)

def _simple_token_count(text: str) -> int:
    return len(re.findall(r'\S+', text))

def _iter_parallel_blocks(text: str):
    for m in re.finditer(r'<Parallel>(.*?)</Parallel>', text, flags=re.DOTALL):
        yield m.group(0), m.group(1)

def _extract_paths_and_summary(inner_text: str):
    paths = re.findall(r'<Path>(.*?)</Path>', inner_text, flags=re.DOTALL)
    summary = re.findall(r'<Summary>(.*?)</Summary>', inner_text, flags=re.DOTALL)
    summary_text = summary[-1] if summary else ''
    return paths, summary_text

def compute_efficiency_gain(sol_with_tags: str, path_mode: str = "max") -> dict:
    """计算 efficiency gain: (total - traversed)/total"""
    total_tokens = _simple_token_count(_strip_tags(sol_with_tags))
    blocks = list(_iter_parallel_blocks(sol_with_tags))

    # outside tokens
    outside_text = re.sub(r'<Parallel>.*?</Parallel>', '', sol_with_tags, flags=re.DOTALL)
    outside_tokens = _simple_token_count(_strip_tags(outside_text))

    traversed_tokens = outside_tokens

    for block_full, inner in blocks:
        paths, summary_text = _extract_paths_and_summary(inner)
        if not paths:
            traversed_tokens += _simple_token_count(_strip_tags(block_full))
            continue

        path_lens = [_simple_token_count(_strip_tags(p)) for p in paths]

        if path_mode == "min":
            path_len = min(path_lens)
        elif path_mode == "max":
            path_len = max(path_lens)
        elif path_mode == "avg":
            path_len = int(sum(path_lens) / len(path_lens))
        else:
            path_len = max(path_lens)

        summary_tokens = _simple_token_count(_strip_tags(summary_text))
        traversed_tokens += (path_len + summary_tokens)

    eff_gain = 0.0
    if total_tokens > 0:
        eff_gain = max((total_tokens - traversed_tokens) / total_tokens, 0.0)

    return {
        "eff_gain": eff_gain,
        "total_tokens": total_tokens,
        "traversed_tokens": traversed_tokens,
        "num_parallel": len(blocks),
    }
# ==================================================


def reward_func(correct, eff_gain=0.0,
                alpha_acc=1.0,
                alpha_eff=0.4,
                alpha_explore=0.1):
    """
    只考虑 acc + efficiency 的奖励函数
    - 正确: R = +1 + alpha_eff * eff_gain
    - 错误: R = -1 + alpha_explore * eff_gain   (探索奖励，防止模型 early collapse)
    """
    if correct:
        return alpha_acc + alpha_eff * eff_gain
    else:
        return -alpha_acc + alpha_explore * eff_gain

def compute_score(
    solution_str: str,
    solution_str_with_special_tokens: str, 
    ground_truth: str,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
) -> float:
    # Limit solution length for efficiency
    solution_str = solution_str[-300:]

    correct, pred = verify(solution_str, ground_truth, strict_box_verify, pause_tokens_index)

    num_parallel = solution_str_with_special_tokens.count("<Parallel>")
    format_correct = 1 if len(check_parallel_thinking_format(solution_str_with_special_tokens))==0 else 0

    # ✅ 新增 efficiency 计算
    eff_info = compute_efficiency_gain(solution_str_with_special_tokens, path_mode="max")
    eff_gain = eff_info["eff_gain"]

    reward = reward_func(correct, eff_gain)

    return {
        "score": reward,
        "acc": correct,
        "pred": pred,
        "eff_gain": eff_gain,
        "num_parallel": num_parallel,
        "total_tokens": eff_info["total_tokens"],
        "traversed_tokens": eff_info["traversed_tokens"],
    }


# def compute_score(
#     solution_str: str,
#     solution_str_with_special_tokens: str, 
#     ground_truth: str,
#     strict_box_verify: bool = False,
#     pause_tokens_index: Optional[list[int]] = None,
# ) -> float:
#     """Compute the reward score for a solution.

#     Args:
#         solution_str: The solution string
#         ground_truth: The ground truth answer
#         strict_box_verify: Whether to use strict box verification
#         pause_tokens_index: Indices of pause tokens

#     Returns:
#         Reward score (1.0 for correct, -1.0 for incorrect)
#     """
#     # Limit solution length for efficiency
#     solution_str = solution_str[-300:]  # The longest answer in MATH-500 has 159 characters

#     # Verify the solution
#     correct, pred = verify(solution_str, ground_truth, strict_box_verify, pause_tokens_index)

#     # Count <Parallel> in solution_str_with_special_token
#     num_parallel = solution_str_with_special_tokens.count("<Parallel>")

#     format_correct = 1 if len(check_parallel_thinking_format(solution_str_with_special_tokens))==0 else 0

#     # reward = 1.0 if correct and num_parallel >= 1 and format_correct else -1.0

#     if correct and num_parallel == 0:
#         reward = 1.0
#     elif correct and num_parallel >= 1 and format_correct == 1:
#         reward = 1.0
#     else:
#         reward = -1.0

#     acc = correct

#     return {
#         "score": reward,
#         "acc": acc,
#         "pred": pred,
#     }

# if __name__ == "__main__":
#     # 一个示例输出：含两个 <Parallel> 块
#     solution_str = """
#     First reasoning part...
#     <Parallel>
#     <Path>Path A1 has about 10 tokens.</Path>
#     <Path>Path A2 is longer, maybe 20 tokens in total.</Path>
#     <Summary>Both paths show the same mid-result.</Summary>
#     </Parallel>

#     Continue reasoning...
#     <Parallel>
#     <Path>Path B1 has 15 tokens of reasoning.</Path>
#     <Path>Path B2 is longer, about 25 tokens of reasoning.</Path>
#     <Path>Path B3 is also long, around 25 tokens too.</Path>
#     <Summary>All paths lead to the final result.</Summary>
#     </Parallel>

#     Final Answer: 42
#     """

#     ground_truth = "40"
#     solution_str_with_special_tokens = solution_str

#     result = compute_score(
#         solution_str=solution_str,
#         solution_str_with_special_tokens=solution_str_with_special_tokens,
#         ground_truth=ground_truth,
#         strict_box_verify=False,
#     )

#     print("==== Test Result (Multiple Parallel Blocks) ====")
#     for k, v in result.items():
#         print(f"{k}: {v}")
