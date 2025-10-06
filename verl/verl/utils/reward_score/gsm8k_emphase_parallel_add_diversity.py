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

import re
import sacrebleu   # 若用 nltk，则替换成 nltk.translate.bleu_score.sentence_bleu
import re
from typing import Optional
import math


import re
from itertools import combinations
from typing import List
# 提取单个 <Parallel> 块中的 <Path> 文本
def extract_paths(par_block: str) -> List[str]:
    return re.findall(r"<Path>(.*?)</Path>", par_block, flags=re.S)

# 计算一个并行块的 self-BLEU（平均两两 BLEU）
def self_bleu(paths: List[str]) -> float:
    if len(paths) < 2:
        return 1.0        # 只有 0/1 条路径 → 完全“同质”
    bleu_scores = []
    for p, q in combinations(paths, 2):
        # sacrebleu 需要 ref=list[str], hyp=str
        bleu = sacrebleu.sentence_bleu(p, [q]).score / 100.0  # ∈[0,1]
        bleu_scores.append(bleu)
    return sum(bleu_scores) / len(bleu_scores)

def avg_self_bleu(text_with_tags: str) -> float:
    par_blocks = re.findall(r"<Parallel>(.*?)</Parallel>", text_with_tags, flags=re.S)
    if not par_blocks:
        return 1.0  # 没有并行就视为完全同质
    block_bleus = [self_bleu(extract_paths(block)) for block in par_blocks]
    return sum(block_bleus) / len(block_bleus)

def extract_solution1(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    if method == "strict":
        # this also tests the formatting of the model
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(0)
            final_answer = final_answer.split("#### ")[1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer


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


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    if method == "strict":
        # 严格模式：只匹配 "Final Answer: 数字"
        solution = re.search(r"Final Answer:\s*([-]?[0-9\.\,]+)", solution_str)
        if solution is None:
            final_answer = None
        else:
            # 取第 1 个分组，去掉逗号和美元符号
            final_answer = solution.group(1).replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall(r"(-?[0-9\.\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # 没有数字则返回 None
            pass
        else:
            invalid_str = ["", "."]
            # 从后往前找第一个不是无效串的数字
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    final_answer = final_answer.replace(",", "").replace("$", "")
                    break

    return final_answer

def compute_score(solution_str, solution_str_with_special_tokens, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, method=method)
    is_parallel = 1.0 if "<Parallel>" in solution_str_with_special_tokens else 0.0
    is_format_correct = 1 if len(check_parallel_thinking_format(solution_str_with_special_tokens)) ==0  else 0.0
    avg_bleu = avg_self_bleu(solution_str_with_special_tokens)
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            if is_parallel == 1.0:
                if is_format_correct:
                    return 1.2 - 0.2 * avg_bleu
                else:
                    return 0.0
            elif is_parallel == 0.0:
                return 1.0
        else:
            return format_score
