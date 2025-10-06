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
import asyncio
import json
import logging
import os
from typing import Any
from uuid import uuid4
import random
from verl.parallel_thinking_generation_v3.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.rollout_trace import rollout_trace_op
import copy
from contextlib import contextmanager
from typing import Dict, Optional, Type
from codetiming import Timer
import torch
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last

def test_mask(mask: torch.Tensor, position_ids: torch.Tensor, start: int, end: int) -> bool:
    """Test if the mask is valid for the given position range."""
    if start < 0 or end > mask.size(0):
        return False
    return mask[start:end, start:end].all().item()



@register("parallel_thinking_agent_v3")
class ParallelThinkingAgentLoopV3(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ParallelThinkingV3AgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.add_diverse_prefix = config.actor_rollout_ref.rollout.agent.add_diverse_prefix
        cls.max_iterations_for_parallel_thinking = config.actor_rollout_ref.rollout.agent.max_iterations_for_parallel_thinking
        cls.num_paths = config.actor_rollout_ref.rollout.agent.num_paths
        cls.max_path_response_length = config.actor_rollout_ref.rollout.agent.max_path_response_length

        cls.eos_token_id = cls.tokenizer.eos_token_id
        cls.start_parallel_token = cls.tokenizer.encode('<Parallel>')[0]
        cls.end_parallel_token = cls.tokenizer.encode('</Parallel>')[0]
        cls.start_path_token = cls.tokenizer.encode('<Path>')[0]
        cls.end_path_token = cls.tokenizer.encode('</Path>')[0]
        cls.start_summary_token = cls.tokenizer.encode('<Summary>')[0]
        cls.end_summary_token = cls.tokenizer.encode('</Summary>')[0]
        cls.new_line_token = cls.tokenizer.encode('\n')

        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=True)

    @rollout_trace_op
    async def run(
        self,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
    ) -> AgentLoopOutput:

        
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            ),
        )
        init_len      = len(prompt_ids)                         
        position_ids  = torch.arange(init_len, dtype=torch.long)

        response_mask = []                                     
        iterations    = 0
        request_id    = uuid4().hex

        left_pad_len = self.prompt_length - init_len
        if left_pad_len < 0:
            print(f"Warning: prompt length {self.prompt_length} is less than initial prompt length {init_len}, truncating prompt.")
            left_pad_len = 0

        position_required_masks = []

        
        parallel_stack: list[dict] = []

        
        def append_tokens(new_ids: list[int],
                        is_parallel: bool = False,
                        path_spans: list[tuple[int, int]] | None = None, manual_mask_positions= None):
            
            nonlocal prompt_ids, position_ids, parallel_stack, position_required_masks, left_pad_len

            start = len(prompt_ids) # correct
            prompt_ids.extend(new_ids) # correct -> merge new_ids into prompt_ids
            response_mask.extend([1] * len(new_ids)) # correct -> extend response_mask with 1s for new_ids
            # masks = [1] * len(new_ids)
            # if manual_mask_positions:
            #     for idx in manual_mask_positions:
            #         if 0 <= idx < len(masks):
            #             masks[idx] = 0
            # response_mask.extend(masks)
            # first generate position_ids with same length as new_ids if not is_parallel just append or we will process it later
            incr = torch.arange(1, len(new_ids) + 1, dtype=torch.long) + position_ids[-1] # -> correct
            position_ids = torch.cat([position_ids, incr]) # correct -> extend position_ids with new_ids' positions

            if not is_parallel:
                return

            blk     = parallel_stack.pop()
            base_p  = blk["base"]
            longest = blk["longest"]
            

            # a) 校正 Path 内位置 ＆ b) 写跨 Path 掩码
            for i, (s_rel, e_rel) in enumerate(path_spans):
                
                # assert new_ids[s_rel] == self.start_path_token, \
                #     f"append_tokens(is_parallel=True) 期望新 token {new_ids[s_rel]} 是 {self.start_path_token}，但实际是 {new_ids[s_rel]}"
                # assert new_ids[e_rel-1] == self.end_path_token, \
                #     f"append_tokens(is_parallel=True) 期望新 token {new_ids[e_rel-1]} 是 {self.end_path_token}，但实际是 {new_ids[e_rel]}"
                s_abs, e_abs = start + s_rel, start + e_rel
                # the first path we do not need to shift
                shift = position_ids[s_abs] - (base_p)
                # print("shift", shift.item())
                if shift.item():
                    position_ids[s_abs:e_abs] -= shift
                # assert position_ids[s_abs] == base_p, \
                #     f"append_tokens(is_parallel=True) 期望 position_ids[s_abs] {position_ids[s_abs]} 是 {base_p}，但实际是 {position_ids[s_abs]}"
                # # print("prompt_ids[s_abs] == self.start_path_token", prompt_ids[s_abs], self.start_path_token)
                # assert prompt_ids[s_abs] == self.start_path_token, \
                #     f"append_tokens(is_parallel=True) 期望 prompt_ids[s_abs] {prompt_ids[s_abs]} 是 {self.start_path_token}，但实际是 {prompt_ids[s_abs]}"
                # # print("prompt_ids[e_abs - 1] == self.end_path_token", prompt_ids[e_abs - 1], self.end_path_token)
                # assert prompt_ids[e_abs - 1] == self.end_path_token, \
                #     f"append_tokens(is_parallel=True) 期望 prompt_ids[e_abs - 1] {prompt_ids[e_abs - 1]} 是 {self.end_path_token}，但实际是 {prompt_ids[e_abs - 1]}"
                path_max_pos = position_ids[e_abs - 1]  
                longest = max(longest, (path_max_pos - base_p).item())
                # longest = max(longest, e_rel - 1)
                for j, (sj_rel, ej_rel) in enumerate(path_spans):
                    if j == i: continue
                    sj_abs, ej_abs = left_pad_len + start + sj_rel, left_pad_len + start + ej_rel
                    
                    if e_abs + left_pad_len > ej_abs:
                        if ej_abs < self.prompt_length + self.response_length:
                            position_required_masks.append((left_pad_len, s_abs + left_pad_len, e_abs + left_pad_len, sj_abs, ej_abs))
                            print(f"mask {s_abs+left_pad_len}:{e_abs+left_pad_len} 与 {sj_abs}:{ej_abs} attention")
                    elif e_abs + left_pad_len < ej_abs:
                        if sj_abs < self.prompt_length + self.response_length:
                            position_required_masks.append((left_pad_len, s_abs + left_pad_len, e_abs + left_pad_len, sj_abs, ej_abs))
                            print(f"mask {s_abs+left_pad_len}:{e_abs+left_pad_len} 与 {sj_abs}:{ej_abs} attention")
                    

            # try:
            #     p_rel = new_ids.index(self.end_parallel_token)   # first find relative pos of </Parallel> in  new_ids 
            # except ValueError:
            #     raise RuntimeError("append_tokens(is_parallel=True) 找不到 </Parallel> 标记")
            s2, e2 = path_spans[1]

            p_end_abs   = start + e2
            
            desired_end = base_p + longest + 1
            shift2      = position_ids[p_end_abs] - desired_end
            if shift2.item():
                position_ids[p_end_abs:] -= shift2
            # print("prompt_ids[p_end_abs], self.end_parallel_token", prompt_ids[p_end_abs], self.end_parallel_token)
            # print("position_ids[p_end_abs], desired_end", position_ids[p_end_abs], desired_end)
            # print(position_ids[p_end_abs], position_ids[p_end_abs-1])
            # assert prompt_ids[p_end_abs] == self.end_parallel_token, \
            #     f"append_tokens(is_parallel=True) 期望 prompt_ids[p_end_abs] {prompt_ids[p_end_abs]} 是 {self.end_parallel_token}，但实际是 {prompt_ids[p_end_abs]}"
            # # print(position_ids)
            # print("prompt_ids",prompt_ids)
            # assert torch.all(position_ids[1:] >= position_ids[:-1]), "position_ids 非递增"
            # check if position_ids is strictly increasing
            # s1, e1 = path_spans[0]
            # s2, e2 = path_spans[1]
            # assert position_ids[start+s1] == base_p, \
            #     f"append_tokens(is_parallel=True) 期望 position_ids[start+s1] {position_ids[start+s1]} 是 {base_p}，但实际是 {position_ids[start+s1]}"
            # assert position_ids[start+s2] == base_p, \
            #     f"append_tokens(is_parallel=True) 期望 position_ids[start+s2] {position_ids[start+s2]} 是 {base_p}，但实际是 {position_ids[start+s2]}"
            
            # print("position_ids[start+s1:start+e1]", position_ids[start+s1:start+e1])
            # print("position_ids[start+s2:start+e2]", position_ids[start+s2:start+e2])
            # print("position_ids[start+e1-1]", position_ids[start+e1-1])
            # print("position_ids[start+e1-1+1]", position_ids[start+e1-1+1])
            # print("position_ids[start+e2-1]", position_ids[start+e2-1])
            # print("position_ids[start+e2-1+1]", position_ids[start+e2-1+1])
            
            # print(position_ids[start+s1:start+e1])
            # print(position_ids[start+s2:start+e2])
            # print(position_ids[start+e1-1])
            # print(position_ids[start+e1-1+1])
            # print(position_ids[start+e2-1])
            # print(position_ids[start+e2-1+1])
            # print(position_ids[start+e2] - position_ids[start+e2-1])
            # if position_ids[start+e2-1] > position_ids[start+e1-1]:
            #     if start+e2-1 > start+e1-1:
            #         assert (position_ids[start+e2] - position_ids[start+e2-1]).item() ==  1, \
            #             f"append_tokens(is_parallel=True) 期望 position_ids[start+e2] - position_ids[start+e2-1] {position_ids[start+e2] - position_ids[start+e2-1]} 是 1，但实际是 {position_ids[start+e2] - position_ids[start+e2-1]}, {position_ids[start+s1:start+e1]}, {position_ids[start+s2:start+e2]}, {self.tokenizer.decode(prompt_ids[start+s1:start+e1], skip_special_tokens=False)}, {self.tokenizer.decode(prompt_ids[start+s2:start+e2], skip_special_tokens=False)},{base_p}, {position_ids}"
            #     elif position_ids[start+e1-1] > position_ids[start+e2-1]:
            #         assert position_ids[start+e1] - position_ids[start+e1-1] == 1, \
            #             f"append_tokens(is_parallel=True) 期望 position_ids[start+e1] - position_ids[start+e1-1] {position_ids[start+e1] - position_ids[start+e1-1]} 是 1，但实际是 {position_ids[start+e1] - position_ids[start+e1-1]}"
            # elif position_ids[start+e1-1] > position_ids[start+e2-1]:
            #     if start+e1-1 > start+e2-1:
            #         assert position_ids[start+e1] - position_ids[start+e1-1] ==  1, \
            #             f"append_tokens(is_parallel=True) 期望 position_ids[start+e1] - position_ids[start+e1-1] {position_ids[start+e1] - position_ids[start+e1-1]} 是 1，但实际是 {position_ids[start+e1] - position_ids[start+e1-1]}"
            #     elif position_ids[start+e2-1] > position_ids[start+e1-1]:
            #         assert position_ids[start+e2] - position_ids[start+e2-1] ==  1, \
            #             f"append_tokens(is_parallel=True) 期望 position_ids[start+e2] - position_ids[start+e2-1] {position_ids[start+e2] - position_ids[start+e2-1]} 是 1，但实际是 {position_ids[start+e2] - position_ids[start+e2-1]}"
           
        def should_stop() -> bool:
            gen_len = len(position_ids) - init_len
            if gen_len >= self.response_length:
                return True
            if (
                self.max_iterations_for_parallel_thinking
                and iterations >= self.max_iterations_for_parallel_thinking
            ):
                return True
            return False

        
        while True:
            sp_main = {**sampling_params, "stop_token_ids": [self.start_parallel_token, self.eos_token_id]}
            ids = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sp_main,
            )
            append_tokens(ids)
            if should_stop() or not await self.check_parallel(ids):
                break

            assert ids[-1] != self.eos_token_id

            # elict parallel thinking 
            base_pos = position_ids[-1].item() + 1 # the position of the <Path> token
            parallel_stack.append({"base": base_pos, "longest": 0})

            parallel_ids, path_spans, manual_mask_positions = await self._call_parallel_thinking(
                prompt_ids, sampling_params
            )
            append_tokens(parallel_ids, is_parallel=True, path_spans=path_spans, manual_mask_positions=manual_mask_positions)

            iterations += 1
            if should_stop():
                break

        response_ids = prompt_ids[-len(response_mask) :]
        prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]
        assert init_len == (len(prompt_ids))
        
        if left_pad_len > 0:
            position_ids = torch.cat([
                torch.zeros(left_pad_len, dtype=torch.long),
                position_ids
            ])
        if len(response_ids) < self.response_length:
            right_pad_len = self.response_length - len(response_ids)
            position_ids = torch.cat([
                position_ids,
                torch.zeros(right_pad_len, dtype=torch.long)])
            # bool_mask[:, -right_pad_len:] = False
        else:
            position_ids = position_ids[:self.prompt_length + self.response_length]
            response_ids = response_ids[: self.response_length]
            response_mask = response_mask[: self.response_length]

        return AgentLoopOutput(
            prompt_ids          = prompt_ids,
            response_ids        = response_ids,
            response_mask       = response_mask,
            position_required_mask       = position_required_masks,
            multiverse_pos_ids  = position_ids.cpu(),        # 1-D
            num_turns           = iterations + 1,
            metrics             = {},
        )



    async def check_parallel(self, response_ids):
        """
        check whether need to conduct parallel thinking
        """
        if response_ids[-1] == self.start_parallel_token:
            return True
        else:
            return False

    async def _call_parallel_thinking(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
    ):
        print("Conducting parallel thinking...")
        num_paths = self.num_paths
        max_len   = self.max_path_response_length
        PATH_OPEN, PATH_CLOSE = self.start_path_token, self.end_path_token
        manual_mask_positions = []

        
        async def _gen_single_path(seed_i: int, prompt_i: list[int]):
            sp = copy.deepcopy(sampling_params)
            sp.update({"seed": seed_i, "n": 1, "stop_token_ids": [PATH_CLOSE, self.eos_token_id],
                    "temperature": 1.0})
            # sp.update({"n": 1, "stop_token_ids": [PATH_CLOSE, self.eos_token_id],
            #         "temperature": 1.0})
            manual_append = False
            ids = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_i + [PATH_OPEN],
                sampling_params=sp,
            )
            if max_len and len(ids) > max_len:
                ids = ids[:max_len]
            
            if ids[-1] != PATH_CLOSE:
                if ids[-1] == self.eos_token_id:
                    ids[-1] = PATH_CLOSE
                else:
                    ids.append(PATH_CLOSE)
                manual_append = True
            return [PATH_OPEN] + ids, manual_append

        tasks = []
        for i in range(num_paths):
            # prompt_i **不再含 PATH_OPEN**，防止 tag 重复
            prompt_i = prompt_ids
            tasks.append(
                asyncio.create_task(
                    _gen_single_path(random.randint(0, 2**31 - 1), prompt_i)
                )
            )

        path_token_lists = await asyncio.gather(*tasks)     

        parallel_ids = []             
        path_spans   = []
        cursor       = 0              

        #### <Path> </Path><Path> </Path> -> path_spans (0, len(path_1)) (len(path_1), len(path_1) + len(path_2))
        for (ids, manual_append_flag) in path_token_lists:
            assert ids[0] == PATH_OPEN and ids[-1] == PATH_CLOSE, \
                f"Path tokens must start with {PATH_OPEN} and end with {PATH_CLOSE}, got {ids}"
            span_start = cursor 
            parallel_ids.extend(ids)
            cursor     += len(ids)
            span_end   = cursor
            path_spans.append((span_start, span_end))
            manual_mask_positions.append(span_start)
            if manual_append_flag:
                manual_mask_positions.append(span_end-1)

        # ---------- 3) add </Parallel><Summary> ----------
        parallel_ids.append(self.end_parallel_token)           # </Parallel>
        manual_mask_positions.append(cursor)
        assert parallel_ids[cursor] == self.end_parallel_token
        cursor += 1
        parallel_ids.extend(self.new_line_token)          # \n
        manual_mask_positions.append(cursor)
        cursor += len(self.new_line_token)
        parallel_ids.append(self.start_summary_token)          # <Summary>
        manual_mask_positions.append(cursor)
        assert parallel_ids[cursor] == self.start_summary_token
        cursor += 1
        
        # if exploration stage alos superass the max length, we will not generate summary
        if len(prompt_ids) + 2 + len(parallel_ids) + 1 < self.prompt_length + self.response_length:
            print("Conducting Summary...")
            manual_append_summary = False
            sp_sum = copy.deepcopy(sampling_params)
            sp_sum.update({"n": 1, "stop_token_ids": [self.end_summary_token, self.eos_token_id]})
            summary_ids = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids + parallel_ids,   # 现在的完整 prompt
                sampling_params=sp_sum,
            )
            if summary_ids[-1] != self.end_summary_token:
                if summary_ids[-1] == self.eos_token_id:
                    summary_ids[-1] = self.end_summary_token
                else:
                    summary_ids.append(self.end_summary_token)
                manual_append_summary = True
            parallel_ids.extend(summary_ids)
            cursor += len(summary_ids)
            if manual_append_summary:
                manual_mask_positions.append(cursor-1)
            # print("asadas", parallel_ids[cursor-1])
            assert parallel_ids[cursor-1] == self.end_summary_token
            assert parallel_ids[-1] == self.end_summary_token, \
                f"Parallel thinking did not end with {self.end_summary_token}, got {parallel_ids[-1]}"
        # print(self.tokenizer.decode(parallel_ids, skip_special_tokens=False))
        return parallel_ids, path_spans, manual_mask_positions        # path_spans 已是相对 parallel_ids       
        

