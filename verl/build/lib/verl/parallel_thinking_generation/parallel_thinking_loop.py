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

from verl.parallel_thinking_generation.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.rollout_trace import rollout_trace_op
import copy
from contextlib import contextmanager
from typing import Dict, Optional, Type
from codetiming import Timer
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

@register("parallel_thinking_agent")
class ParallelThinkingAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ToolAgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
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

        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template([{}], add_generation_prompt=False, tokenize=True)

    @rollout_trace_op
    async def run(self, messages: list[dict[str, Any]], sampling_params: dict[str, Any]) -> AgentLoopOutput:
        metrics = {}
        request_id = uuid4().hex
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            ),
        )
        response_mask = []
        init_len = len(prompt_ids)
        iterations = 0
        while True:
            if len(prompt_ids) + 1 >= self.prompt_length + self.response_length:
                break
            sp_main = sampling_params.copy()
            # print(sp_main)                      
            sp_main['stop_token_ids'] = [self.start_parallel_token]  
            
            
            with _timer("generate_sequences", metrics):
                response_ids = await self.server_manager.generate(
                    request_id=request_id, prompt_ids=prompt_ids, sampling_params=sp_main
                )
            prompt_ids += response_ids
            response_mask += [1] * len(response_ids)
            iterations += 1

            # reach max response length
            if len(response_mask) >= self.response_length:
                break

            # reach max iterations
            if self.max_iterations_for_parallel_thinking and iterations >= self.max_iterations_for_parallel_thinking:
                break

            # no parallel thinking
            have_parallel_thinkings = await self.check_parallel(response_ids)
            if not have_parallel_thinkings:
                break

            # conduct parallel thinking
            parallel_ids = await self._call_parallel_thinking(
                prompt_ids, sampling_params
            )
            prompt_ids += parallel_ids
            response_mask += [1] * len(parallel_ids)
            iterations += 1

            print(self.tokenizer.decode(
                prompt_ids, skip_special_tokens=False
            ))

        response_ids = prompt_ids[-len(response_mask) :]
        prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]
        assert init_len == (len(prompt_ids))
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            num_turns=iterations + 1,
            metrics=metrics,
        )
        return output


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
        prompt_ids,
        sampling_params,
    ):
        """
        generate multiple <Path>…</Path> in parallel and then summary these paths，
        Return:
            {
                "path_0": "<Path> … </Path>",
                "path_1": "<Path> … </Path>",
                ...
                "summary": "<Summary> … </Summary>"
            }
        """
        # ========= 0. 预处理 & 参数 =========
        num_paths = self.num_paths    
        truncate_len = self.max_path_response_length
        metrics = {}
        working_ids = prompt_ids.copy()
        prpmpt_length = len(working_ids)

        sp_paths = copy.deepcopy(sampling_params)
        sp_paths['n'] = 1   
        sp_paths["temperature"] = 1.0        # 确保非0                     
        sp_paths['stop_token_ids'] = [self.end_path_token]      

        PATH_OPEN = self.start_path_token
        PATH_CLOSE = self.end_path_token

        # prompt_with_path = prompt_ids + [PATH_OPEN]
        async def _gen_single_path(prompt_ids, sampling_params):
            ids = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids + [self.start_path_token],
                sampling_params=sampling_params,
            )
            if self.max_path_response_length and len(ids) > self.max_path_response_length:
                ids = ids[: self.max_path_response_length]
            if ids[-1] != self.end_path_token:
                ids += [self.end_path_token]
            return ids
        if len(working_ids) + 2 < self.prompt_length + self.response_length:
            tasks = [asyncio.create_task(
                        _gen_single_path(working_ids, sp_paths)
                    ) for _ in range(num_paths)]
            response_ids_batch = await asyncio.gather(*tasks)   # List[List[int]]

            if response_ids_batch and isinstance(response_ids_batch[0], int):
                response_ids_batch = [response_ids_batch]

            # paths_dict = {}
            paths_token_concat = []
            for i, ids in enumerate(response_ids_batch):
                if truncate_len and len(ids) > truncate_len:
                    ids = ids[:truncate_len]

                if ids[-1] != PATH_CLOSE:
                    full_ids = [PATH_OPEN] + ids + [PATH_CLOSE]
                else:
                    full_ids = [PATH_OPEN] + ids
                # paths_dict[f"path_{i}"] = txt
                paths_token_concat += full_ids
            working_ids += paths_token_concat
            working_ids += [self.end_parallel_token] 
            working_ids += [self.start_summary_token]  # "<Summary>"
        
        if len(working_ids) + 1 < self.prompt_length + self.response_length:
            sp_sum = copy.deepcopy(sampling_params)
            sp_sum['n'] = 1
            sp_sum['stop_token_ids'] = [self.end_summary_token]           

            req_sum = uuid4().hex
            summary_ids = await self.server_manager.generate(
                request_id=req_sum,
                prompt_ids=working_ids,
                sampling_params=sp_sum,
            )
        
            if summary_ids and isinstance(summary_ids[0], list):
                summary_ids = summary_ids[0]

            summary_full = (
                summary_ids + [self.end_summary_token] if summary_ids[-1] != self.end_summary_token else summary_ids
            )
            # summary_txt = self.tokenizer.decode(
            #     summary_full, skip_special_tokens=False
            # )
            # paths_dict["summary"] = summary_txt

            working_ids += summary_full
        
        # print(paths_dict)
        return working_ids[prpmpt_length:]
    
    
