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
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

from typing import List, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import numpy as np
import matplotlib.pyplot as plt
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


class ParallelThinkingSFTDataset(Dataset):
    """
    This is an in-memory SFTDataset

    Arguments:
        config (OmegaConf): the data config
    """

    def __init__(self, parquet_files: Union[str, List[str]], tokenizer, config):
        prompt_key = config.get("prompt_key", "prompt")
        prompt_dict_keys = config.get("prompt_dict_keys", None)
        response_key = config.get("response_key", "response")
        response_dict_keys = config.get("response_dict_keys", None)
        max_length = config.get("max_length", 1024)
        truncation = config.get("truncation", "error")
        use_shm = config.get('use_shm', False)

        assert truncation in ["error", "left", "right"]
        self.truncation = truncation
        self.use_shm = use_shm

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key if isinstance(prompt_key, (tuple, list)) else [prompt_key]
        self.response_key = response_key if isinstance(response_key, (tuple, list)) else [response_key]
        self.prompt_dict_keys = prompt_dict_keys if prompt_dict_keys else []
        self.response_dict_keys = response_dict_keys if response_dict_keys else []

        self.max_length = max_length

        self.start_parallel_token = self.tokenizer.encode('<Parallel>')[0]
        self.end_parallel_token = self.tokenizer.encode('</Parallel>')[0]
        self.start_path_token = self.tokenizer.encode('<Path>')[0]
        self.end_path_token = self.tokenizer.encode('</Path>')[0]
        self.start_summary_token = self.tokenizer.encode('<Summary>')[0]
        self.end_summary_token = self.tokenizer.encode('</Summary>')[0]

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        self.prompts = self.dataframe[self.prompt_key]
        for key in self.prompt_dict_keys:
            # type(x): pandas.core.series.Series
            # type(x[0]): numpy.ndarray
            # type(x[0][0]): dict
            print(key)
            try:
                self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            except Exception:
                print(f"self.prompts={self.prompts}")
                raise
        if isinstance(self.prompts, pd.DataFrame):
            self.prompts = self.prompts.squeeze()
        self.prompts = self.prompts.tolist()
        print(self.prompts)
        self.responses = self.dataframe[self.response_key]
        for key in self.response_dict_keys:
            try:
                self.responses = self.responses.apply(lambda x: series_to_item(x)[key], axis=1)  # noqa: B023
            except Exception:
                print(f"self.responses={self.responses}")
                raise
        if isinstance(self.responses, pd.DataFrame):
            self.responses = self.responses.squeeze()
        self.responses = self.responses.tolist()

    def generate_parallel_thinking_reasponse_mask(self, response_ids: torch.Tensor) -> torch.Tensor:
        """
        Generate a structure-aware causal mask for parallel thinking:
        - Normal causal mask before <Parallel>
        - Within <Path>: only attend to own path and prior context
        - Between <Path>s: mutually invisible
        - After </Parallel>: restore causal behavior until next <Parallel>

        Args:
            response_ids (torch.Tensor): (seq_len,) or (1, seq_len)

        Returns:
            torch.BoolTensor: (seq_len, seq_len) causal attention mask
        """
        if response_ids.dim() == 2:
            assert response_ids.size(0) == 1, "Only support single-sample batching for now"
            response_ids = response_ids[0]

        seq_len = response_ids.size(0)
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))

        path_ranges = []
        in_parallel = False
        i = 0
        while i < seq_len:
            tok = response_ids[i].item()

            if tok == self.start_parallel_token:
                in_parallel = True
                path_ranges = []

            elif in_parallel and tok == self.start_path_token:
                path_start = i
                while i < seq_len:
                    if response_ids[i].item() == self.end_path_token:
                        break
                    i += 1
                path_end = i
                path_ranges.append((path_start, path_end))

            elif tok == self.end_parallel_token:
                in_parallel = False
                for idx1, (s1, e1) in enumerate(path_ranges):
                    for idx2, (s2, e2) in enumerate(path_ranges):
                        if idx1 != idx2:
                            mask[s1:e1+1, s2:e2+1] = False
                path_ranges = []

            i += 1

        return mask
    def generate_multiverse_attention_mask(self, input_ids, device='cpu'):
        seq_len = len(input_ids)
        # Start with a lower triangular matrix (causal mask)
        bool_attention_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)) # Keep bool intermediate mask

        # Assuming single-token tags for simplicity based on original code
        # If tags can be multi-token, this conversion needs adjustment
        parallel_start_id = self.start_parallel_token
        parallel_end_id = self.end_parallel_token
        path_start_id =  self.start_path_token
        path_end_id = self.end_path_token
    
        structure_stack = []
        i = 0
        while i < seq_len:
            current_token_id = input_ids[i]

            # Check <Parallel> start
            if current_token_id == parallel_start_id:
                structure_stack.append({'type': 'parallel', 'start_marker_index': i, 'path_spans': []})
                i += 1
                continue

            # Check <Path> start
            elif current_token_id == path_start_id:
                structure_stack.append({'type': 'path', 'start_marker_index': i})
                i += 1
                continue

            # Check </Path> end
            elif current_token_id == path_end_id:
                path_end_marker_index = i + 1

                if not structure_stack or structure_stack[-1]['type'] != 'path':
                    raise ValueError(f"</Path> found at index {i} without a matching <Path> block on stack.")

                closed_path_block = structure_stack.pop()

                # Find the nearest enclosing parallel block to add this path span
                enclosing_parallel_block = None
                for block in reversed(structure_stack):
                    if block['type'] == 'parallel':
                        enclosing_parallel_block = block
                        break

                if enclosing_parallel_block is None:
                    raise ValueError(f"Path block ending at {i} is not enclosed within any <Parallel> block.")

                # Add the span including markers
                path_start_marker_index = closed_path_block['start_marker_index']
                if path_start_marker_index < path_end_marker_index:
                    enclosing_parallel_block['path_spans'].append((path_start_marker_index, path_end_marker_index))

                i = path_end_marker_index
                continue

            # Check </Parallel> end
            elif current_token_id == parallel_end_id:
                parallel_end_marker_index = i + 1

                if not structure_stack or structure_stack[-1]['type'] != 'parallel':
                    raise ValueError(f"</Parallel> found at index {i} without a matching <Parallel> block on stack.")

                closed_parallel_block = structure_stack.pop()
                #print(closed_parallel_block)
                path_spans_in_this_block = closed_parallel_block['path_spans']

                num_paths = len(path_spans_in_this_block)
                if num_paths > 1:
                    all_i_indices_to_mask = []
                    all_j_indices_to_mask = []
                    for path_idx_a in range(num_paths):
                        start_a, end_a = path_spans_in_this_block[path_idx_a]
                        # Ensure valid span before creating range
                        if start_a >= end_a: continue
                        indices_a = torch.arange(start_a, end_a, device=device)

                        for path_idx_b in range(path_idx_a + 1, num_paths):
                            start_b, end_b = path_spans_in_this_block[path_idx_b]
                            # Ensure valid span before creating range
                            if start_b >= end_b: continue
                            indices_b = torch.arange(start_b, end_b, device=device)

                            # Use broadcasting to get all (i, j) pairs efficiently
                            grid_i, grid_j = torch.meshgrid(indices_a, indices_b, indexing='ij')

                            all_i_indices_to_mask.append(grid_i.flatten())
                            all_j_indices_to_mask.append(grid_j.flatten())

                    if all_i_indices_to_mask: # Check if there's anything to mask
                        final_i = torch.cat(all_i_indices_to_mask)
                        final_j = torch.cat(all_j_indices_to_mask)

                        # Apply mask using advanced indexing (ensure indices are valid)
                        # For bool mask, False means masked
                        bool_attention_mask[final_i, final_j] = False
                        bool_attention_mask[final_j, final_i] = False # Symmetric masking
                elif num_paths <= 1:
                    # No masking needed if 0 or 1 path within the parallel block
                    pass

                i = parallel_end_marker_index
                continue

            # Move to next token if no tag matched
            i += 1
        # --- End of parsing loop ---

        # Final check for unclosed blocks
        if structure_stack:
            print(structure_stack)
            print(input_ids)
            unclosed_types = [block['type'] for block in structure_stack]
            raise ValueError(f"Input sequence ended with unclosed blocks: {unclosed_types}")

        # # Convert the final boolean mask to float format (0.0 for True, -inf for False)
        # float_attention_mask = torch.full_like(bool_attention_mask, -torch.inf, dtype=torch.float)
        # float_attention_mask = float_attention_mask.masked_fill(bool_attention_mask, 0.0)
        # print(bool_attention_mask)
        return bool_attention_mask

    def compute_structured_position_ids(self, response_ids: torch.Tensor) -> torch.Tensor:
        """
        Generate position_ids with left-padding and structural awareness:
        - Left padding is skipped
        - Each <Path> block continues counting from current position
        - After </Parallel>, continue from the max length of any <Path>
        """
        if response_ids.dim() == 2:
            assert response_ids.size(0) == 1, "Only support single-sample batching for now"
            response_ids = response_ids[0]

        pad_token_id = self.tokenizer.pad_token_id
        seq_len = response_ids.size(0)
        pos_ids = torch.zeros(seq_len, dtype=torch.long)
        nonpad_mask = (response_ids != pad_token_id)

        curr_pos = 0
        i = 0
        while i < seq_len:
            if not nonpad_mask[i]:
                i += 1
                continue

            tok = response_ids[i].item()

            if tok == self.start_parallel_token:
                pos_ids[i] = curr_pos
                i += 1
                # curr_pos += 1
                path_lengths = []
                path_start_indices = []
                path_end_indices = []
                temp_i = i

                # Collect path blocks
                while temp_i < seq_len:
                    if not nonpad_mask[temp_i]:
                        temp_i += 1
                        continue
                    if response_ids[temp_i].item() == self.start_path_token:
                        path_start = temp_i
                        local_pos = curr_pos + 1
                        while temp_i < seq_len:
                            if not nonpad_mask[temp_i]:
                                temp_i += 1
                                continue
                            pos_ids[temp_i] = local_pos
                            if response_ids[temp_i].item() == self.end_path_token:
                                break
                            local_pos += 1
                            temp_i += 1
                        path_end = temp_i
                        path_len = pos_ids[path_end] - curr_pos
                        path_lengths.append(path_len)
                        path_start_indices.append(path_start)
                        path_end_indices.append(path_end)
                    elif response_ids[temp_i].item() == self.end_parallel_token:
                        break
                    temp_i += 1

                i = temp_i
                max_path_len = max(path_lengths) if path_lengths else 0
                if i < seq_len and nonpad_mask[i]:
                    pos_ids[i] = curr_pos + max_path_len + 1
                    curr_pos += max_path_len + 1 + 1
            else:
                pos_ids[i] = curr_pos
                curr_pos += 1
            i += 1

        return pos_ids

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        tokenizer = self.tokenizer

        prompt = self.prompts[item]
        response = self.responses[item]

        # apply chat template
        prompt_chat = [{"role": "user", "content": prompt}]
        # print(prompt_chat)
        # string
        prompt_chat_str = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
        response_chat_str = response + tokenizer.eos_token

        # tokenize
        prompt_ids_output = tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)
        prompt_ids = prompt_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]

        response_ids_output = tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
        response_ids = response_ids_output["input_ids"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]

        attention_mask_1d = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)


        # response_attention_mask = self.generate_parallel_thinking_reasponse_mask(response_ids)
        # print(response_attention_mask.shape)
        # print(prompt_attention_mask)
        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)

        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            padded_attention_mask_1d = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask_1d.dtype)
            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask_1d = torch.cat((attention_mask_1d, padded_attention_mask_1d))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-self.max_length :]
                attention_mask_1d = attention_mask_1d[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask_1d = attention_mask_1d[: self.max_length]
            elif self.truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")

        attention_mask = self.generate_parallel_thinking_reasponse_mask(input_ids)
        # print(attention_mask)

        if sequence_length < self.max_length:
            attention_mask[:,-(self.max_length - sequence_length):] = False
            attention_mask[-(self.max_length - sequence_length):,:] = False
        
        # print(attention_mask)
        attention_mask = attention_mask.unsqueeze(0)
        # attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        float_attention_mask = torch.full_like(attention_mask, -torch.inf, dtype=torch.float)
        float_attention_mask = float_attention_mask.masked_fill(attention_mask, 0.0)

        position_ids = self.compute_structured_position_ids(input_ids)

        loss_mask = attention_mask_1d.clone()
        if prompt_length > 1:
            # mask out prompt for SFT.
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0
        # mask out the last token in response
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        # print(float_attention_mask.shape)
        # print(loss_mask.shape)
        # print(position_ids.shape)

        return {
            "input_ids": input_ids,
            "attention_mask": float_attention_mask,
            "bool_attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }

def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("TongZheng1999/Qwen3-4B-Base-add-special-token")

    dataset = ParallelThinkingSFTDataset(
        parquet_files=["/cq_1/share_1603164/user/kidzheng/Parallel-Scaling-for-LLMs-/verl/verl/utils/dataset/data/gsm8k/train.parquet"],  # 为空
        tokenizer=tokenizer,
        config={"prompt_key": "prompt", "response_key": "extra_info", "response_dict_keys": ["answer"], "max_length": 70}
    )

    dataset.prompts = ["What is 2 + 3?"]
    dataset.responses = [
        "Dave <Parallel><Path> First, </Path><Path> k = 8  Then, </Path><Path> k = 8 , </Path></Parallel>\n<Summary> Both </Summary> 26"
    ]

    sample = dataset[0]

    input_ids = sample["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    print("\n========== TOKENS ==========")
    print(tokens)
    print("\n========== POSITION_IDS ==========")
    print(sample["position_ids"])
    print("\n========== LOSS_MASK ==========")
    print(sample["loss_mask"])
    print("\n========== ATTENTION_MASK (last 10x10 block) ==========")
    mask = sample["bool_attention_mask"]
    print(mask.shape)

    def visualize_attention_mask(mask, tokens, save_path="attention_mask.png"):
        import matplotlib.pyplot as plt
        import numpy as np

        seq_len = len(tokens)
        max_len = min(seq_len, 100)
        cropped_mask = mask[:max_len, :max_len]

        plt.figure(figsize=(10, 10))
        plt.imshow(cropped_mask, cmap='gray', aspect='auto')
        plt.title("Attention Mask")
        plt.xlabel("Key Positions")
        plt.ylabel("Query Positions")
        plt.xticks(np.arange(max_len), tokens[:max_len], rotation=90, fontsize=12)
        plt.yticks(np.arange(max_len), tokens[:max_len], fontsize=12)
        plt.grid(False)
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(save_path, dpi=300)
        print(f"Saved attention mask figure to {save_path}")
        
        plt.show()

    visualize_attention_mask(np.array(mask.squeeze()), tokens)

if __name__ == "__main__":
    main()
