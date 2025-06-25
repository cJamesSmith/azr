import os
from functools import partial
import subprocess
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import re
import uuid
from functools import partial

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from verl import DataProto
from verl.protocol import DataProtoItem
from verl.utils.dataset.rl_dataset import collate_fn
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

import absolute_zero_reasoner.rewards.custom_evaluate as custom_evaluate
from absolute_zero_reasoner.rewards.code_reward import (
    parse_code_input_output,
    parse_inputs_message,
    parse_code_function,
    ast_edit_distance,
    get_code_complexity_reward,
    get_halstead_reward,
    get_type_counts_reward,
)
from absolute_zero_reasoner.rewards.custom_evaluate import get_format_reward, extract_answer, extract_thought
from absolute_zero_reasoner.data_construction.process_data import boxed_instruction, instruction_following
from absolute_zero_reasoner.data_construction.constructor import get_code_problem_predictor_prompt
from absolute_zero_reasoner.utils.dataset.rl_dataset import RLHFDataset
from absolute_zero_reasoner.utils.logging_utils.stdout import PrettyPrinter
from absolute_zero_reasoner.utils.code_utils.checks import check_composite_function, check_no_definitions

import io
import os
import sys
import ast
import json
import time
import argparse
import numpy as np
import multiprocessing

def extract_code(full_output):
    matches = re.findall(r"```python(.*?)```", full_output, re.DOTALL)
    if matches:
        code_output = matches[-1].strip()
    else:
        code_output = "We can not extract the code in the output. "
    return code_output

def modify(c):
    # Remove any occurrences of "plaintext\n"
    c = c.replace("plaintext\n", "")
    
    # Convert literal "\n" to actual newlines
    c = c.replace("\\n", "\n")
    
    # Ensure there's a trailing newline
    if not c.endswith("\n"):
        c += "\n"
    
    return c

def extract_test_cases(full_output):
    # First, try extracting with the updated triple-backtick pattern
    pattern_input_backticks = r'\*\*Test Input:\*\*\s*```(.*?)```'
    pattern_output_backticks = r'\*\*Test Output:\*\*\s*```(.*?)```'
    matches_input = re.findall(pattern_input_backticks, full_output, re.DOTALL)
    matches_output = re.findall(pattern_output_backticks, full_output, re.DOTALL)

    fail_case = [""]
    # For Test Input: either use the updated triple-backtick version or fallback to plain text
    if matches_input:
        test_input = [modify(matches_input[-1].lstrip('\n'))]
    else:
        # Fallback pattern without backticks: capture until **Test Output:**
        pattern_input_plain = r'\*\*Test Input:\*\*\s*([\s\S]*?)(?=\*\*Test Output:\*\*)'
        matches_input_plain = re.findall(pattern_input_plain, full_output, re.DOTALL)
        if matches_input_plain:
            test_input = [modify(matches_input_plain[-1].strip())]
        else:
            test_input = fail_case
    
    # For Test Output: either use the updated triple-backtick version or fallback to plain text
    if matches_output:
        test_output = [modify(matches_output[-1].lstrip('\n'))]
    else:
        # Fallback: capture until the **Explanation:** marker or end-of-string
        pattern_output_plain = r'\*\*Test Output:\*\*\s*([\s\S]*?)(?=\*\*Explanation:|\*\*Test Input:|$)'
        matches_output_plain = re.findall(pattern_output_plain, full_output, re.DOTALL)
        if matches_output_plain:
            test_output = [modify(matches_output_plain[-1].strip())]
        else:
            test_output = fail_case
    
    # Also extract from the last occurrence of **Test Input:** to the end
    index = full_output.rfind("**Test Input:**")
    if index != -1:
        example_text = [full_output[index:]]
    else:
        example_text = fail_case
    
    # If any essential piece is missing, return empties
    if example_text == fail_case or test_input == fail_case or test_output == fail_case:
        return fail_case, fail_case, fail_case
    
    return test_input, test_output, example_text

def str2bool(x):
    return x.lower() in ("1", "true", "yes")

def normalize_reward(reward_arr):
    if np.all(reward_arr == 1):
        return reward_arr
    mean = np.mean(reward_arr)
    std = np.std(reward_arr)
    if std.item() == 0:
        return None
    return (reward_arr - mean) / std

def normalize_balance_std(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    pos_mask = x > 0
    neg_mask = x < 0
    sum_pos = x[pos_mask].sum()
    sum_neg_abs = abs(x[neg_mask].sum())
    if sum_pos * sum_neg_abs == 0:
        return None
    scale_factor = sum_neg_abs / sum_pos
    x[pos_mask] *= scale_factor
    return x / x.std()

def length_regularize(reward_arr, response_length_list):
    reward_arr = np.sign(reward_arr)
    pos_list = np.where(reward_arr == 1)[0].tolist()
    neg_list = np.where(reward_arr == -1)[0].tolist()
    pos_response_length = np.array([response_length_list[j] for j in pos_list])
    threshold = np.median(pos_response_length).item()
    if np.sum((pos_response_length - threshold)**2) == 0: # no variance
        return normalize_balance_std(np.sign(reward_arr))
    threshold = max(min(threshold, 8000), 1000)
    length_reg_reward = np.zeros(len(reward_arr), float)
    length_reg_reward[pos_list] = - pos_response_length + threshold
    length_reg_reward[neg_list] = np.min(length_reg_reward).copy()
    length_reg_reward = normalize_balance_std(length_reg_reward)
    return length_reg_reward

class CodeIORewardManager():
    """The reward manager."""
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        num_examine: int,
        split: str,
        reward_fn_extraction_type: str,
        math_metric: str,
        splitter: str,
        output_path: str,
        generation_reward_config: Dict[str, Any],
        debug: bool = False,
        max_prompt_length: int = 8192,
        valid_program_filter: str = 'all',
        batched_estimate: bool = False,
        extract_code_block: bool = True,
        num_inputs: int = 10,
        code_f_reward_type: str = 'accuracy',
        boxed_retry: bool = False,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = partial(custom_evaluate.get_reward, math_metric=math_metric, boxed_retry=boxed_retry)
        self.reward_fn_extraction_type = reward_fn_extraction_type
        self.split = split
        self.splitter = splitter
        self.output_path = output_path
        self.max_prompt_length = max_prompt_length
        self.generation_reward_config = generation_reward_config
        self.valid_program_filter = valid_program_filter
        self.batched_estimate = batched_estimate
        self.debug = debug
        self.extract_code_block = extract_code_block
        self.use_original_code_as_ref = generation_reward_config.use_original_code_as_ref
        self.num_inputs = num_inputs
        self.code_f_reward_type = code_f_reward_type
        self.boxed_retry = boxed_retry

    @staticmethod
    def extract_input_output(extracted_content: str, return_input: bool = True, return_output: bool = False) -> Tuple[str, str]:
        input_pattern = r"```input\s*\n?(.*?)\n?```"
        output_pattern = r"```output\s*\n?(.*?)\n?```"
        assert not (return_input and return_output), "Cannot return both input and output"
        assert return_input or return_output, "Must return at least one of input or output"

        # Use flags for case-insensitive matching and dotall
        flags = re.DOTALL | re.IGNORECASE
        if return_input:
            input_matches = list(re.finditer(input_pattern, extracted_content, flags))
            if not input_matches:
                # Try alternative pattern without explicit input block
                input_matches = list(re.finditer(r"# Input:\s*(.*?)(?=\n```|$)", extracted_content, flags))
            if not input_matches:
                # Match input() function call and preserve quotes
                input_matches = list(re.finditer(r'input\s*\((.*?)\)', extracted_content, flags))
            if not input_matches:
                # Match <input> tag with optional closing tag, strip spaces
                input_matches = list(re.finditer(r"<input>\s*(.*?)(?:</input>|\s*$)", extracted_content, flags))
            if not input_matches:
                # Match "The input is" pattern case-insensitively
                input_matches = list(re.finditer(r"the input is\s*(.*?)\.?$", extracted_content, flags))
            # if still no input matches, use the extracted answer as the input
            # Don't strip() here to preserve quotes
            input_snippet = input_matches[-1].group(1) if input_matches else extracted_content
            return input_snippet

        if return_output:
            output_matches = list(re.finditer(output_pattern, extracted_content, flags))
            if not output_matches:
                # Try alternative pattern without explicit output block
                output_matches = list(re.finditer(r"# Output:\s*(.*?)(?=\n```|$)", extracted_content, flags))
            if not output_matches:
                # Match output() function call and preserve quotes
                output_matches = list(re.finditer(r'output\s*\((.*?)\)', extracted_content, flags))
            if not output_matches:
                # Match <output> tag with optional closing tag, strip spaces
                output_matches = list(re.finditer(r"<output>\s*(.*?)(?:</output>|\s*$)", extracted_content, flags))
            if not output_matches:
                # Match "The output is" pattern case-insensitively, strip space after "is" and period at end
                output_matches = list(re.finditer(r"the output is\s*(.*?)\.?$", extracted_content, flags))
            # if still no output matches, use the extracted answer as the output
            output_snippet = output_matches[-1].group(1) if output_matches else extracted_content
            return output_snippet

    def _get_data_dict(self, data_item: DataProtoItem, problem_type: str, executor, banned_words: List[str], uid: str, banned_assertion_keywords: List[str]) -> Dict:
        prompt_ids = data_item.batch['prompts']

        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch['responses']
        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # decode
        sequences = torch.cat((valid_prompt_ids, valid_response_ids))
        sequences_str = self.tokenizer.decode(sequences)

        ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
        data_source = data_item.non_tensor_batch['data_source']
        extra_info = data_item.non_tensor_batch['extra_info']
        non_special_tokens_sequences_str = self.tokenizer.decode(self.tokenizer.encode(sequences_str), skip_special_tokens=True)
        
        generation = non_special_tokens_sequences_str.split(self.splitter)[1].strip().strip('\"\'')
        extracted_content = extract_answer(generation, self.reward_fn_extraction_type, boxed_retry=self.boxed_retry)
        thought = extract_thought(generation)

        data_dict = {
            'generation': generation,
            'data_source': data_source,
            'ground_truth': ground_truth,
            'extra_info': extra_info,
            'non_special_tokens_sequences_str': non_special_tokens_sequences_str,
            'valid_response_length': valid_response_length,
            'extracted_content': extracted_content,
            'thought': thought,
            'uid': uid,
        }
        if problem_type.startswith('gen'):
            data_dict['references'] = [ref['snippet'] for ref in data_item.non_tensor_batch['extra_info']['chosen_references']]
            if problem_type != 'gen_code_f':
                data_dict['composite_functions'] = data_item.non_tensor_batch['extra_info']['composite_functions'].tolist()
            else:
                data_dict['imports'] = [ref['imports'] for ref in data_item.non_tensor_batch['extra_info']['chosen_references']]
            if self.use_original_code_as_ref:
                data_dict['original_references'] = [ref['original_snippet'] for ref in data_item.non_tensor_batch['extra_info']['chosen_references']]
        elif problem_type.startswith('pred') and 'code_f' not in problem_type:
            data_dict['program'] = data_item.non_tensor_batch['problem']
            data_dict['input'] = data_item.non_tensor_batch['extra_info']['input']
            data_dict['output'] = data_item.non_tensor_batch['extra_info']['output']
            data_dict['imports'] = data_item.non_tensor_batch['extra_info'].get('imports', [])
        elif problem_type.startswith('pred') and 'code_f' in problem_type:
            data_dict['program'] = data_item.non_tensor_batch['problem']
            data_dict['given_inputs'] = data_item.non_tensor_batch['extra_info']['given_inputs']
            data_dict['given_outputs'] = data_item.non_tensor_batch['extra_info']['given_outputs']
            data_dict['hidden_inputs'] = data_item.non_tensor_batch['extra_info']['hidden_inputs']
            data_dict['hidden_outputs'] = data_item.non_tensor_batch['extra_info']['hidden_outputs']
            data_dict['message'] = data_item.non_tensor_batch['extra_info']['message']
            data_dict['imports'] = data_item.non_tensor_batch['extra_info'].get('imports', [])

        # if QA task, we only need to check the format
        if problem_type is None:
            format_score = get_format_reward(solution_str=generation, extraction_type=self.reward_fn_extraction_type) if self.generation_reward_config.format_reward else 1.
            data_dict['format_score'] = format_score
            return data_dict
        # first go through, we only checking the format
        elif problem_type.startswith('gen') and 'code_f' not in problem_type:
            success, result = parse_code_input_output(
                extracted_content,
                parse_output=False,
                remove_after_return=self.generation_reward_config.remove_after_return and self.split == 'train',
                remove_comments=self.generation_reward_config.remove_comments and self.split == 'train',
                remove_print=self.generation_reward_config.remove_print and self.split == 'train',
                reject_multiple_functions=self.generation_reward_config.reject_multiple_functions,
                f_replace_location=self.generation_reward_config.f_replace_location,
                reject_test_input_in_code=self.generation_reward_config.reject_test_input_in_code,
            )
            if len(data_dict['composite_functions']) > 0 and success:
                # first, check if the composite function names are redefined in the code, which we do not allow
                success = check_no_definitions(result['code'], [f'g_{i}' for i in range(len(data_dict['composite_functions']))])
                if not success: # if the composite function names are redefined, we do not allow the code
                    data_dict['code_validity'] = False
                    data_dict['format_score'] = 0.
                    return data_dict

                composite_imports = '\n'.join(
                    '\n'.join(list(d['imports'])) if list(d['imports']) else '' for d in data_dict['composite_functions']
                ).strip()

                composite_snippets = '\n\n'.join(d['snippet'] for d in data_dict['composite_functions']).strip()

                # cache the original code
                result['original_code'] = result['code']

                result['code'] = f"{composite_imports}\n\n{composite_snippets}\n\n{result['code']}".strip()
                # TODO: composite function check
                success = check_composite_function(
                    code = result['code'],
                    composite_functions = [d['snippet'] for d in data_dict['composite_functions']],
                )
            if success:
                code_validity, output = executor.check_all(
                    code=result['code'],
                    inputs=result['input'],
                    banned_keywords=banned_words,
                    check_determinism=True,
                    imports=list(set(result['imports'])),
                    check_error=problem_type == 'gen_code_e',
                    banned_keywords_for_errors_and_exceptions=banned_assertion_keywords,
                )
                if not code_validity:
                    data_dict['code_validity'] = False
                    data_dict['format_score'] = 0.
                    return data_dict
                # means the code is valid, we append any good programs, but we eval format separately
                data_dict['answer'] = {
                    'snippet': result['code'],
                    'original_snippet': result['original_code'] if 'original_code' in result else result['code'],
                    'input': result['input'],
                    'output': output,
                    'imports': result['imports'],
                    'thought': thought,
                    'composite_functions': data_dict['composite_functions']
                }
                format_score = get_format_reward(solution_str=generation, extraction_type=self.reward_fn_extraction_type) if self.generation_reward_config.format_reward else 1.
                data_dict['format_score'] = format_score
                data_dict['code_validity'] = True
                return data_dict
            else:
                data_dict['code_validity'] = False
                data_dict['format_score'] = 0.
                return data_dict

        elif problem_type == 'gen_code_f':
            success, result = parse_inputs_message(
                extracted_content,
                num_inputs=self.num_inputs,
            )
            if success and len(result['inputs']) == self.num_inputs: # for code_f, we need to ensure the number of inputs is correct
                outputs = []
                for inpt in result['inputs']:
                    code_validity, output = executor.check_all(
                        code=data_dict['references'][0],
                        inputs=inpt,
                        banned_keywords=[],
                        check_determinism=True,
                        imports=data_dict['imports'][0],
                        check_error=False,
                        banned_keywords_for_errors_and_exceptions=[],
                    )
                    if not code_validity:
                        data_dict['code_validity'] = False
                        data_dict['format_score'] = 0.
                        return data_dict
                    outputs.append(output)
                data_dict['answer'] = {
                    'snippet': data_dict['references'][0],
                    'inputs': result['inputs'],
                    'outputs': outputs,
                    'message': result['message'],
                    'imports': data_dict['imports'][0],
                    'thought': thought,
                }
                format_score = get_format_reward(solution_str=generation, extraction_type=self.reward_fn_extraction_type) if self.generation_reward_config.format_reward else 1.
                data_dict['format_score'] = format_score
                data_dict['code_validity'] = True
                return data_dict
            else:
                data_dict['code_validity'] = False
                data_dict['format_score'] = 0.
                return data_dict

        # if prediction is the task
        elif problem_type.startswith('pred'):
            # Check required blocks
            if problem_type.endswith('code_i'): # parse input
                input_snippet = self.extract_input_output(extracted_content, return_input=True, return_output=False) \
                    if self.extract_code_block else extracted_content
                if input_snippet is None:
                    data_dict['format_score'] = 0.
                    return data_dict
                format_score = get_format_reward(solution_str=generation, extraction_type=self.reward_fn_extraction_type) if self.generation_reward_config.format_reward else 1.
                data_dict['format_score'] = format_score
                data_dict['answer'] = input_snippet
                return data_dict
            elif problem_type.endswith('code_o') or problem_type.endswith('code_e'): #  parse output, code_e format is same as code_o
                output_snippet = self.extract_input_output(extracted_content, return_input=False, return_output=True) \
                    if self.extract_code_block else extracted_content
                if output_snippet is None:
                    data_dict['format_score'] = 0.
                    return data_dict
                format_score = get_format_reward(solution_str=generation, extraction_type=self.reward_fn_extraction_type) if self.generation_reward_config.format_reward else 1.
                data_dict['format_score'] = format_score
                data_dict['answer'] = output_snippet
                return data_dict
            elif problem_type.endswith('code_f'):
                success, code_snippet = parse_code_function(extracted_content)
                if not success:
                    data_dict['format_score'] = 0.
                    return data_dict
                format_score = get_format_reward(solution_str=generation, extraction_type=self.reward_fn_extraction_type) if self.generation_reward_config.format_reward else 1.
                data_dict['format_score'] = format_score
                data_dict['answer'] = {
                    'snippet': code_snippet,
                    'given_inputs': data_dict['given_inputs'],
                    'given_outputs': data_dict['given_outputs'],
                    'hidden_inputs': data_dict['hidden_inputs'],
                    'hidden_outputs': data_dict['hidden_outputs'],
                    'message': data_dict['message'],
                    'imports': data_dict['imports'],
                    'thought': thought,
                    'gold_program': data_dict['program'],
                }
                return data_dict
            else:
                raise ValueError(f"Invalid problem type: {problem_type}")
        else:
            raise ValueError(f"Invalid problem type: {problem_type}")

    def __call__(
        self,
        data: List[DataProto],
        executor = None,
        rollout_actor_wg = None,
        banned_words: List[str] = [],
        banned_assertion_keywords: List[str] = [],
        n_samples: int = 1,
        max_ground_truth_test: int = 8,
        k_code: int = 1,
        k_case: int = 1,
    ) -> Tuple[torch.Tensor, Dict, List[Dict], List[Dict]]:
        """We will expand this function gradually based on the available datasets"""

        code_batch, case_batch, selected_data = data

        # preprocess
        for data in selected_data:
            data["full_code_generation"] = []
            data["code_response_length"] = []
            data["full_case_generation"] = []
            data["case_response_length"] = []
            data["generated_code"] = []
            max_k = min(max_ground_truth_test, len(data["test_input"]))
            data["num_ground_truth_test"] = max_k
            data["all_case_input"] = (data["test_input"][:max_k]).copy()
            data["all_case_output"] = (data["test_output"][:max_k]).copy()
            data["case_input"] = []
            data["case_output"] = []
            data["case_text"] = []

        # process generated codes
        for idx, code_b in enumerate(code_batch):
            code_text = self.tokenizer.decode(code_b.batch['responses'], skip_special_tokens=True)
            code_output = extract_code(code_text)
            selected_data[idx//k_code]['full_code_generation'] = selected_data[idx//k_code]['full_code_generation'] + [code_text]
            selected_data[idx//k_code]['generated_code'] = selected_data[idx//k_code]['generated_code'] + [code_output]
            selected_data[idx//k_code]['code_response_length'].append(len(self.tokenizer.encode(code_text, add_special_tokens=False)))

        # process generated unit tests
        for idx, case_b in enumerate(case_batch):
            case_text = self.tokenizer.decode(case_b.batch['responses'], skip_special_tokens=True)
            test_input, test_output, example_text = extract_test_cases(case_text)
            selected_data[idx // k_case]["full_case_generation"] = selected_data[idx // k_case]["full_case_generation"] + [case_text]
            selected_data[idx // k_case]["case_input"] = selected_data[idx // k_case]["case_input"] + test_input
            selected_data[idx // k_case]["case_output"] = selected_data[idx // k_case]["case_output"] + test_output
            selected_data[idx // k_case]["case_text"] = selected_data[idx // k_case]["case_text"] + example_text
            selected_data[idx // k_case]["all_case_input"] = selected_data[idx // k_case]["all_case_input"] + test_input
            selected_data[idx // k_case]["all_case_output"] = selected_data[idx // k_case]["all_case_output"] + test_output
            selected_data[idx // k_case]["case_response_length"].append(len(self.tokenizer.encode(case_text, add_special_tokens=False)))

        # data = execute_scripts(selected_data, num_chunks=k_code * k_case * len(selected_data))
        with open("./outputs-execute.json", "w", encoding="utf-8") as f:
            json.dump(selected_data, f, indent=2, ensure_ascii=False)

        subprocess.run(
            'python /home/aiops/chenxw/Absolute-Zero-Reasoner/absolute_zero_reasoner/rewards/exe.py',
            shell=True,
            check=True,
        )

        with open("./outputs-execute.json", 'r') as f:
            data = json.load(f)

        code_reward_tensor = torch.zeros_like(code_batch.batch['responses'], dtype=torch.float32)
        case_reward_tensor = torch.zeros_like(case_batch.batch['responses'], dtype=torch.float32)

        for i in range(len(data)):
            if data[i]["all_case_bool_table"] is None:
                continue
            
            t = data[i]["num_ground_truth_test"]
            all_test_table_i = np.array(data[i]["all_case_bool_table"])[:, :t].copy()
            all_case_table_i = np.array(data[i]["all_case_bool_table"])[:, t:].copy()

            # reward for code
            code_reward = np.mean(all_test_table_i, 1)
            #code_reward = all_test_table_i.all(axis=1).astype(float)
            code_reward = normalize_reward(code_reward)
            if code_reward is not None:
                code_reward = length_regularize(code_reward, data[i]["code_response_length"])
                if code_reward is not None:
                    code_reward = code_reward.tolist()
                    for j in range(len(code_reward)):
                        # code_data_i = {}
                        # code_data_i["prompt"] = data[i]["code_generation_prompt"]
                        # code_data_i["reward"] = code_reward[j]
                        # code_data.append(code_data_i)
                        prompt_ids = code_batch[i*k_code + j].batch['prompts']
                        prompt_length = prompt_ids.shape[-1]
                        valid_response_length = code_batch[i*k_code + j].batch['attention_mask'][prompt_length:].sum()
                        code_reward_tensor[i*k_code + j, valid_response_length - 1] = code_reward[j]
            
            # reward for case
            correct_code_list = np.where(all_test_table_i.all(axis=1))[0].tolist()
            if len(correct_code_list) > 0:
                # get reward sign
                correct_code_table = all_case_table_i[correct_code_list, :].copy()
                index_list = np.where(np.all(correct_code_table, axis=0))[0].tolist()
                reward_sign = -np.ones(correct_code_table.shape[1], dtype=float)
                reward_sign[index_list] = 1
                case_reward = reward_sign.copy()
                # get reward scale
                wrong_code_list = [j for j in range(all_case_table_i.shape[0]) if j not in correct_code_list]
                if len(wrong_code_list) > 0:
                    reward_scale = np.ones(correct_code_table.shape[1], dtype=float)
                    correct_case_list = np.where(correct_code_table.all(axis=0))[0].tolist()
                    wrong_case_list = [j for j in range(all_case_table_i.shape[1]) if j not in correct_case_list]
                    if len(correct_case_list):
                        wrong_code_correct_case_table = all_case_table_i[wrong_code_list, :][:, correct_case_list].copy()
                        if True:  # TODO: True by default, but suggest False after 100+ optimization steps
                            mean_p01 = np.mean(~wrong_code_correct_case_table, 0)
                        else:
                            mean_p01 = (~np.any(wrong_code_correct_case_table, axis=0)).astype(float)
                        reward_scale[correct_case_list] = reward_scale[correct_case_list] * mean_p01
                    if len(wrong_case_list):
                        wrong_code_wrong_case_table = all_case_table_i[wrong_code_list, :][:, wrong_case_list].copy()
                        if True:  # TODO: True by default, but suggest False after 100+ optimization steps
                            mean_p00 = np.mean(wrong_code_wrong_case_table, 0)
                        else:
                            mean_p00 = (np.any(wrong_code_wrong_case_table, axis=0)).astype(float)
                        reward_scale[wrong_case_list] = reward_scale[wrong_case_list] * mean_p00
                    case_reward = case_reward * reward_scale
                
                case_reward = normalize_reward(case_reward)
                if case_reward is not None:
                    case_reward = length_regularize(case_reward, data[i]["case_response_length"])
                    if case_reward is not None:
                        case_reward = case_reward.tolist()
                        for j in range(len(case_reward)):
                            # case_data_i = {}
                            # case_data_i["prompt"] = data[i]["case_generation_prompt"]
                            # if data[i]["case_response_length"][j] < max_generation_len:
                            #     case_data_i["response"] = data[i]["full_case_generation"][j] + "<|im_end|>"
                            # else:
                            #     case_data_i["response"] = data[i]["full_case_generation"][j]
                            # case_data_i["reward"] = case_reward[j]
                            # case_data.append(case_data_i)
                            prompt_ids = case_batch[i*k_case + j].batch['prompts']
                            prompt_length = prompt_ids.shape[-1]
                            valid_response_length = case_batch[i*k_case + j].batch['attention_mask'][prompt_length:].sum()
                            case_reward_tensor[i*k_case + j, valid_response_length - 1] = case_reward[j]

        return code_reward_tensor, case_reward_tensor

    def _get_problem_generator_rewards_and_valid_programs(
        self,
        data_dicts: List[Dict],
        problem_type: str,
        n_samples: int,
        rollout_actor_wg,
        executor,
        input_type_counters: Dict[str, Dict[str, int]] = None,
        output_type_counters: Dict[str, Dict[str, int]] = None,
        error_type_counters: Dict[str, Dict[str, int]] = None,
    ) -> Tuple[Dict[str, Dict[str, float]], List[Dict[str, str]]]:
        """This function uses samples to estimate the accuracy reward for each program, also computes the code complexity and mean edit distance of generated programs.
            Also returns the valid programs using filters.
            Args:
                data_dicts: List[Dict]: A list of data dictionaries.
                problem_type: str: The type of problem.
                n_samples: int: The number of samples to use.
                rollout_actor_wg: RolloutActorWG: The rollout actor.
                executor: PythonExecutor/CodeBoxExecutor: The executor.
                type_counters: Dict[str, Dict[str, int]]: The type counters.
            Returns:
               rewards: Dict[str, Dict[str, float]]: A dictionary of rewards for each program.
               valid_programs: List[Dict[str, str]]: A list of valid programs.
        """
        if problem_type.endswith('code_i'):
            type_counters = input_type_counters
        elif problem_type.endswith('code_o'):
            type_counters = output_type_counters
        elif problem_type.endswith('code_e'):
            type_counters = error_type_counters
        valid_data_dicts = [data_dict for data_dict in data_dicts if data_dict['code_validity']]
        uid2valid_dict_idx = {data_dict['uid']: i for i, data_dict in enumerate(valid_data_dicts)}
        valid_uids = [data_dict['uid'] for data_dict in data_dicts if data_dict['code_validity']]
        invalid_uids = [data_dict['uid'] for data_dict in data_dicts if not data_dict['code_validity']]
        assert len(valid_uids) + len(invalid_uids) == len(data_dicts)
        accuracies = {uid: 1.0 for uid in invalid_uids} # for invalid uids, we give maximum accuracy to the model
        rewards = defaultdict(dict)
        valid_programs = []
        if len(valid_uids) > 0:
            if self.reward_fn_extraction_type.startswith('boxed'):
                instruction_template = boxed_instruction
            elif self.reward_fn_extraction_type.startswith('answer'):
                instruction_template = instruction_following
            elif self.reward_fn_extraction_type.startswith('none'):
                instruction_template = '{}'
            else:
                raise ValueError(f"Invalid instruction type: {self.reward_fn_extraction_type}")
            prompts = []
            if problem_type.endswith('code_i'):
                pt = 'code_i'
            elif problem_type.endswith('code_o'):
                pt = 'code_o'
            elif problem_type.endswith('code_e'):
                pt = 'code_e'
            elif problem_type.endswith('code_f'):
                pt = 'code_f'
            else:
                raise ValueError(f"Invalid problem type: {problem_type}")
            for data_dict in valid_data_dicts:
                if pt == 'code_f':
                    num_given_inputs = len(data_dict['answer']['inputs']) // 2
                    num_given_outputs = len(data_dict['answer']['outputs']) // 2
                    data_dict['answer']['given_inputs'] = data_dict['answer']['inputs'][:num_given_inputs]
                    data_dict['answer']['given_outputs'] = data_dict['answer']['outputs'][:num_given_outputs]
                    data_dict['answer']['hidden_inputs'] = data_dict['answer']['inputs'][num_given_inputs:]
                    data_dict['answer']['hidden_outputs'] = data_dict['answer']['outputs'][num_given_outputs:]
                    io_prompt = instruction_template.format(
                        get_code_problem_predictor_prompt(
                            problem_type=problem_type,
                            snippet=data_dict['answer']['snippet'],
                            message=data_dict['answer']['message'],
                            input_output_pairs=zip(data_dict['answer']['given_inputs'], data_dict['answer']['given_outputs']),
                        )
                    )
                else:
                    io_prompt = instruction_template.format(
                        get_code_problem_predictor_prompt(
                            problem_type=pt,
                            snippet=data_dict['answer']['snippet'],
                            input_args=data_dict['answer']['input'],
                            output=data_dict['answer']['output'],
                        )
                    )
                prompts_dict = {
                    'prompt': [{'role': 'user', 'content': io_prompt}],
                    'uid': data_dict['uid'],
                    'problem': data_dict['answer'],
                    'data_source': data_dict['data_source'],
                    'ground_truth': data_dict['answer']['output'] if pt != 'code_f' else data_dict['answer']['snippet'],
                    'extra_info': data_dict['extra_info'],
                    'program': data_dict['answer']['snippet'],
                    'imports': data_dict['answer']['imports'],
                    'references': data_dict['references'],
                }
                if pt == 'code_f':
                    prompts_dict.update({
                        'given_inputs': data_dict['answer']['given_inputs'],
                        'given_outputs': data_dict['answer']['given_outputs'],
                        'hidden_inputs': data_dict['answer']['hidden_inputs'],
                        'hidden_outputs': data_dict['answer']['hidden_outputs'],
                        'message': data_dict['answer']['message'],
                    })
                else:
                    prompts_dict.update({
                        'input': data_dict['answer']['input'],
                        'output': data_dict['answer']['output'],
                        'original_program': data_dict['answer']['original_snippet'],
                        'composite_functions': data_dict['answer']['composite_functions'],
                    })
                prompts.append(prompts_dict)

            # sampling to estimate the accuracy
            PrettyPrinter.section_header("Sampling to Estimate Accuracy")
            prompts = prompts * n_samples # repeat the prompts n_samples times
            pd.DataFrame(prompts).to_parquet(f'{self.output_path}/temp.parquet') # RLHFDataset expects parquet
            temp_data = RLHFDataset(
                parquet_files=f'{self.output_path}/temp.parquet',
                tokenizer=self.tokenizer,
                prompt_key='prompt',
                max_prompt_length=self.max_prompt_length,
                filter_prompts=True,
                return_raw_chat=False,
                truncation='error'
            )
            os.remove(f'{self.output_path}/temp.parquet') # we do not need this file after we load in the dataset
            sampler = torch.utils.data.SequentialSampler(data_source=temp_data)

            dataloader = torch.utils.data.DataLoader(
                dataset=temp_data,
                batch_size=len(temp_data),
                drop_last=False,
                shuffle=False,
                collate_fn=collate_fn,
                sampler=sampler,
            )
            assert len(dataloader) == 1
            data = next(iter(dataloader))
            batch = DataProto.from_single_dict(data)
            gen_batch = batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': True,
                'validate': True,
            }
            # pad to be divisible by dp_size
            gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, rollout_actor_wg.world_size)
            output_gen_batch_padded = rollout_actor_wg.generate_sequences(gen_batch_padded)
            # unpad
            output_gen_batch = unpad_dataproto(output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            # Store generated outputs
            batch = batch.union(output_gen_batch)
            batched_responses = []
            for b in batch:
                batch_dict = {
                        'extracted_answers': extract_answer(
                            self.tokenizer.decode(b.batch['responses'], skip_special_tokens=True),
                            self.reward_fn_extraction_type,
                            boxed_retry=self.boxed_retry,
                        ),
                        'uid': b.non_tensor_batch['uid'],
                        'problem': b.non_tensor_batch['problem'],
                        'data_source': b.non_tensor_batch['data_source'],
                        'extra_info': b.non_tensor_batch['extra_info'],
                        'program': b.non_tensor_batch['program'],
                        'references': b.non_tensor_batch['references'],
                        'imports': b.non_tensor_batch['imports'],
                    }
                if pt == 'code_f':
                    batch_dict.update({
                        'given_inputs': b.non_tensor_batch['given_inputs'],
                        'given_outputs': b.non_tensor_batch['given_outputs'],
                        'hidden_inputs': b.non_tensor_batch['hidden_inputs'],
                        'hidden_outputs': b.non_tensor_batch['hidden_outputs'],
                        'message': b.non_tensor_batch['message'],
                    })
                else:
                    batch_dict.update({
                        'input': b.non_tensor_batch['input'],
                        'output': b.non_tensor_batch['output'],
                        'original_program': b.non_tensor_batch['original_program'],
                        'composite_functions': b.non_tensor_batch['composite_functions'].tolist(),
                    })
                batched_responses.append(batch_dict)
            df = pd.DataFrame(batched_responses)

            # estimating accuracy using python executor
            PrettyPrinter.section_header("Estimating Accuracy Using Python Executor")
            for valid_uid in valid_uids:
                df_valid = df[df['uid'] == valid_uid]
                if df_valid.empty: # the prompt got filtered out TODO: check
                    accuracies[valid_uid] = 0.0
                    continue
                if pt != 'code_f':
                    answers = [self.extract_input_output(
                        answer,
                        return_input=problem_type.endswith('code_i'),
                        return_output=(problem_type.endswith('code_o') or problem_type.endswith('code_e')) # code_e output format is same as code_o
                    ) for answer in df_valid['extracted_answers'].tolist()]
                else:
                    answers = [parse_code_function(answer) for answer in df_valid['extracted_answers'].tolist()]
                answer_cache = {} # for the same uid, the answer is the same and the program is assumed to be deterministic, therefore we cache the answer -> accuracy mapping
                if pt == 'code_f':
                    hidden_outputs = df_valid['hidden_outputs'].tolist()[0].tolist()
                    hidden_inputs = df_valid['hidden_inputs'].tolist()[0].tolist()
                else:
                    gold_output = df_valid['output'].tolist()[0]
                    program = df_valid['program'].tolist()[0]
                    # gold_input = df_valid['input'].tolist()[0]
                imports = df_valid['imports'].tolist()[0]
                problem_accuracies = []
                if problem_type.endswith('code_i'):
                    if self.batched_estimate:
                        problem_accuracies = executor.eval_k_input_prediction(code=program, gold_output=gold_output, k_agent_inputs=answers, imports=list(set(imports)))
                    else:
                        for answer in answers:
                            if answer in answer_cache:
                                problem_accuracies.append(answer_cache[answer])
                                continue
                            acc_reward = executor.eval_input_prediction(code=program, gold_output=gold_output, agent_input=answer, imports=list(set(imports)))
                            if acc_reward is not None:
                                problem_accuracies.append(acc_reward)
                            answer_cache[answer] = acc_reward
                        # if self.debug:
                        #     batched_problem_accuracies = executor.eval_k_input_prediction(code=program, gold_output=gold_output, k_agent_inputs=answers, imports=list(set(imports)))
                        #     assert np.mean(batched_problem_accuracies) == np.mean(problem_accuracies), f"Gen I batch accuracy: {np.mean(batched_problem_accuracies)}, Single accuracy: {np.mean(problem_accuracies)}"
                elif problem_type.endswith('code_o'):
                    if self.batched_estimate:
                        problem_accuracies = executor.eval_k_output_prediction(code=program, gold_output=gold_output, k_agent_outputs=answers, imports=list(set(imports)))
                    else:
                        for answer in answers:
                            if answer in answer_cache:
                                problem_accuracies.append(answer_cache[answer])
                                continue
                            acc_reward = executor.eval_output_prediction(code=program, gold_output=gold_output, agent_output=answer, imports=list(set(imports)))
                            if acc_reward is not None:
                                problem_accuracies.append(acc_reward)
                            answer_cache[answer] = acc_reward
                        # if self.debug:
                        #     batched_problem_accuracies = executor.eval_k_output_prediction(code=program, gold_output=gold_output, k_agent_outputs=answers, imports=list(set(imports)))
                        #     assert np.mean(batched_problem_accuracies) == np.mean(problem_accuracies), f"Gen O batch accuracy: {np.mean(batched_problem_accuracies)}, Single accuracy: {np.mean(problem_accuracies)}"
                elif problem_type.endswith('code_e'): # string matching for errors
                    for answer in answers:
                        answer = answer.split(' ')[0].split(':')[0]
                        if answer.lower() == gold_output.lower():
                            problem_accuracies.append(1.0)
                        else:
                            problem_accuracies.append(0.0)
                elif problem_type.endswith('code_f'):
                    for parsed, answer in answers: # for each input/output set, we sampled n codes to estimate the accuracy
                        if not parsed: # the code answer is not parsed, we assume the code is not valid
                            problem_accuracies.append(0.0)
                            continue
                        code_accuracies = []
                        for inpt, outpt in zip(hidden_inputs, hidden_outputs):
                            code_accuracies.append(executor.eval_input_prediction(code=answer, gold_output=outpt, agent_input=inpt, imports=list(set(imports))))
                        answer_acc = np.mean([a for a in code_accuracies if a is not None]) if code_accuracies else 0.0
                        if self.code_f_reward_type == 'binary':
                            problem_accuracies.append(1.0 if answer_acc == 1.0 else 0.0)
                        elif self.code_f_reward_type == 'if_one_correct':
                            problem_accuracies.append(1.0 if answer_acc > 0 else 0.0)
                        elif self.code_f_reward_type == 'accuracy':
                            problem_accuracies.append(answer_acc)
                        else:
                            raise ValueError(f"Invalid code_f_reward_type: {self.code_f_reward_type}")
                accuracies[valid_uid] = sum(problem_accuracies) / len(problem_accuracies) if problem_accuracies else 0.0

                # filtering valid programs
                if self.valid_program_filter == 'all':
                    valid_programs.append(valid_data_dicts[uid2valid_dict_idx[valid_uid]]['answer'])
                elif self.valid_program_filter == 'non_one':
                    if accuracies[valid_uid] < 1.0:
                        valid_programs.append(valid_data_dicts[uid2valid_dict_idx[valid_uid]]['answer'])
                elif self.valid_program_filter == 'non_extremes':
                    if accuracies[valid_uid] > 0.0 and accuracies[valid_uid] < 1.0:
                        valid_programs.append(valid_data_dicts[uid2valid_dict_idx[valid_uid]]['answer'])
                else:
                    raise ValueError(f"Invalid valid program filter: {self.valid_program_filter}")

        # getting other rewards
        PrettyPrinter.section_header("Getting Other Rewards")
        # outputting rewards
        for d in data_dicts:
            uid = d['uid']
            if self.generation_reward_config.generation_accuracy_convertion == 'one_minus':
                rewards[uid]['accuracy'] = (1 - accuracies[uid]) if accuracies[uid] > 0 else 0.0
            elif self.generation_reward_config.generation_accuracy_convertion == 'inverse':
                rewards[uid]['accuracy'] = 1 - accuracies[uid]
            else:
                raise ValueError(f"Invalid generation accuracy convertion: {self.generation_reward_config.generation_accuracy_convertion}")

        if not problem_type.endswith('code_f'):
            code_key = 'original_snippet' if self.use_original_code_as_ref else 'snippet'
            reference_key = 'original_references' if self.use_original_code_as_ref else 'references'
            if problem_type.endswith('code_i'):
                type_counter_key = 'input'
            elif problem_type.endswith('code_o'):
                type_counter_key = 'output'
            elif problem_type.endswith('code_e'):
                type_counter_key = 'error'
            else:
                raise ValueError(f"Invalid problem type: {problem_type}")
            for data_dict in data_dicts:
                rewards[data_dict['uid']]['complexity'] = get_code_complexity_reward(data_dict['answer'][code_key]) if 'answer' in data_dict else 0.0
            for data_dict in data_dicts:
                rewards[data_dict['uid']]['mean_edit_distance'] = np.mean([ast_edit_distance(data_dict['answer'][code_key], ref) for ref in data_dict[reference_key]]) if 'answer' in data_dict else 0.0
            for data_dict in data_dicts:
                rewards[data_dict['uid']]['halstead'] = get_halstead_reward(data_dict['answer'][code_key]) if 'answer' in data_dict else 0.0
            for data_dict in data_dicts:
                rewards[data_dict['uid']]['type_counts'] = get_type_counts_reward(
                    data_dict['answer'][type_counter_key],
                    type_counters,
                    hierarchical=self.generation_reward_config.answer_diversity_reward.hierarchical
                ) if 'answer' in data_dict else 0.0
            if self.debug:
                for data_dict in data_dicts:
                    if 'answer' in data_dict:
                        continue
        else:
            for data_dict in data_dicts:
                rewards[data_dict['uid']]['input_type_counts'] = []
                rewards[data_dict['uid']]['output_type_counts'] = []
                if 'answer' in data_dict:
                    for inpt, outpt in zip(data_dict['answer']['inputs'], data_dict['answer']['outputs']):
                        rewards[data_dict['uid']]['input_type_counts'].append(get_type_counts_reward(
                            inpt,
                            input_type_counters,
                            hierarchical=self.generation_reward_config.answer_diversity_reward.hierarchical
                        ))
                        rewards[data_dict['uid']]['output_type_counts'].append(get_type_counts_reward(
                            outpt,
                            output_type_counters,
                            hierarchical=self.generation_reward_config.answer_diversity_reward.hierarchical
                        ))
                    rewards[data_dict['uid']]['input_type_counts'] = np.mean(rewards[data_dict['uid']]['input_type_counts'])
                    rewards[data_dict['uid']]['output_type_counts'] = np.mean(rewards[data_dict['uid']]['output_type_counts'])
                else:
                    rewards[data_dict['uid']]['input_type_counts'] = 0.0
                    rewards[data_dict['uid']]['output_type_counts'] = 0.0

        # turn into normal dict
        rewards = dict(rewards)
        return rewards, valid_programs
