import os

from datasets import load_dataset

import Config


def load_json_data(data_dir, key, cache_dir):
    file_path = os.path.join(data_dir, f"{key}.json")
    dataset = load_dataset("json", data_files=file_path, split="train", cache_dir=cache_dir)
    return dataset


def tokenize(prompt):
    result = Config.tokenizer(
        prompt,
        truncation=True,
        max_length=Config.max_seq_length,
        padding=False,
        return_tensors=None,
    )

    # "self-supervised learning" means the labels are also the inputs:
    result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(data_point):
    full_prompt = f"""
    Instruction：Please convert the code to CUDA C code. \n
    Input: {data_point["c_code"]} \n
    Output: {data_point["cu_code"]} ,
    """
    return tokenize(full_prompt)


def generate_and_tokenize_prompt_for_pre(data):
    full_prompt = f"Instruction：Please convert the code to CUDA C code.\n Input: {data['c_code']}\n' Output: "
    return full_prompt
