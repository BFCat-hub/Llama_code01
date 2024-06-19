import json

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

import Config
from utils.data_utils import load_json_data, generate_and_tokenize_prompt
from peft import PeftModel

model_path = r"D:\model\code-llama-ins"
    
tokenizer = AutoTokenizer.from_pretrained(model_path)
Config.tokenizer = tokenizer

from utils.data_utils import generate_and_tokenize_prompt_for_pre

all_dataset = load_json_data(r'E:\codes\python\c2cudallama\dataset', "all",
                             cache_dir=r'E:\codes\python\c2cudallama\data_cache')
# test_dataset = all_dataset.train_test_split(test_size=0.1)["test"]
print(len(all_dataset))

input_prompts = []
for data in all_dataset:
    input_prompts.append(generate_and_tokenize_prompt_for_pre(data))

model = AutoModelForCausalLM.from_pretrained(model_path, load_in_8bit=True, torch_dtype=torch.float16,
                                             device_map="auto", )

model_save_dir = r"E:\codes\python\c2cudallama\Checkpoints"
model = PeftModel.from_pretrained(model, model_save_dir)

pipeline = transformers.pipeline(
    "translation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    tokenizer=tokenizer
)

ge_results = []

for index, ip in enumerate(input_prompts):
    try:
        sequences = pipeline(ip,
                             do_sample=True,
                             top_k=10,
                             temperature=0.1,
                             top_p=0.98,
                             num_return_sequences=1,
                             eos_token_id=tokenizer.eos_token_id,
                             max_length=1000,
                             )
        seq_str = sequences[0]["translation_text"]
        if len(seq_str.split("Output:")) > 1:
            res_str = seq_str.split("Output:")[1]
            ge_results.append({"answer": all_dataset[index]["cu_code"], "prediction": res_str})
            print(ge_results)
    except Exception as e:
        print("seq_str:", seq_str)
        print("index:", index)
        print("prompt:", input_prompts[index])
        print(e)
    finally:
        with open(r"E:\codes\python\c2cudallama\output\ge_res.json", "w", encoding="utf-8") as f:
            json.dump(ge_results, f)
