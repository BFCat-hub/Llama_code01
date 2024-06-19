import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime

import torch
from peft import LoraConfig, prepare_model_for_int8_training, get_peft_model, get_peft_model_state_dict
from transformers import TrainingArguments, HfArgumentParser, AutoModelForCausalLM, Trainer, DataCollatorForSeq2Seq, \
    AutoTokenizer

import Config
from utils.data_utils import load_json_data, generate_and_tokenize_prompt

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default='D:/model/code-llama-ins',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    lora_dropout: float = field(default=0.05)   #lora_dropout 参数表示在训练过程中，在 Lora 模块中随机丢弃部分神经元的概率。
    lora_alpha: int = field(default=16)
    lora_r: int = field(default=16)
    wandb_project: str = field(default="c2cuda-project")


@dataclass
class DataArguments:
    max_seq_length: int = field(default=1160)
    full_data_dir: str = field(default="E:\codes\python\c2cudallama\dataset")
    data_cache_dir: str = field(default="data_cache")


def parse_args():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments)) #TrainingArguments包含训练过程中的许多重要参数，例如学习率、批处理大小、训练的最大步数等
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()
    train_args.run_name = f"codellama-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"

    Config.model_args = model_args
    Config.data_args = data_args
    Config.train_args = train_args
    Config.max_seq_length = data_args.max_seq_length
    return model_args, data_args, train_args


def set_lora_config(model_args: ModelArguments):
    config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return config


def set_wandb(model_args: ModelArguments):
    wandb_project = model_args.wandb_project
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project


def load_dataset(data_args: DataArguments):
    all_dataset = load_json_data(data_args.full_data_dir, "all", data_args.data_cache_dir)
    tok_all_dataset = all_dataset.map(generate_and_tokenize_prompt)
                                                                                                                                                               
    tok_train_dataset = tok_all_dataset.train_test_split(test_size=0.1)["train"]
    tok_test_dataset = tok_all_dataset.train_test_split(test_size=0.1)["test"]

    print(f"Train data:{len(tok_train_dataset)}\t Test data:{len(tok_test_dataset)}")
    return tok_train_dataset, tok_test_dataset


def set_tokenizer(model_args: ModelArguments):
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path) #从模型路径加载llama的分词器
    tokenizer.add_eos_token = True
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    Config.tokenizer = tokenizer
    return tokenizer


def set_model(model_args: ModelArguments, train_args: TrainingArguments):
    model = AutoModelForCausalLM.from_pretrained(  
        model_args.model_name_or_path,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    ) #从路径加载模型
    model.train()  # put model back into training mode
    model = prepare_model_for_int8_training(model)

    lora_config = set_lora_config(model_args)
    model = get_peft_model(model, lora_config)
    if torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    return model


def train(tokenizer: AutoTokenizer, model: AutoModelForCausalLM, tok_train_dataset, tok_valid_dataset,
          train_args: TrainingArguments, model_args: ModelArguments):
    trainer = Trainer(
        model=model,
        train_dataset=tok_train_dataset,
        eval_dataset=tok_valid_dataset,
        args=train_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    trainer.train()
    if train_args.output_dir is not None and "" != train_args.output_dir:
        if not os.path.exists(train_args.output_dir):
            os.mkdir(train_args.output_dir)
        trainer.save_model(output_dir=train_args.output_dir)


if __name__ == '__main__':
    model_args, data_args, train_args = parse_args()
    set_wandb(model_args)

    tokenizer = set_tokenizer(model_args)
    tok_train_dataset, tok_valid_dataset = load_dataset(data_args)
    model = set_model(model_args, train_args)
    train(tokenizer, model, tok_train_dataset, tok_valid_dataset, train_args, model_args)
