from transformers import BertTokenizer
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import json

# 加载分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def load_data_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 分析数据长度分布函数
def analyze_length_distribution(data, tokenizer):
    lengths = [len(tokenizer.encode(item['c_code'])) for item in data]
    return lengths


# 示例数据
data = load_data_from_json('test02.json')

# 分析长度分布
lengths = analyze_length_distribution(data, tokenizer)

# 统计长度分布
length_counter = Counter(lengths)
print("Length distribution:", length_counter)

# 绘制长度分布图
plt.hist(lengths, bins=50)
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.show()

# 打印百分位数
for percentile in [50, 75, 90, 95, 99]:
    length_at_percentile = np.percentile(lengths, percentile)
    print(f"{percentile}th percentile: {length_at_percentile}")

# 选择覆盖95%的数据长度作为max_length
max_length = int(np.percentile(lengths, 95))
print(f"Selected max_length: {max_length}")


# 分词函数
def tokenize(prompt, tokenizer, max_length):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )

    # 自监督学习的标签和输入相同
    result["labels"] = result["input_ids"].copy()

    return result


# 生成并分词函数
def generate_and_tokenize_prompt(data_point, tokenizer, max_length):
    prompt = data_point["text"]  # 假设 data_point 包含一个 "text" 字段
    tokenized_prompt = tokenize(prompt, tokenizer, max_length)
    return tokenized_prompt


# 示例调用
data_point = {"text": "The quick brown fox jumps over the lazy dog."}
tokenized_prompt = generate_and_tokenize_prompt(data_point, tokenizer, max_length)
print(tokenized_prompt)
