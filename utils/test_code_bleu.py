import json
from pathlib import Path

from codebleu import calc_codebleu

predictions = []
references = []
with open("../output/ge_res.json", "r", encoding="utf-8") as f:
    datas = json.load(f)
    for data in datas:
        predictions.append(data["prediction"])
        references.append(data["answer"])

# lang_so_file_path = Path("./my-languages.so")
# print(lang_so_file_path)
result = calc_codebleu(references, predictions, lang="c", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
print(result)
# {
#   'codebleu': 0.5537,
#   'ngram_match_score': 0.1041,
#   'weighted_ngram_match_score': 0.1109,
#   'syntax_match_score': 1.0,
#   'dataflow_match_score': 1.0
# }
