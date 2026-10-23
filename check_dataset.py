import json

with open("dataset.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        example = json.loads(line)
        print("INPUT :", example["input"])
        print("TARGET:", example["target"])
        print("-" * 60)