from datasets import load_dataset

dataset = load_dataset("json", data_files="dataset.jsonl")

def format_example(example):
    return {
        "text": (
            f"User: {example['input']}\n"
            f"Assistant: {example['target']}"
        )
    }

formatted_dataset = dataset["train"].map(format_example)

print(formatted_dataset[0])