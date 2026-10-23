import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


# 1. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# 2. Load model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

print(f"Original parameters: {model.num_parameters():,}")


# 3. Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)


# 4. Attach LoRA
model = get_peft_model(model, lora_config)

model.print_trainable_parameters()


# 5. Load dataset
dataset = load_dataset("json", data_files="dataset.jsonl")


# 6. Format examples
def format_example(example):
    return {
        "text": (
            f"User: {example['input']}\n"
            f"Assistant: {example['target']}"
        )
    }


dataset = dataset["train"].map(format_example)


# 7. Tokenize
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=256,
    )


tokenized_dataset = dataset.map(
    tokenize,
    remove_columns=dataset.column_names,
)


# 8. Training configuration
training_args = TrainingArguments(
    output_dir="./aether-lora",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none",
    fp16=False,
)


# 9. Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


# 10. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)


# 11. TRAIN 🔥
trainer.train()


# 12. Save LoRA adapter
model.save_pretrained("./aether-lora")
tokenizer.save_pretrained("./aether-lora")

print("Training complete!")