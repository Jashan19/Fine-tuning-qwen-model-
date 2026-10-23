from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "./aether-lora"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

prompt = "I made app today that gained suddely four million users."

text = f"User: {prompt}\nAssistant:"

inputs = tokenizer(text, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=80,
    do_sample=False
)

response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print(response)