from transformers import AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "I studied DSA for 2 hours today."

tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print("TEXT:")
print(text)

print("\nTOKENS:")
print(tokens)

print("\nTOKEN IDs:")
print(token_ids)

print("\nNUMBER OF TOKENS:")
print(len(token_ids))