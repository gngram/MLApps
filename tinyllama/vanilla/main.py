from transformers import AutoModelForCausalLM, AutoTokenizer

# Load TinyLlama model
# https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

cache_dir = "./model_cache"

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load document
with open("document.txt", "r") as f:
    document = f.read()

# Prepare the prompt
prompt = f"Summarize the following text:\n{document}\n\nSummary:"

# Tokenize and generate response
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)

# Decode summary
summary = tokenizer.decode(output[0], skip_special_tokens=True)
print(summary)

