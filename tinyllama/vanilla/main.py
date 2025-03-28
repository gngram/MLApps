import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check if a filename was provided
if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <document_file>")
    sys.exit(1)

document_file = sys.argv[1]  # Get file name from argument

# Load TinyLlama model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
cache_dir = "./model_cache"

# Move model to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load document from the file
try:
    with open(document_file, "r", encoding="utf-8") as f:
        document = f.read()
except FileNotFoundError:
    print(f"Error: File '{document_file}' not found.")
    sys.exit(1)

# Prepare the prompt
prompt = f"Summarize the following text:\n{document}\n\nSummary:"

# Tokenize and move input tensors to GPU
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate response on GPU
with torch.no_grad():  # Disable gradient calculations (saves memory)
    output = model.generate(**inputs, max_new_tokens=200)

# Decode summary
summary = tokenizer.decode(output[0], skip_special_tokens=True)
print(summary)

