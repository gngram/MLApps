import torch
import sys
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Ensure correct command-line arguments
if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <document_file>")
    sys.exit(1)

document_file = sys.argv[1]  # Document file path

# Load TinyLlama model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
cache_dir = "./model_cache"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device: ", device)

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load Sentence Transformer for embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load document
try:
    with open(document_file, "r", encoding="utf-8") as f:
        document = f.read()
except FileNotFoundError:
    print(f"Error: File '{document_file}' not found.")
    sys.exit(1)

# Split document into smaller chunks (e.g., by paragraph)
document_chunks = document.split("\n\n")
chunk_embeddings = np.array([embedder.encode(chunk) for chunk in document_chunks], dtype="float32")

# Create FAISS index
index = faiss.IndexFlatL2(chunk_embeddings.shape[1])  # L2 distance
index.add(chunk_embeddings)

print(f"Indexed {len(document_chunks)} document chunks.")

# Interactive Q&A loop
while True:
    question = input("\nEnter your question (or type 'exit' to quit): ").strip()
    if question.lower() == "exit":
        print("Exiting Q&A.")
        break

    # Convert question to an embedding and find the closest chunk
    question_embedding = np.array([embedder.encode(question)], dtype="float32")
    _, retrieved_indices = index.search(question_embedding, k=1)
    best_chunk = document_chunks[retrieved_indices[0][0]]

    print("\nRetrieved Relevant Chunk:\n", best_chunk)

    # Generate response using retrieved context
    prompt = f"Use the following information to answer the question:\n{best_chunk}\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=150)

    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\nGenerated Answer:\n", answer)

