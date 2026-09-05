import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model path (checks local cache first, fallback to HF hub)
local_paths = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gpt2-indo-instruct-tuned"),
    "/home/akhyar/Dokumen/Code/python/MemoryBank/gpt2-indo-instruct-tuned",
    "/home/akhyar/Dokumen/Code/python/MemoryBank-bencmark/gpt2-indo-instruct-tuned",
]
model_path = "izzulgod/gpt2-indo-instruct-tuned"
for p in local_paths:
    if os.path.exists(os.path.join(p, "model.safetensors")):
        model_path = p
        break

print(f"Loading model from: {model_path}")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

# Create prompt
prompt = "User: Siapa presiden pertama Indonesia?\nAI:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate response
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|endoftext|>")
    )

# Decode response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n--- Output Model ---")
print(response)
