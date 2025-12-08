from datasets import load_dataset
from transformers import AutoTokenizer
import os 

# Prepare path

dataset_name = "NeelNanda/c4-10k"
save_path = "data/" + dataset_name
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Load and tokenize
raw = load_dataset(dataset_name, split="train")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize(example):
    return {"tokens": tokenizer(example["text"], truncation=True)["input_ids"]}

tokenized = raw.map(tokenize)
tokenized.save_to_disk(save_path)
print(f"Tokenized dataset saved to {save_path}")
