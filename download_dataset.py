# download_dataset.py
# Download 20k datapoints from xl-zhao/PromptCoT-2.0-Concepts dataset

from datasets import load_dataset
import json

# Load the dataset
print("Loading dataset from Hugging Face...")
ds = load_dataset("xl-zhao/PromptCoT-2.0-Concepts")

# Determine which split to use (usually 'train' or default)
if isinstance(ds, dict):
    # If dataset has multiple splits, use 'train' or the first available split
    split_name = 'train' if 'train' in ds else list(ds.keys())[0]
    dataset = ds[split_name]
    print(f"Using split: {split_name}")
else:
    dataset = ds

print(f"Total datapoints available: {len(dataset)}")

# Limit to 20k datapoints
num_datapoints = min(20000, len(dataset))
dataset_subset = dataset.select(range(num_datapoints))

print(f"Selected {num_datapoints} datapoints")

# Save to JSONL file
output_file = "./data/base/promptcot_concepts_20k.jsonl"
print(f"Saving to {output_file}...")

with open(output_file, 'w', encoding='utf-8') as f:
    for item in dataset_subset:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"✓ Successfully saved {num_datapoints} datapoints to {output_file}")

