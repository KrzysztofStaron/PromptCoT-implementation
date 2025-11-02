# generation.py — PROMPTCoT: Generate z (rationale) and x (problem) from c (concepts)
import json
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", padding_side='left')
tokenizer.pad_token = tokenizer.eos_token

# Load the prompt generation model prompt_model
# Trained to generate: Concepts -> Rationale, Problem
# So it can generate both z and x from c
prompt_model = PeftModel.from_pretrained(base, "./models/prompt_model", is_trainable=False)
prompt_model.eval()

# Load concept pool from mathematics_concepts dataset (more diverse)
concepts = set()
with open("./data/base/mathematics_concepts.jsonl") as f:
    for line in f:
        concepts.update(json.loads(line)["concepts"])
        
concepts = list(concepts)

data = []
N = 10

for i in range(N):
    # 1. Sample 3 concepts
    c = random.sample(concepts, 3)
    c_str = " | ".join(c)

    # Generate both rationale (z) and problem (x) from concepts (c)
    # Format matches training: "Concepts: {c}\nRationale:" -> model generates z, then "\nProblem:", then x
    prompt = f"Concepts: {c_str}\nRationale:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(prompt_model.device)
    
    with torch.no_grad():
        # Generate full sequence: rationale + problem
        output_ids = prompt_model.generate(
            **inputs,
            max_new_tokens=384,  # Enough for both rationale and problem
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode full output
    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract rationale (z) - everything after "Rationale:" and before "Problem:"
    if "Problem:" in full_output:
        z = full_output.split("Rationale:")[-1].split("Problem:")[0].strip()
        # Extract problem (x) - everything after "Problem:"
        x = full_output.split("Problem:")[-1].strip()
    else:
        # If model didn't generate "Problem:", take everything after "Rationale:" as rationale
        # and skip problem generation for this sample
        z = full_output.split("Rationale:")[-1].strip()
        x = ""
        print(f"Warning: Generated {i+1} did not produce Problem section")
    
    # Clean up: stop at any chat formatting tokens
    if "<" in x:
        x = x.split("<")[0].strip()
    if "<" in z:
        z = z.split("<")[0].strip()

    data.append({
        "concepts": c,
        "rationale": z,
        "problem": x
    })

    print(f"Generated {i+1}/{N}: rationale length={len(z)}, problem length={len(x)}")

# Final save
output_file = "synthetic_generated.jsonl"
with open(output_file, "w") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\nSaved {len(data)} generated (c, z, x) triples to {output_file}")