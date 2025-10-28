
# Collect 100+ real problems
# "Annotate (c, z, x)"
# Train qφ and pθ on seed
# "Implement reward R(c,x,z)"
# Run EM loop (E → M)
# Generate 100k+ problems
# Add verification (SymPy / pytest)
# Run self-play or SFT
# """"
# Components:
# 1. Seed triplets (c, z, x)
# 2. Rationale model qφ(z|c,x)
# 3. Prompt model pθ(x|z,c)
# 4. EM loop with reward
# 5. Self-play or SFT with verification
# """"

# 1. Seed triplets

import json
from transformers import AutoModelForCausalLM

with open("./data/MATH500.jsonl", "r") as f:
    MATH500 = [json.loads(line) for line in f]

# Seed dataset
SEED_TRIPLETS = MATH500

# qφ(z|c,x)
rationale_model = AutoModelForCausalLM.from_pretrained("gpt2")

# pθ(x|z,c)
prompt_model = AutoModelForCausalLM.from_pretrained("gpt2")


# finetune data
rationale_finetune_data = []
prompt_finetune_data = []

# finetune qφ(z|c,x) and pθ(x|z,c) on SEED_TRIPLETS
for item in SEED_TRIPLETS:
    concepts = item["concepts"]
    problem = item["problem"]
    rationale = item["rationale"]

    rationale_input = f"Concepts: {concepts}\nProblem: {problem}\nRationale:"
    rationale_output = rationale
    prompt_input = f"Concepts: {concepts}\nRationale: {rationale}\nProblem:"
    prompt_output = problem

    rationale_finetune_data.append({"prompt": rationale_input, "output": rationale_output})
    prompt_finetune_data.append({"prompt": prompt_input, "output": prompt_output})


    break

