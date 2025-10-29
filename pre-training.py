import json

# For now all it does is prepare the data for finetuning

def write_jsonl(data, filename):
    """Write data to a JSONL file"""
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

with open("./data/annotated.jsonl", "r") as f:
    MATH500 = [json.loads(line) for line in f]

# Seed dataset
SEED_TRIPLETS = MATH500

# finetune data
rationale_finetune_data = []
prompt_finetune_data = []

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


write_jsonl(rationale_finetune_data, "./data/rationale_finetune_data.jsonl")
write_jsonl(prompt_finetune_data, "./data/prompt_finetune_data.jsonl")

# TODO:pre-train qφ(z|c,x) and pθ(x|z,c) on SEED_TRIPLETS
