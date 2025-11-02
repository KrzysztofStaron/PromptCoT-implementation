# generate_fixed.py — PROMPTCoT 2.0 CORRECT INFERENCE
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

# Load the prompt generation model pθ(x | c)
# Generates problems directly from concepts
pθ = PeftModel.from_pretrained(base, "PanzerBread/promptcot-p", subfolder="latest")

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

    problem_prompt = f"""You are a world-class math and coding problem generator.

Concepts: {c_str}

Generate ONE hard, original, verifiable reasoning problem that REQUIRES the above concepts.
- Math: Must end with "Put your final answer within \\boxed{{}}."
- NEVER write the solution.
- NEVER include assistant tokens.
- Keep under 250 words.
- Output ONLY the problem text.

Problem:"""

    inputs = tokenizer(problem_prompt, return_tensors="pt").to(pθ.device)
    x_ids = pθ.generate(
        **inputs,
        max_new_tokens=384,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id
    )
    x_text = tokenizer.decode(x_ids[0], skip_special_tokens=True)
    x = x_text.split("Problem:")[-1].strip()
    
    # Clean up: stop at any chat formatting tokens
    if "<" in x:
        x = x.split("<")[0].strip()

    data.append({
        "concepts": c,
        "problem": x
    })

    print(f"Generated {i+1}/{N}")

# Final save
with open("synthetic_10.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")