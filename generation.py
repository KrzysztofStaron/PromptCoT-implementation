# generate.py — COPY-PASTE
import json
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", padding_side='left')
tokenizer.pad_token = tokenizer.eos_token

pθ = PeftModel.from_pretrained(base, "PanzerBread/promptcot-p", subfolder="latest")
qφ = PeftModel.from_pretrained(base, "PanzerBread/promptcot-q", subfolder="latest")

# Load all concepts
concepts = set()
with open("./data/annotated.jsonl") as f:
    for line in f:
        concepts.update(json.loads(line)["concepts"])
concepts = list(concepts)

data = []
for i in range(100000):
    c = random.sample(concepts, 3)
    prompt = f"Concepts: {' | '.join(c)}\nProblem: x\nRationale:"
    z = qφ.generate(tokenizer(prompt, return_tensors="pt").to(qφ.device).input_ids, max_new_tokens=64, do_sample=True)[0]
    z = tokenizer.decode(z, skip_special_tokens=True).split("Rationale:")[-1].strip()
    
    full = f"Concepts: {' | '.join(c)}\nRationale: {z}\nProblem:"
    x = pθ.generate(tokenizer(full, return_tensors="pt").to(pθ.device).input_ids, max_new_tokens=256, do_sample=True)[0]
    x = tokenizer.decode(x, skip_special_tokens=True).split("Problem:")[-1].strip()
    
    data.append({"concepts": c, "rationale": z, "problem": x})
    if i % 1000 == 0:
        print(f"Generated {i}/100000")

with open("synthetic_100k.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")