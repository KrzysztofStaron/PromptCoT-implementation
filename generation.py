import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from hf_config import HF_REPO_ID, HF_P_BASE_PATH
import datasets

# Model config
MODEL_NAME = "Qwen/Qwen2.5-7B"

# Load prompt model (no quantization for generation)
base_p = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side='left',
    truncation_side='left'
)
tokenizer.pad_token = tokenizer.eos_token

# Load latest adapters
p_latest_path = f"{HF_P_BASE_PATH}latest/"
pθ = PeftModel.from_pretrained(
    base_p,
    HF_REPO_ID,
    is_trainable=False,
    subfolder=p_latest_path.rstrip("/"),
)

# Load first 80 examples from PromptCoT-2.0-Concepts dataset
ds = datasets.load_dataset("xl-zhao/PromptCoT-2.0-Concepts", split="train[:80]")
prompts = [example["prompt"] for example in ds]

BATCH_SIZE = 8
results = []

pθ.eval()
with torch.no_grad():
    for i in range(0, len(prompts), BATCH_SIZE):
        batch_prompts = prompts[i:i + BATCH_SIZE]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(pθ.device)

        outputs = pθ.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for prompt, text in zip(batch_prompts, texts):
            results.append({"prompt": prompt, "generation": text})

with open("generated_prompts.jsonl", "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
