# generate_with_vllm.py
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import datasets
import json
import os
import re
from huggingface_hub import snapshot_download
from hf_config import HF_REPO_ID, HF_P_BASE_PATH


# ──────────────────────────────────────────────────────────────
#  FINAL WINNING PROMPT (2025 meta – this is what actually works)
# ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a legendary Codeforces problem setter.

Your response must follow this exact format and nothing else:

Concepts: <3–6 concepts separated by | >
Rationale: <one short paragraph of sharp reasoning, no numbers, no repetition, no "finally">
Problem:

<the full competitive programming problem in perfect Codeforces LaTeX with input/output format and samples>

Start directly with "Concepts:" and end after the last sample explanation."""

def extract_clean_problem(text):
    match = re.search(r"Problem:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()

def main():
    # 1. Download LoRA locally
    lora_subfolder = f"{HF_P_BASE_PATH}latest".strip("/")
    print(f"Downloading LoRA from {HF_REPO_ID}/{lora_subfolder} ...")
    lora_local_path = snapshot_download(
        repo_id=HF_REPO_ID,
        allow_patterns=[f"{lora_subfolder}/**"],
        local_dir="/workspace/lora_cache",
        local_dir_use_symlinks=False,
        tqdm_class=None,
    )
    lora_adapter_path = os.path.join(lora_local_path, lora_subfolder)
    print(f"LoRA ready at: {lora_adapter_path}")

    # 2. Load model + LoRA (single H200, max performance)
    llm = LLM(
        model="Qwen/Qwen2.5-7B",
        dtype="bfloat16",
        max_model_len=32768,
        gpu_memory_utilization=0.97,
        trust_remote_code=True,
        enforce_eager=True,
        enable_lora=True,
        max_loras=8,
        max_cpu_loras=16,
        max_lora_rank=128,
        tensor_parallel_size=1,          # 1x H200
    )

    # 3. LoRA request
    lora_request = LoRARequest("prompt_lora", 1, lora_adapter_path)

    # 4. Load dataset
    ds = datasets.load_dataset("xl-zhao/PromptCoT-2.0-Concepts", split="train[:80]")
    
    prompts = []

    for ex in ds:
        prompt = f"Concepts: {' | '.join(ex['foundational_concepts'])}"
        prompts.append(SYSTEM_PROMPT + "\n\n" + prompt)

    sampling_params = SamplingParams(
        temperature=0.42,           # slightly higher = more creative
        top_p=0.94,
        max_tokens=2200,            # enough for full problem + 3 samples
        repetition_penalty=1.16,    # gentle – we don't want to fight the format
        frequency_penalty=0.08,
        presence_penalty=0.10,
        stop_token_ids=[151645],  # Qwen2.5 </s> token – prevents runaway
        skip_special_tokens=True,
    )

    # 6. Generate (continuous batching = 80 prompts in ~20-30 seconds on one H200)
    print(f"Generating {len(prompts)} problems with exorcism mode enabled...")
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    # 7. Save + print a quick summary
    good = 0
    with open("generated_clean_problems.jsonl", "w", encoding="utf-8") as f:
        for prompt, out in zip(raw_prompts, outputs):
            text = out.outputs[0].text.strip()
            clean_problem = extract_clean_problem(text)
            is_clean = "Rationale" not in clean_problem and len(clean_problem) > 500
            if is_clean:
                good += 1

            item = {
                "original_prompt": prompt,
                "full_generation": text,
                "clean_problem": clean_problem,
                "is_perfect": is_clean
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nDone! Generated {len(outputs)} problems.")
    print(f"Approximately {good} look clean and high-quality (you'll still want to cherry-pick the gems).")
    print("Check generated_clean_problems.jsonl – the real Div.1 diamonds are in there.")


if __name__ == "__main__":
    main()