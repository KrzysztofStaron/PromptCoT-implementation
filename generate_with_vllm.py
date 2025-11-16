# generate_with_vllm.py
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import datasets
import json
import os
from hf_config import HF_REPO_ID, HF_P_BASE_PATH


def main():
    # 1. Load your model + LoRA on H200 (bf16 for compatibility)
    # Use local path if model is pre-downloaded, otherwise HF model ID
    model_path = "/workspace/models/Qwen2.5-7B" if os.path.exists("/workspace/models/Qwen2.5-7B") else "Qwen/Qwen2.5-7B"
    llm = LLM(
        model=model_path,
        dtype="bfloat16",               # supported dtype on your vLLM version
        max_model_len=32768,
        gpu_memory_utilization=0.97,
        trust_remote_code=True,
        enforce_eager=True,             # stable on single GPU
        enable_lora=True,
        max_loras=1,
        max_cpu_loras=4,
    )

    # 2. Point to your LoRA on HuggingFace (same repo layout as training)
    lora_path = f"{HF_REPO_ID}/{HF_P_BASE_PATH}latest"   # e.g. "PanzerBread/PromptCoT/coding-0.1/p/latest"
    lora_request = LoRARequest("prompt_lora", 1, lora_path)

    # 3. Load dataset
    ds = datasets.load_dataset("xl-zhao/PromptCoT-2.0-Concepts", split="train[:80]")
    prompts = [example["prompt"] for example in ds]

    # 4. Generation settings (same as your HF script)
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
        skip_special_tokens=True,
    )

    # 5. Generate all at once — vLLM will continuous-batch internally
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    # 6. Save as JSONL
    with open("generated_prompts.jsonl", "w", encoding="utf-8") as f:
        for prompt, out in zip(prompts, outputs):
            item = {
                "prompt": prompt,
                "generation": out.outputs[0].text,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()