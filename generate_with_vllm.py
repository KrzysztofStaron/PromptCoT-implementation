# generate_with_vllm.py
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import datasets
import json
import os
from hf_config import HF_REPO_ID, HF_P_BASE_PATH


def main():
    # 1. Download LoRA adapters locally (vLLM doesn't support HF subfolders)
    from huggingface_hub import snapshot_download
    lora_hf_path = f"{HF_REPO_ID}"  # Base repo: "PanzerBread/PromptCoT"
    lora_subfolder = f"{HF_P_BASE_PATH}latest"  # Subfolder: "coding-0.1/p/latest"
    
    print(f"Downloading LoRA adapters from {lora_hf_path}/{lora_subfolder}...")
    lora_local_path = snapshot_download(
        repo_id=lora_hf_path,
        allow_patterns=f"{lora_subfolder}/*",
        local_dir="/workspace/lora_cache",
        local_dir_use_symlinks=False
    )
    lora_adapter_path = os.path.join(lora_local_path, lora_subfolder)
    print(f"LoRA adapters downloaded to: {lora_adapter_path}")
    
    # 2. Load base model + LoRA on H200 (bf16 for compatibility)
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
        max_lora_rank=128,              # match your LoRA adapters' rank
    )

    # 3. Create LoRA request with local path
    lora_request = LoRARequest("prompt_lora", 1, lora_adapter_path)

    # 4. Load dataset
    ds = datasets.load_dataset("xl-zhao/PromptCoT-2.0-Concepts", split="train[:80]")
    prompts = [example["prompt"] for example in ds]

    # 5. Generation settings
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
        skip_special_tokens=True,
    )

    # 6. Generate all at once — vLLM will continuous-batch internally
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    # 7. Save as JSONL
    with open("generated_prompts.jsonl", "w", encoding="utf-8") as f:
        for prompt, out in zip(prompts, outputs):
            item = {
                "prompt": prompt,
                "generation": out.outputs[0].text,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()