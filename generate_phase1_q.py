# generate_phase1_q.py
# Generate Rationale using Phase 1 q_phi model
# Input: Concepts + Problem -> Output: Rationale

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import datasets
import json
import os
import re
from huggingface_hub import snapshot_download
from hf_config import HF_REPO_ID, HF_VERSION

# Phase 1 q_phi model path
HF_Q_PHASE1_PATH = f"{HF_VERSION}/q/cold-start"

def main():
    # 1. Download LoRA adapter locally
    lora_subfolder = HF_Q_PHASE1_PATH.strip("/")
    print(f"Downloading Phase 1 q_phi LoRA from {HF_REPO_ID}/{lora_subfolder} ...")
    lora_local_path = snapshot_download(
        repo_id=HF_REPO_ID,
        allow_patterns=[f"{lora_subfolder}/**"],
        local_dir="/workspace/lora_cache",
        local_dir_use_symlinks=False,
        tqdm_class=None,
    )
    lora_adapter_path = os.path.join(lora_local_path, lora_subfolder)
    print(f"LoRA ready at: {lora_adapter_path}")

    # 2. Load base model + LoRA
    llm = LLM(
        model="unsloth/DeepSeek-R1-Distill-Qwen-7B",
        dtype="bfloat16",
        max_model_len=8192,  # Matches training MAX_SEQ_LENGTH
        gpu_memory_utilization=0.97,
        trust_remote_code=True,
        enforce_eager=True,
        enable_lora=True,
        max_loras=8,
        max_cpu_loras=16,
        max_lora_rank=128,
        tensor_parallel_size=1,
    )

    # 3. LoRA request
    lora_request = LoRARequest("q_phi_lora", 1, lora_adapter_path)

    # 4. Load dataset (using concepts dataset for testing)
    ds = datasets.load_dataset("xl-zhao/PromptCoT-2.0-Concepts", split="train[:20]")
    
    prompts = []
    for ex in ds:
        # Format: Concepts + Problem -> Rationale
        # We need a problem, so we'll use a placeholder or extract from dataset if available
        concepts = " | ".join(ex['foundational_concepts'])
        # For demo, using a simple problem prompt
        # In practice, you'd have actual problems from your dataset
        problem_text = "Given an array of integers, find the maximum sum of a contiguous subarray."
        
        prompt = f"Concepts: {concepts}\nProblem: {problem_text}\nRationale:"
        prompts.append(prompt)

    # 5. Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,  # Rationale should be shorter than full problem
        repetition_penalty=1.1,
        stop_token_ids=[151645],  # Qwen </s> token
        skip_special_tokens=True,
    )

    # 6. Generate rationales
    print(f"Generating rationales for {len(prompts)} examples...")
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    # 7. Save results
    results = []
    with open("generated_rationales.jsonl", "w", encoding="utf-8") as f:
        for prompt, out in zip(prompts, outputs):
            rationale = out.outputs[0].text.strip()
            
            # Extract concepts and problem from prompt
            concepts_match = re.search(r"Concepts:\s*(.*?)\nProblem:", prompt, re.DOTALL)
            problem_match = re.search(r"Problem:\s*(.*?)\nRationale:", prompt, re.DOTALL)
            
            concepts = concepts_match.group(1).strip() if concepts_match else ""
            problem = problem_match.group(1).strip() if problem_match else ""
            
            item = {
                "concepts": concepts,
                "problem": problem,
                "rationale": rationale,
                "full_prompt": prompt,
            }
            results.append(item)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"\nDone! Generated {len(outputs)} rationales.")
    print(f"Results saved to generated_rationales.jsonl")
    
    # Print first example
    if results:
        print("\n--- Example Output ---")
        print(f"Concepts: {results[0]['concepts']}")
        print(f"Problem: {results[0]['problem']}")
        print(f"Rationale: {results[0]['rationale']}")


if __name__ == "__main__":
    main()

