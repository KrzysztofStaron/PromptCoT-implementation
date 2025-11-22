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

    # 4. Test prompts (user-provided)
    test_prompts = [
        "The capital of France is",
        "Write a Python function to reverse a string:",
        "Solve the equation 3x + 5 = 20 for x.",
        "Explain quantum computing like I'm 10 years old."
    ]
    
    # 5. Format prompts for Phase 1 q_phi model (Concepts + Problem -> Rationale)
    # Since these are general prompts, we'll treat them as problems and add placeholder concepts
    prompts = []
    for test_prompt in test_prompts:
        # For testing, we'll use generic concepts or extract from the prompt
        # In practice, you'd have actual concepts from your dataset
        prompt = f"Concepts: general knowledge | problem solving\nProblem: {test_prompt}\nRationale:"
        prompts.append(prompt)
    
    # Also include dataset examples if available
    try:
        ds = datasets.load_dataset("xl-zhao/PromptCoT-2.0-Concepts", split="train[:5]")
        for ex in ds:
            concepts = " | ".join(ex['foundational_concepts'])
            problem_text = "Given an array of integers, find the maximum sum of a contiguous subarray."
            prompt = f"Concepts: {concepts}\nProblem: {problem_text}\nRationale:"
            prompts.append(prompt)
    except Exception as e:
        print(f"Note: Could not load dataset examples: {e}")
        print("Using only test prompts.")

    # 6. Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,  # Rationale should be shorter than full problem
        repetition_penalty=1.1,
        stop_token_ids=[151645],  # Qwen </s> token
        skip_special_tokens=True,
    )

    # 7. Generate rationales
    print(f"Generating rationales for {len(prompts)} examples...")
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    # 8. Save results
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
    
    # Print all test prompt results
    print("\n" + "="*60)
    print("TEST PROMPT RESULTS:")
    print("="*60)
    for i, result in enumerate(results[:len(test_prompts)], 1):
        print(f"\n--- Test Prompt {i} ---")
        print(f"Input: {test_prompts[i-1]}")
        print(f"Concepts: {result['concepts']}")
        print(f"Problem: {result['problem']}")
        print(f"Generated Rationale: {result['rationale']}")
        print("-"*60)


if __name__ == "__main__":
    main()

