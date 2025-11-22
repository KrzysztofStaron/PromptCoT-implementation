# generate_phase1_q.py
# Generate using Phase 0 joint model
# Input: Concepts -> Output: Rationale + Problem
# Format: "Concepts: {c}\nRationale: {r}\nProblem: {p}"

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import datasets
import json
import os
import re
from huggingface_hub import snapshot_download
from hf_config import HF_REPO_ID, HF_VERSION

# Phase 0 joint model path
HF_PHASE0_JOINT_PATH = f"{HF_VERSION}/joint"

def main():
    # 1. Download LoRA adapter locally
    lora_subfolder = HF_PHASE0_JOINT_PATH.strip("/")
    print(f"Downloading Phase 0 joint LoRA from {HF_REPO_ID}/{lora_subfolder} ...")
    try:
        lora_local_path = snapshot_download(
            repo_id=HF_REPO_ID,
            allow_patterns=[f"{lora_subfolder}/**"],
            local_dir="/workspace/lora_cache",
            local_dir_use_symlinks=False,
            tqdm_class=None,
        )
        lora_adapter_path = os.path.join(lora_local_path, lora_subfolder)
        
        # Verify adapter files exist
        adapter_config_path = os.path.join(lora_adapter_path, "adapter_config.json")
        if not os.path.exists(adapter_config_path):
            # Check what files actually exist
            if os.path.exists(lora_adapter_path):
                existing_files = os.listdir(lora_adapter_path)
                print(f"\nFound directory but missing adapter_config.json")
                print(f"Files in {lora_adapter_path}: {existing_files}")
            else:
                print(f"\nDirectory does not exist: {lora_adapter_path}")
            
            raise FileNotFoundError(
                f"\n❌ LoRA adapter not found at {lora_adapter_path}\n"
                f"Expected file: {adapter_config_path}\n"
                f"\nPhase 0 joint model may not be trained/uploaded yet.\n"
                f"Please run: python train_phase0_joint.py\n"
                f"And ensure it uploads to: {HF_REPO_ID}/{lora_subfolder}"
            )
        print(f"✅ LoRA ready at: {lora_adapter_path}")
    except Exception as e:
        print(f"\n❌ ERROR: Failed to load LoRA adapter")
        print(f"Path: {HF_REPO_ID}/{lora_subfolder}")
        print(f"Error: {e}")
        print(f"\n💡 Solution: Train Phase 1 model first:")
        print(f"   python train_phase1_q.py")
        print(f"\n   Or check if the model exists at: {HF_REPO_ID}/{lora_subfolder}")
        return

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
    lora_request = LoRARequest("phase0_joint_lora", 1, lora_adapter_path)

    # 4. Test prompts (user-provided)
    test_prompts = [
        "The capital of France is",
        "Write a Python function to reverse a string:",
        "Solve the equation 3x + 5 = 20 for x.",
        "Explain quantum computing like I'm 10 years old."
    ]
    
    # 5. Format prompts for Phase 0 joint model (Concepts -> Rationale + Problem)
    # Phase 0 was trained on: "Concepts: {c}\nRationale: {r}\nProblem: {p}"
    # So input should be: "Concepts: {c}\nRationale:"
    prompts = []
    for test_prompt in test_prompts:
        # For testing, we'll use generic concepts or extract from the prompt
        # In practice, you'd have actual concepts from your dataset
        prompt = f"Concepts: general knowledge | problem solving\nRationale:"
        prompts.append(prompt)
    
    # Also include dataset examples if available
    try:
        ds = datasets.load_dataset("xl-zhao/PromptCoT-2.0-Concepts", split="train[:5]")
        for ex in ds:
            concepts = " | ".join(ex['foundational_concepts'])
            prompt = f"Concepts: {concepts}\nRationale:"
            prompts.append(prompt)
    except Exception as e:
        print(f"Note: Could not load dataset examples: {e}")
        print("Using only test prompts.")

    # 6. Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=2200,  # Enough for Rationale + full Problem with samples
        repetition_penalty=1.1,
        stop_token_ids=[151645],  # Qwen </s> token
        skip_special_tokens=True,
    )

    # 7. Generate (Phase 0 generates Rationale + Problem)
    print(f"Generating Rationale + Problem for {len(prompts)} examples...")
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    # 8. Save results
    results = []
    with open("generated_outputs.jsonl", "w", encoding="utf-8") as f:
        for prompt, out in zip(prompts, outputs):
            generated_text = out.outputs[0].text.strip()
            
            # Extract concepts from prompt
            concepts_match = re.search(r"Concepts:\s*(.*?)\nRationale:", prompt, re.DOTALL)
            concepts = concepts_match.group(1).strip() if concepts_match else ""
            
            # Parse Rationale and Problem from generated text
            # Format: "Rationale: {r}\nProblem: {p}"
            rationale_match = re.search(r"Rationale:\s*(.*?)(?=\nProblem:)", generated_text, re.DOTALL)
            problem_match = re.search(r"Problem:\s*(.*)", generated_text, re.DOTALL)
            
            rationale = rationale_match.group(1).strip() if rationale_match else ""
            problem = problem_match.group(1).strip() if problem_match else ""
            
            item = {
                "concepts": concepts,
                "rationale": rationale,
                "problem": problem,
                "full_generation": generated_text,
                "full_prompt": prompt,
            }
            results.append(item)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"\nDone! Generated {len(outputs)} outputs (Rationale + Problem).")
    print(f"Results saved to generated_outputs.jsonl")
    
    # Print all test prompt results
    print("\n" + "="*60)
    print("TEST PROMPT RESULTS:")
    print("="*60)
    for i, result in enumerate(results[:len(test_prompts)], 1):
        print(f"\n--- Test Prompt {i} ---")
        print(f"Input: {test_prompts[i-1]}")
        print(f"Concepts: {result['concepts']}")
        print(f"\nGenerated Rationale:\n{result['rationale']}")
        print(f"\nGenerated Problem:\n{result['problem']}")
        print("-"*60)


if __name__ == "__main__":
    main()

