# emTest.py
# Test models from different EM training iterations

import os
import re
import json
import torch
import threading
from datasets import load_dataset
from huggingface_hub import snapshot_download
from dotenv import load_dotenv
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from hf_config import HF_REPO_ID, HF_VERSION

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# --- Config ---
MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-7B"

# Model subfolders to test
MODEL_SUBFOLDERS = {
    0: "math-0.3/p/cold-start",  # Only run from here
}

OUTPUT_FILES = {
    0: "output_math03_coldstart.jsonl"
}

# Number of examples to test per model
NUM_TEST_EXAMPLES = 8

def parse_promptcot_dataset(examples):
    """Parse PromptCoT dataset examples - reused from train_phase0_p.py"""
    prompts = examples['prompt']
    completions = examples['completion']
    parsed_examples = []

    for p, c in zip(prompts, completions):
        concepts_match = re.search(r"Foundational Concepts:(.*?)Difficulty Level:", p, re.DOTALL)
        if concepts_match:
            concepts_text = concepts_match.group(1).strip()
            concepts_cleaned = re.sub(r"\d+\.\s*", "", concepts_text)
            concepts_cleaned = " | ".join([line.strip() for line in concepts_cleaned.split('\n') if line.strip()])
        else:
            concepts_cleaned = p

        rationale_match = re.search(r"<!-- BEGIN RATIONALE -->(.*?)(?:<!-- END RATIONALE -->|(?=<!-- BEGIN PROBLEM -->))", c, re.DOTALL)
        problem_match = re.search(r"<!-- BEGIN PROBLEM -->(.*?)<!-- END PROBLEM -->", c, re.DOTALL)

        if rationale_match and problem_match:
            rationale = rationale_match.group(1).strip()
            problem = problem_match.group(1).strip()
            parsed_examples.append({
                "concepts": concepts_cleaned,
                "rationale": rationale,
                "problem": problem
            })

    return parsed_examples

def load_test_dataset(num_examples=NUM_TEST_EXAMPLES):
    """Load and parse test dataset"""
    print(f"Loading {num_examples} test examples...")
    ds = load_dataset("xl-zhao/PromptCoT-Problem-Generation-Dataset", split=f"train[:{num_examples}]")

    # Parse examples
    parsed_examples = []
    for example in ds:
        parsed = parse_promptcot_dataset({"prompt": [example["prompt"]], "completion": [example["completion"]]})
        parsed_examples.extend(parsed)

    print(f"Loaded {len(parsed_examples)} test examples")
    return parsed_examples

def get_adapter_path(subfolder):
    """Get adapter path, downloading from HF if needed"""
    if subfolder is None:
        return None

    if os.path.exists(subfolder):
        return subfolder
    else:
        print(f"  Downloading adapter from {HF_REPO_ID} subfolder {subfolder}...")
        try:
            downloaded_path = snapshot_download(repo_id=HF_REPO_ID, allow_patterns=f"{subfolder}/*", token=HF_TOKEN)
            return os.path.join(downloaded_path, subfolder)
        except Exception as e:
            print(f"  Warning: Could not download {subfolder}: {e}")
            return subfolder

def test_single_model(iteration, subfolder, output_file, test_examples):
    """Test a single model using vLLM - runs in parallel thread"""
    try:
        if iteration == -1:
            print(f"Testing Base Model")
        else:
            print(f"Testing Iteration {iteration}")

        # Initialize vLLM engine
        print(f"  Initializing vLLM engine...")
        llm = LLM(
            model=MODEL_NAME,
            enable_lora=(subfolder is not None),
            max_lora_rank=128,
            gpu_memory_utilization=0.8,
            max_num_batched_tokens=24576,
            max_num_seqs=256,
            enable_chunked_prefill=True,
            block_size=16,
            token=HF_TOKEN
        )
        print("  vLLM engine initialized")

        # Get adapter path
        adapter_path = get_adapter_path(subfolder)

        # Create LoRA request if we have an adapter
        lora_request = None
        if adapter_path is not None:
            lora_request = LoRARequest(f"adapter_{iteration}", 1, adapter_path)
            print("  LoRA adapter loaded")

        # Prepare prompts - all models use the same format
        prompts = [f"[CONCEPTS]\n{example['concepts']}\n[/CONCEPTS]\n\n[RATIONALE]\n" for example in test_examples]

        print(f"  Generating for {len(prompts)} prompts...")

        # Generate with vLLM - same parameters for all models
        params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=8192,
        )

        outputs = llm.generate(prompts, params, lora_request=lora_request)

        # Process results - same format for all models
        results = []
        for i, (example, output) in enumerate(zip(test_examples, outputs)):
            generated_text = output.outputs[0].text.strip()

            results.append({
                "input": f"[CONCEPTS]\n{example['concepts']}\n[/CONCEPTS]\n\n[RATIONALE]\n",
                "output": generated_text,
                "ground_truth_problem": example['problem'],
                "ground_truth_rationale": example['rationale']
            })

        # Save results
        save_results_to_jsonl(results, output_file)

        # Clean up
        del llm
        torch.cuda.empty_cache()

        if iteration == -1:
            print("Completed testing base model")
        else:
            print(f"Completed testing iteration {iteration}")

    except Exception as e:
        if iteration == -1:
            print(f"Error testing base model: {e}")
        else:
            print(f"Error testing iteration {iteration}: {e}")

def save_results_to_jsonl(results, filename):
    """Save results to JSONL file"""
    print(f"Saving {len(results)} results to {filename}...")
    with open(filename, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"Results saved to {filename}")

def main():
    """Main function to test all models in parallel"""
    print("Starting EM Test Script (Parallel vLLM)")
    print("=" * 50)

    # Load test dataset
    test_examples = load_test_dataset(NUM_TEST_EXAMPLES)

    # Test all models in parallel
    iterations_to_test = list(MODEL_SUBFOLDERS.keys())
    print("iterations_to_test: ", iterations_to_test)
    threads = []

    for iteration in iterations_to_test:
        subfolder = MODEL_SUBFOLDERS[iteration]
        output_file = OUTPUT_FILES[iteration]

        # Create and start thread for this model
        thread = threading.Thread(
            target=test_single_model,
            args=(iteration, subfolder, output_file, test_examples),
            name=f"Model_{iteration}"
        )
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    print("\n" + "=" * 50)
    print("EM Test Script Complete!")
    print("Generated files:")
    for i in iterations_to_test:
        print(f"  - {OUTPUT_FILES[i]}")

if __name__ == "__main__":
    main()
