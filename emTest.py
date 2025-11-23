# emTest.py
# Test models from different EM training iterations

import os
import re
import json
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from huggingface_hub import snapshot_download
from dotenv import load_dotenv
from hf_config import HF_REPO_ID, HF_VERSION

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# --- Config ---
MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-7B"
MAX_SEQ_LENGTH = 8192
DTYPE = None
LOAD_IN_4BIT = True

# Model subfolders to test
MODEL_SUBFOLDERS = {
    0: f"{HF_VERSION}/joint",           # Joint model from Phase 0
    1: f"{HF_VERSION}/p/iter-1",        # p_theta after iteration 1
    2: f"{HF_VERSION}/p/iter-2",        # p_theta after iteration 2
    3: f"{HF_VERSION}/p/iter-3",        # p_theta after iteration 3
}

# Output files
OUTPUT_FILES = {
    0: "output_iter0.jsonl",
    1: "output_iter1.jsonl",
    2: "output_iter2.jsonl",
    3: "output_iter3.jsonl"
}

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

def load_test_dataset(num_examples=10):
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

def load_model_with_adapter(subfolder, adapter_name="test_adapter"):
    """Load base model and adapter from HuggingFace - reused from train_phase2_em.py"""
    print(f"Loading model with adapter from {HF_REPO_ID}/{subfolder}...")

    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        token=HF_TOKEN
    )
    print("  Base model loaded")

    # Apply LoRA adapter structure (matches training configuration)
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
    )

    # Load adapter
    try:
        # Check if local or HF
        if os.path.exists(subfolder):
            adapter_path = subfolder
            print(f"  Adapter found locally: {adapter_path}")
        else:
            # Download adapter from HF if not available locally
            print(f"  Downloading adapter from {HF_REPO_ID} subfolder {subfolder}...")
            downloaded_path = snapshot_download(repo_id=HF_REPO_ID, allow_patterns=f"{subfolder}/*", token=HF_TOKEN)
            adapter_path = os.path.join(downloaded_path, subfolder)
            print("  Adapter downloaded successfully"

        model.load_adapter(adapter_path, adapter_name=adapter_name)
        model.set_adapter(adapter_name)
        print("  Adapter loaded successfully")
    except Exception as e:
        print(f"  Warning: Could not load adapter {subfolder}: {e}")
        print("  Continuing with base model only")

    FastLanguageModel.for_inference(model)
    print("  Model ready for inference")

    return model, tokenizer

def generate_with_model(model, tokenizer, prompt, max_new_tokens=1024, temperature=0.7, top_p=0.9):
    """Generate text using the model - reused from generate_phase0.py"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the output
    generated_text = generated[len(prompt):].strip()

    return generated_text

def test_joint_model(model, tokenizer, test_examples):
    """Test joint model (iter0): Concepts -> Problem + Rationale"""
    print("Testing joint model (iter0)...")
    results = []

    for i, example in enumerate(test_examples):
        print(f"  Testing example {i+1}/{len(test_examples)}")

        # For joint model: generate from "Concepts: {c}\nProblem:" (should produce Problem + Rationale)
        prompt = f"Concepts: {example['concepts']}\nProblem:"
        generated_text = generate_with_model(model, tokenizer, prompt)

        results.append({
            "input_concepts": example['concepts'],
            "generated_output": generated_text,
            "ground_truth_problem": example['problem'],
            "ground_truth_rationale": example['rationale']
        })

    return results

def test_p_theta_model(model, tokenizer, test_examples):
    """Test p_theta model: Concepts + Rationale -> Problem"""
    print("Testing p_theta model...")
    results = []

    for i, example in enumerate(test_examples):
        print(f"  Testing example {i+1}/{len(test_examples)}")

        # For p_theta: generate from "Concepts: {c}\nRationale: {r}\nProblem:" (should produce Problem)
        prompt = f"Concepts: {example['concepts']}\nRationale: {example['rationale']}\nProblem:"
        generated_text = generate_with_model(model, tokenizer, prompt)

        results.append({
            "input_concepts": example['concepts'],
            "input_rationale": example['rationale'],
            "generated_problem": generated_text,
            "ground_truth_problem": example['problem']
        })

    return results

def save_results_to_jsonl(results, filename):
    """Save results to JSONL file"""
    print(f"Saving {len(results)} results to {filename}...")
    with open(filename, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"Results saved to {filename}")

def main():
    """Main function to test all models"""
    print("Starting EM Test Script")
    print("=" * 50)

    # Load test dataset
    test_examples = load_test_dataset(10)

    # Test each model
    for iteration in range(4):
        print(f"\n{'='*50}")
        print(f"Testing Iteration {iteration}")
        print(f"{'='*50}")

        subfolder = MODEL_SUBFOLDERS[iteration]
        output_file = OUTPUT_FILES[iteration]

        try:
            # Load model
            model, tokenizer = load_model_with_adapter(subfolder, f"iter{iteration}_adapter")

            # Test model
            if iteration == 0:
                # Joint model (iter0)
                results = test_joint_model(model, tokenizer, test_examples)
            else:
                # p_theta models (iter1-3)
                results = test_p_theta_model(model, tokenizer, test_examples)

            # Save results
            save_results_to_jsonl(results, output_file)

            # Clean up
            del model, tokenizer
            torch.cuda.empty_cache()
            print(f"Completed testing iteration {iteration}")

        except Exception as e:
            print(f"Error testing iteration {iteration}: {e}")
            continue

    print("\n" + "=" * 50)
    print("EM Test Script Complete!")
    print("Generated files:")
    for i in range(4):
        print(f"  - {OUTPUT_FILES[i]}")

if __name__ == "__main__":
    main()
