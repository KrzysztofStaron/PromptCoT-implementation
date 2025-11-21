"""
Cold Start Data Generation Script for PromptCoT 2.0

This script streams the PromptCoT-2.0-Concepts dataset and generates olympiad-level
coding problems using the PromptCoT-2.0-Prompt-Generation-Model.

It automatically detects and uses the best available backend:
1. vLLM (Fastest, best for batching)
2. Unsloth (Fast inference)
3. Transformers (Fallback)

Usage:
    python coldStartData.py --max_problems 100 --output_file generated_problems.jsonl --batch_size 10
"""

import json
import argparse
import os

# IMPORTANT: Unsloth must be imported before other libraries to patch correctly
# We try to import it very early, even if we might not use it if vLLM works
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

from datasets import load_dataset
import torch

# Conditional imports for different backends
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

MODEL_NAME = "xl-zhao/PromptCoT-2.0-Prompt-Generation-Model"

# Force structured output with mandatory detailed rationale
PROMPT_SUFFIX = """

Output strictly in the following format:

Rationale:
<your detailed step-by-step reasoning here – explain exactly how you connected ALL the given concepts, why the combination is novel/difficult, and how the problem tests deep understanding of each concept. Be extremely thorough.>

Problem Title:
<a concise, creative title>

Problem Statement:
<the full problem – make it worthy of Codeforces 2800+ rating>"""

class BaseGenerator:
    def __init__(self, model_name):
        self.model_name = model_name

    def generate(self, prompts):
        raise NotImplementedError

class VLLMGenerator(BaseGenerator):
    def __init__(self, model_name):
        super().__init__(model_name)
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM not available")
        try:
            
            num_gpus = torch.cuda.device_count()
            print(f"VLLM: Detected {num_gpus} GPUs. Enabling Tensor Parallelism.")
            
            self.llm = LLM(
                model=model_name, 
                trust_remote_code=True,
                tensor_parallel_size=num_gpus,
                
                # Critical optimization for 4x RTX 5090:
                kv_cache_dtype="fp8_e5m2",       # Drops KV cache memory ~4x vs fp16
                block_size=32,                   # Blackwell architecture sweet spot
                enable_chunked_prefill=True,     # Allows huge prefill batches
                max_num_batched_tokens=65536,    # 64k tokens per iteration
                max_num_seqs=4096,               # Max concurrent sequences
                
                # Reduced slightly from 0.98 to avoid OOM on startup
                gpu_memory_utilization=0.95,
                max_model_len=32768,
                enforce_eager=False
            )
            self.sampling_params = SamplingParams(
                temperature=0.8,      # ↑ from 0.7 – gives much better reasoning length/quality
                top_p=0.95,
                max_tokens=8192,       # safe for long rationales + huge problems
                skip_special_tokens=True
            )
            print("Initialized vLLM backend.")
        except ImportError:
            raise ImportError("vLLM not available")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize vLLM: {e}")

    def generate(self, prompts):
        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
        return [output.outputs[0].text for output in outputs]

class UnslothGenerator(BaseGenerator):
    def __init__(self, model_name):
        super().__init__(model_name)
        if not UNSLOTH_AVAILABLE:
            raise ImportError("Unsloth not available")
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=32768,
                dtype=None,
                load_in_4bit=False, # Use full precision or bf16 if available
            )
            FastLanguageModel.for_inference(self.model)
            print("Initialized Unsloth backend.")
        except ImportError:
            raise ImportError("Unsloth not available")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Unsloth: {e}")

    def generate(self, prompts):
        
        results = []
        for prompt in prompts:
            inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.8,
                use_cache=True
            )
            results.append(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
        return results

class TransformersGenerator(BaseGenerator):
    def __init__(self, model_name):
        super().__init__(model_name)
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map="auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            )
            print("Initialized Transformers backend.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Transformers: {e}")

    def generate(self, prompts):
        results = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(**inputs, max_new_tokens=4096, temperature=0.8)
            results.append(self.tokenizer.decode(outputs[0], skip_special_tokens=True))
        return results

def get_best_generator(model_name):
    # Try vLLM first
    try:
        return VLLMGenerator(model_name)
    except (ImportError, RuntimeError) as e:
        print(f"vLLM skipped: {e}")

    # Try Unsloth second
    try:
        return UnslothGenerator(model_name)
    except (ImportError, RuntimeError) as e:
        print(f"Unsloth skipped: {e}")

    # Fallback to Transformers
    return TransformersGenerator(model_name)

def main(max_problems=None, output_file="generated_problems.jsonl", batch_size=1):
    # Load dataset - use streaming
    print("Loading dataset: xl-zhao/PromptCoT-2.0-Concepts")
    dataset = load_dataset("xl-zhao/PromptCoT-2.0-Concepts", split="train", streaming=True)

    # Initialize generator
    generator = get_best_generator(MODEL_NAME)

    generated_count = 0

    # Check if output file exists and count existing entries for resuming
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_count = sum(1 for _ in f)
        print(f"Found existing file with {existing_count} entries. Resuming from entry {existing_count}...")
        generated_count = existing_count
        # Skip already processed entries
        dataset = dataset.skip(existing_count)

    batch_data = []

    with open(output_file, 'a', encoding='utf-8') as f:
        for i, example in enumerate(dataset):
            if max_problems and generated_count >= max_problems:
                break

            batch_data.append(example)

            # Process batch when full or at the end
            if len(batch_data) >= batch_size:
                _process_batch(generator, batch_data, f)
                generated_count += len(batch_data)
                print(f"Generated {generated_count} problems so far...")
                batch_data = []

        # Process remaining
        if batch_data and (not max_problems or generated_count < max_problems):
            _process_batch(generator, batch_data, f)
            generated_count += len(batch_data)

    print(f"Successfully generated {generated_count} problems and saved to {output_file}")

def _process_batch(generator, batch_data, file_handle):
    prompts = [ex['prompt'] + PROMPT_SUFFIX for ex in batch_data]
    
    try:
        generated_texts = generator.generate(prompts)
        
        for example, generated_text in zip(batch_data, generated_texts):
            output_entry = {
                "concepts": example['foundational_concepts'],
                "level": example['level'],
                "generated_problem": generated_text
            }
            file_handle.write(json.dumps(output_entry, ensure_ascii=False) + '\n')
        
        file_handle.flush()
    except Exception as e:
        print(f"Error processing batch: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate problems from PromptCoT concepts dataset")
    parser.add_argument("--max_problems", type=int, default=None,
                       help="Maximum number of problems to generate (default: all)")
    parser.add_argument("--output_file", type=str, default="generated_problems.jsonl",
                       help="Output file path")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for generation (2048-3072 recommended for vLLM on 4x RTX 5090)")

    args = parser.parse_args()
    
    # Auto-adjust batch size for vLLM if default is used
    if args.batch_size == 1:
        try:
            # Simple check without importing full vLLM if not needed
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                
                # 4x5090 with fp8 KV cache -> 2048 is conservative and blazing fast
                args.batch_size = 2048
                print(f"4x RTX 5090 detected -> auto batch size = {args.batch_size} (very safe & max throughput)")
        except ImportError:
            pass

    main(max_problems=args.max_problems, output_file=args.output_file, batch_size=args.batch_size)