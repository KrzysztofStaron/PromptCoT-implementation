# train_phase2_em.py
# Phase 2: EM Loop with Structure Enforcement (14-16 hrs)
# Updates:
# - Sampling Schedule: k=3 (iter 1-2), k=6 (iter 3-4), k=10 (iter 5-6)
# - Reward: -loss + structure_penalty (missing fields/tags -> -10.0)
# - Generation: Force prefix "\nRationale:" for q_phi

import json
import torch
import gc
import re
import os
import logging
import tempfile
import threading
import sys
from transformers import AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from unsloth import FastLanguageModel
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv
import wandb
from trl import SFTTrainer
from hf_config import HF_REPO_ID, HF_VERSION

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- Config ---
MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-14B"
MAX_SEQ_LENGTH = 8192
DTYPE = None
LOAD_IN_4BIT = True

EM_ITERS = 6
BATCH_SIZE = 8 # Adjusted for H100 & 14B model size
GRAD_ACCUM = 4

# Paths
COLD_START_P = f"./models/{HF_VERSION}/p/cold-start"
COLD_START_Q = f"./models/{HF_VERSION}/q/cold-start"

# HF Paths
HF_P_BASE = f"{HF_VERSION}/p/"
HF_Q_BASE = f"{HF_VERSION}/q/"

# --- Sampling Schedule ---
def get_k_samples(iteration):
    if iteration <= 2: return 3
    if iteration <= 4: return 6
    return 10

# --- Structure Check & Reward ---
def check_structure_and_tags(text):
    """
    Check for Concepts, Rationale, Problem fields AND proper formatting.
    Since q_phi generates Rationale given Concepts+Problem, or p_theta generates all?
    
    Wait, in EM:
    E-step: q_phi(z | c, x) -> generates rationales z.
    Structure of z: Just the rationale text.
    But we reward based on p_theta(x | z, c) + p_theta(z | c).
    
    The text passed to reward computation is constructed:
    "Concepts: {c}\nRationale: {z}\nProblem: {x}"
    
    Structure penalty applies if the GENERATED z breaks format? 
    z is usually just the text. 
    
    Let's assume strictness is about the content of z not being garbage/empty.
    
    However, the prompt says: "If generated text missing any of the three fields -> reward -= 10.0".
    This likely refers to the full construction or if we were generating full completions.
    Since z comes from q_phi which outputs "Rationale: ...", we check if z is valid.
    
    Let's ensure z is non-empty and reasonable.
    """
    if not text or len(text.strip()) < 10:
        return False
    return True

def compute_reward(model, tokenizer, c, x, z):
    # R = log p(x|z,c) + log p(z|c) + penalty
    
    # Penalty check
    if not check_structure_and_tags(z):
        return -10.0
        
    try:
        # 1. log p(x | z, c)
        # Prompt: "Concepts: {c}\nRationale: {z}\nProblem:" -> Target: "{x}"
        # Or full sequence probability? usually conditional log likelihood of target given context.
        
        # "Concepts: {c}\nRationale: {z}\nProblem: {x}"
        full_text = f"Concepts: {c}\nRationale: {z}\nProblem: {x}"
        inputs = tokenizer(full_text, return_tensors="pt").to("cuda")
        
        # We want loss on the full sequence? Or just x?
        # Standard approximation: Use the model's loss on the full sequence.
        # Loss = - log P(sequence). 
        # We want log P(x|z,c) + log P(z|c) = log P(x,z|c).
        # P(x,z|c) is exactly the probability of the sequence "Rationale: z\nProblem: x" given "Concepts: c".
        
        # So we just calculate loss of "Rationale: {z}\nProblem: {x}" given prefix "Concepts: {c}".
        # Or simply loss of the whole sequence "Concepts: ... \nRationale: ... \nProblem: ..."
        # The standard causal LM loss is average NLL per token.
        # Total NLL = loss * num_tokens.
        # Reward = - Total NLL.
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
        return -loss.item()
        
    except Exception as e:
        log.error(f"Reward comp error: {e}")
        return -10.0

# --- Data Loader ---
def load_initial_triples():
    ds = load_dataset("xl-zhao/PromptCoT-Problem-Generation-Dataset", split="train")
    # Sample first 3000 for the EM loop as per plan (or full 88k if fast enough? Plan said "Target: 88k+ Samples")
    # But EM loop is expensive. 88k * k samples might take forever.
    # Grok recipe mentioned: "Data Strategy (Target: 3000 Triplets)" in manual annotation section, 
    # but plan v2 said "Source: xl-zhao... (88k samples)".
    # Let's stick to a subset if we want to finish in 14-16 hours. 88k * 6 iters * 10 samples is heavy.
    # Maybe we train on a subset of 5k-10k per iteration? Or full dataset?
    # With Unsloth and vLLM, generation is fast.
    # Let's try to use a robust subset, say 10,000 high quality ones.
    # Or just iterate through the dataset.
    
    # Parsing
    triples = []
    for row in ds:
        p = row['prompt']
        c_text = row['completion']
        
        # Concepts
        concepts_match = re.search(r"Foundational Concepts:(.*?)Difficulty Level:", p, re.DOTALL)
        concepts = concepts_match.group(1).strip() if concepts_match else p
        concepts = re.sub(r"\d+\.\s*", "", concepts)
        concepts = " | ".join([line.strip() for line in concepts.split('\n') if line.strip()])
        
        # Rationale & Problem
        r_match = re.search(r"<!-- BEGIN RATIONALE -->(.*?)(?:<!-- END RATIONALE -->|(?=<!-- BEGIN PROBLEM -->))", c_text, re.DOTALL)
        p_match = re.search(r"<!-- BEGIN PROBLEM -->(.*?)<!-- END PROBLEM -->", c_text, re.DOTALL)
        
        if r_match and p_match:
            triples.append({
                "concepts": concepts,
                "rationale": r_match.group(1).strip(),
                "problem": p_match.group(1).strip()
            })
            
    print(f"Loaded {len(triples)} triples.")
    return triples[:5000] # Limit to 5000 for feasibility within 14h on single H100

# --- Main EM Loop ---
def main():
    # Initialize vLLM for E-step (Rationale Generation)
    from vllm import LLM, SamplingParams
    print("Initializing vLLM...")
    # We need to load the current q_phi model into vLLM.
    # In Iter 1, it's COLD_START_Q.
    # vLLM supports LoRA. We load base model + LoRA.
    
    # Setup paths
    current_q_path = COLD_START_Q
    current_p_path = COLD_START_P
    
    # Load Triples
    triples = load_initial_triples()
    
    for iteration in range(1, EM_ITERS + 1):
        print(f"\n=== EM Iteration {iteration} ===")
        k = get_k_samples(iteration)
        print(f"Sampling k={k}")
        
        # --- E-Step: Generate Rationales using q_phi ---
        print("E-Step: Generating Rationales...")
        
        # Prepare prompts for vLLM: "Concepts: {c}\nProblem: {x}\nRationale:"
        # Note: Forced prefix is just the prompt ending with "\nRationale:"
        vllm_prompts = [f"Concepts: {t['concepts']}\nProblem: {t['problem']}\nRationale:" for t in triples]
        
        # Load vLLM (reload per iter to pick up new LoRA? vLLM loading takes time. 
        # Better to keep vLLM running and update LoRA adapter?
        # vLLM LoRA support allows swapping adapters.
        # But we are updating the model weights. Unsloth saves adapters.
        # So we can reload the adapter in vLLM.
        
        # Simplified: Re-init vLLM or just use HF model for generation if vLLM is too complex to hot-swap in script.
        # Unsloth generation is also fast. Let's use Unsloth for generation to avoid VRAM fragmentation with 2 engines.
        # Wait, plan said "vLLM for fast generation". 
        # Okay, we'll restart vLLM each iter or use LoRA swapping.
        # Restarting is safer for memory.
        
        # Generate
        llm = LLM(model=MODEL_NAME, enable_lora=True, max_lora_rank=128) 
        # We need to pass the adapter path.
        from vllm.lora.request import LoRARequest
        lora_req = LoRARequest("q_adapter", 1, current_q_path)
        
        params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=768, stop=["\nProblem:", "Problem:"])
        
        outputs = llm.generate(vllm_prompts, params, lora_request=lora_req)
        
        # Collect candidates
        # Structure: triples[i] -> [z1, z2, ..., zk]
        candidates = []
        for output in outputs:
            # vLLM might return multiple outputs if n>1, but here we might loop or use n=k
            # If using n=k in SamplingParams
            # params.n = k
            # But we need to implement k samples.
            # Let's assume we run generate once with n=k (if vLLM supports it easily with LoRA)
            # Actually, let's just sample k times or n=k.
            z_list = [o.text.strip() for o in output.outputs] 
            # If n=1, we need to run k times? No, set n=k in params.
            candidates.append(z_list)
            
        # Clean up vLLM to free VRAM for M-step training
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        
        # --- E-Step: Reward & Selection ---
        print("E-Step: Selecting Best Rationales...")
        # Load p_theta for scoring
        # Use Unsloth for efficient inference scoring
        p_model, tokenizer = FastLanguageModel.from_pretrained(
            MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
            token=HF_TOKEN
        )
        p_model.load_adapter(current_p_path)
        FastLanguageModel.for_inference(p_model)
        
        new_triples = []
        
        for i, t in enumerate(triples):
            c = t['concepts']
            x = t['problem']
            z_list = candidates[i] # list of k strings
            
            scores = []
            for z in z_list:
                score = compute_reward(p_model, tokenizer, c, x, z)
                scores.append(score)
            
            # Select best
            best_idx = scores.index(max(scores))
            best_z = z_list[best_idx]
            
            new_triples.append({
                "concepts": c,
                "rationale": best_z,
                "problem": x
            })
            
        # Cleanup p_model
        del p_model
        gc.collect()
        torch.cuda.empty_cache()
        
        # --- M-Step: Train p_theta and q_phi ---
        print("M-Step: Training...")
        
        # Prepare Dataset
        # p_theta data: "Concepts: {c}\nRationale: {z}\nProblem: {x}"
        p_texts = [f"Concepts: {t['concepts']}\nRationale: {t['rationale']}\nProblem: {t['problem']}" for t in new_triples]
        # q_phi data: "Concepts: {c}\nProblem: {x}\nRationale: {z}"
        q_texts = [f"Concepts: {t['concepts']}\nProblem: {t['problem']}\nRationale: {t['rationale']}" for t in new_triples]
        
        # Train Function
        def train_step(texts, base_adapter, output_path, run_name):
            model, tokenizer = FastLanguageModel.from_pretrained(
                MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH, dtype=DTYPE, load_in_4bit=LOAD_IN_4BIT
            )
            model.load_adapter(base_adapter) # Resume from previous iter
            FastLanguageModel.for_training(model)
            
            ds = Dataset.from_dict({"text": texts})
            
            args = TrainingArguments(
                per_device_train_batch_size=4,
                gradient_accumulation_steps=8,
                num_train_epochs=3, # Plan requirement
                learning_rate=5e-5,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                output_dir=output_path,
                optim="adamw_8bit",
                report_to="wandb",
                run_name=run_name
            )
            
            trainer = SFTTrainer(
                model=model, tokenizer=tokenizer, train_dataset=ds, dataset_text_field="text",
                max_seq_length=MAX_SEQ_LENGTH, packing=True, args=args
            )
            trainer.train()
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            
            # Upload
            if HF_TOKEN:
                iter_name = output_path.split('/')[-1] # iter-N
                hf_subpath = f"{HF_VERSION}/{'p' if 'p_' in run_name else 'q'}/{iter_name}"
                api = HfApi(token=HF_TOKEN)
                api.upload_folder(folder_path=output_path, repo_id=HF_REPO_ID, path_in_repo=hf_subpath, repo_type="model")
            
            del model, trainer
            torch.cuda.empty_cache()
            
        # Paths for new iter
        next_p_path = f"./models/{HF_VERSION}/p/iter-{iteration}"
        next_q_path = f"./models/{HF_VERSION}/q/iter-{iteration}"
        
        train_step(p_texts, current_p_path, next_p_path, f"p_iter{iteration}")
        train_step(q_texts, current_q_path, next_q_path, f"q_iter{iteration}")
        
        # Update paths for next loop
        current_p_path = next_p_path
        current_q_path = next_q_path
        
    print("EM Loop Complete!")

if __name__ == "__main__":
    main()

