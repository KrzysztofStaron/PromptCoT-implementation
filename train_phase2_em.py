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
import shutil
import glob
from transformers import AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from unsloth import FastLanguageModel
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, create_repo, snapshot_download
from dotenv import load_dotenv
import wandb
from trl import SFTTrainer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from hf_config import HF_REPO_ID, HF_VERSION

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- Config ---
MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-7B"
MAX_SEQ_LENGTH = 8192
DTYPE = None
LOAD_IN_4BIT = True

EM_ITERS = 6
BATCH_SIZE = 8 # Adjusted for H100 & 7B model size
GRAD_ACCUM = 4
NUM_TRIPLETS = 1000

# Paths
# We load models from HuggingFace directly
HF_JOINT_SUBFOLDER = f"{HF_VERSION}/joint"
HF_COLD_START_Q_SUBFOLDER = f"{HF_VERSION}/q/cold-start"

# Local cache/output paths for iterations
OUTPUT_DIR_BASE = f"./models/{HF_VERSION}"

# HF Paths for uploads
HF_P_BASE = f"{HF_VERSION}/p/"
HF_Q_BASE = f"{HF_VERSION}/q/"

# --- Cleanup Old Iterations ---
def cleanup_old_iterations(base_dir, keep_count=3):
    """Keep only the keep_count newest iteration directories, delete older ones."""
    if not os.path.exists(base_dir):
        return
    
    # Find all iter-* directories
    iter_dirs = glob.glob(os.path.join(base_dir, "iter-*"))
    
    if len(iter_dirs) <= keep_count:
        return
    
    # Extract iteration numbers and sort
    def get_iter_num(path):
        basename = os.path.basename(path)
        try:
            return int(basename.split("-")[1])
        except (IndexError, ValueError):
            return -1
    
    iter_dirs.sort(key=get_iter_num, reverse=True)
    
    # Delete oldest ones (keep only keep_count newest)
    for old_dir in iter_dirs[keep_count:]:
        if os.path.exists(old_dir):
            print(f"Deleting old iteration: {old_dir}")
            shutil.rmtree(old_dir)

# --- Sampling Schedule ---
def get_k_samples(iteration):
    if iteration <= 2: return 3
    if iteration <= 4: return 6
    return 10

# --- Structure Check & Reward ---
def check_structure_and_tags(text):
    """Check if rationale text is non-empty and reasonable."""
    if not text or len(text.strip()) < 10:
        return False
    return True

def compute_reward(model, tokenizer, c, x, z):
    """Compute reward as negative log-likelihood of the full sequence."""
    if not check_structure_and_tags(z):
        return -10.0
        
    try:
        full_text = f"Concepts: {c}\nRationale: {z}\nProblem: {x}"
        inputs = tokenizer(full_text, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
        return -loss.item()
        
    except Exception as e:
        log.error(f"Reward comp error: {e}")
        return -10.0

# --- Data Loader ---
def load_initial_triples():
    ds = load_dataset("xl-zhao/PromptCoT-Problem-Generation-Dataset", split=f"train[:{NUM_TRIPLETS}]")
    
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
    return triples[:NUM_TRIPLETS] # Limit to NUM_TRIPLETS for feasibility within 14h on single H100

# --- Helper Functions ---

def run_e_step_generation(triples, k, current_q_subfolder):
    """Run E-Step: Generate Rationales using q_phi (vLLM)."""
    print("E-Step: Generating Rationales...")
    
    # Prepare prompts for vLLM: "Concepts: {c}\nProblem: {x}\nRationale:"
    vllm_prompts = [f"Concepts: {t['concepts']}\nProblem: {t['problem']}\nRationale:" for t in triples]
    
    # Generate with vLLM (restart each iteration to pick up new LoRA adapters)
    llm = LLM(model=MODEL_NAME, enable_lora=True, max_lora_rank=128) 
    
    # Download adapter from HF if not available locally
    if not os.path.exists(current_q_subfolder):
         print(f"Downloading adapter from {HF_REPO_ID} subfolder {current_q_subfolder}...")
         try:
             q_adapter_path = snapshot_download(repo_id=HF_REPO_ID, allow_patterns=f"{current_q_subfolder}/*")
             q_adapter_path = os.path.join(q_adapter_path, current_q_subfolder)
         except Exception as e:
             print(f"Warning: Could not download {current_q_subfolder}: {e}")
             q_adapter_path = current_q_subfolder
    else:
         q_adapter_path = current_q_subfolder
    
    lora_req = LoRARequest("q_adapter", 1, q_adapter_path)
    
    params = SamplingParams(n=k, temperature=0.8, top_p=0.95, max_tokens=768, stop=["\nProblem:", "Problem:"])
    
    outputs = llm.generate(vllm_prompts, params, lora_request=lora_req)
    
    # Collect candidates: triples[i] -> [z1, z2, ..., zk]
    candidates = []
    for output in outputs:
        z_list = [o.text.strip() for o in output.outputs] 
        candidates.append(z_list)
        
    # Clean up vLLM to free VRAM for M-step training
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    
    return candidates

def run_e_step_selection(triples, candidates, current_p_subfolder):
    """Run E-Step: Select Best Rationales using p_theta."""
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
    
    # Load adapter
    print(f"Loading p_theta adapter from {HF_REPO_ID} subfolder {current_p_subfolder}")
    try:
         # Check if local or HF
         if os.path.exists(current_p_subfolder):
             p_model.load_adapter(current_p_subfolder)
         else:
             p_model.load_adapter(HF_REPO_ID, subfolder=current_p_subfolder, adapter_name="p_adapter")
             p_model.set_adapter("p_adapter")
    except Exception as e:
         print(f"Error loading p_theta: {e}")

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
    
    return new_triples

def run_training_step(texts, base_adapter_subfolder, output_path, run_name):
    """Run a single training step (SFT) for either p_theta or q_phi."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH, dtype=DTYPE, load_in_4bit=LOAD_IN_4BIT
    )
    
    loaded_adapter = False
    try:
        # Try loading existing adapter
        if os.path.exists(base_adapter_subfolder):
            model.load_adapter(base_adapter_subfolder)
            loaded_adapter = True
        else:
            # Try HF load
            try:
                 model.load_adapter(HF_REPO_ID, subfolder=base_adapter_subfolder)
                 loaded_adapter = True
            except:
                 pass
    except Exception as e:
        print(f"Adapter load failed: {e}")

    if loaded_adapter:
        print(f"Resumed adapter from {base_adapter_subfolder}")
        FastLanguageModel.for_training(model)
    else:
        print("Initializing NEW LoRA adapter")
        # Add LoRA config
        model = FastLanguageModel.get_peft_model(
            model,
            r=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                            "gate_proj", "up_proj", "down_proj",
                            ],
            lora_alpha=32,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
        )
    
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
        
        # Also upload to latest
        hf_latest_path = f"{HF_VERSION}/{'p' if 'p_' in run_name else 'q'}/latest"
        api.upload_folder(folder_path=output_path, repo_id=HF_REPO_ID, path_in_repo=hf_latest_path, repo_type="model")
    
    del model, trainer
    torch.cuda.empty_cache()

def run_m_step_training(new_triples, current_p_subfolder, current_q_subfolder, iteration):
    """Run M-Step: Train p_theta and q_phi."""
    print("M-Step: Training...")
    
    # Prepare Dataset
    # p_theta data: "Concepts: {c}\nRationale: {z}\nProblem: {x}"
    p_texts = [f"Concepts: {t['concepts']}\nRationale: {t['rationale']}\nProblem: {t['problem']}" for t in new_triples]
    # q_phi data: "Concepts: {c}\nProblem: {x}\nRationale: {z}"
    q_texts = [f"Concepts: {t['concepts']}\nProblem: {t['problem']}\nRationale: {t['rationale']}" for t in new_triples]
    
    # Paths for new iter
    # Local paths for saving
    next_p_path = f"./models/{HF_VERSION}/p/iter-{iteration}"
    next_q_path = f"./models/{HF_VERSION}/q/iter-{iteration}"
    
    # Train both models
    run_training_step(p_texts, current_p_subfolder, next_p_path, f"p_iter{iteration}")
    run_training_step(q_texts, current_q_subfolder, next_q_path, f"q_iter{iteration}")
    
    # Cleanup old iterations (keep only 3 newest)
    cleanup_old_iterations(f"./models/{HF_VERSION}/p", keep_count=3)
    cleanup_old_iterations(f"./models/{HF_VERSION}/q", keep_count=3)
    
    # Return paths for next iteration
    return next_p_path, next_q_path

# --- Main EM Loop ---
def main():
    # Setup paths - initial iteration uses HF paths
    current_q_subfolder = HF_COLD_START_Q_SUBFOLDER
    current_p_subfolder = HF_JOINT_SUBFOLDER # p_theta starts as the joint model from Phase 0
    
    # Load Triples
    triples = load_initial_triples()
    
    for iteration in range(1, EM_ITERS + 1):
        print(f"\n=== EM Iteration {iteration} ===")
        k = get_k_samples(iteration)
        print(f"Sampling k={k}")
        
        # --- E-Step: Generate Rationales using q_phi ---
        candidates = run_e_step_generation(triples, k, current_q_subfolder)
        
        # --- E-Step: Reward & Selection ---
        new_triples = run_e_step_selection(triples, candidates, current_p_subfolder)
        
        # --- M-Step: Train p_theta and q_phi ---
        current_p_subfolder, current_q_subfolder = run_m_step_training(
            new_triples, current_p_subfolder, current_q_subfolder, iteration
        )
        
    print("EM Loop Complete!")

if __name__ == "__main__":
    main()
