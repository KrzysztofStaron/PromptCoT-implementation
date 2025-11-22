# train_phase2_em.py
# Phase 2: EM Loop with Structure Enforcement (14-16 hrs)
# Updates:
# - Sampling Schedule: k=3 (iter 1-2), k=6 (iter 3-4), k=10 (iter 5-6)
# - Reward: -loss + structure_penalty (missing fields/tags -> -10.0)
# - Generation: Force prefix "\nRationale:" for q_phi

from unsloth import FastLanguageModel
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
import argparse
from transformers import AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
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
BATCH_SIZE = 64 # Adjusted for H100 & 7B model size (Increased for higher utilization)
GRAD_ACCUM = 4
NUM_TRIPLETS = 1

# Command-line arguments
parser = argparse.ArgumentParser(description="Train PromptCoT Phase 2 EM Loop")
parser.add_argument("--no-upload", action="store_true", help="Disable uploading to HuggingFace")
args = parser.parse_args()


# Paths
# We load models from HuggingFace directly
HF_COLD_START_P_SUBFOLDER = f"{HF_VERSION}/joint"
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
    """Compute reward as log pθ(x|z,c) + log pθ(z|c).
    
    Reward = -(loss_x + loss_z) where:
    - loss_x = NLL of problem given concepts and rationale
    - loss_z = NLL of rationale given concepts
    """
    if not check_structure_and_tags(z):
        return -100.0  # Large penalty for bad structure
        
    try:
        # Compute log pθ(x | z, c)
        input_x = tokenizer(
            f"Concepts: {c}\nRationale: {z}\nProblem: {x}",
            return_tensors="pt"
        )
        input_x = {k: v.to(model.device) for k, v in input_x.items()}
        
        with torch.no_grad():
            loss_x = model(**input_x, labels=input_x["input_ids"]).loss
        del input_x  # Free memory immediately
        
        # Compute log pθ(z | c)
        input_z = tokenizer(
            f"Concepts: {c}\nRationale: {z}",
            return_tensors="pt"
        )
        input_z = {k: v.to(model.device) for k, v in input_z.items()}
        
        with torch.no_grad():
            loss_z = model(**input_z, labels=input_z["input_ids"]).loss
        del input_z  # Free memory immediately
        
        # Reward = log pθ(x|z,c) + log pθ(z|c) = -(loss_x + loss_z)
        reward = -(loss_x.item() + loss_z.item())
        return reward
        
    except Exception as e:
        log.error(f"Reward comp error: {e}")
        return -100.0  # Same large penalty on error

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
    llm = LLM(model=MODEL_NAME, enable_lora=True, max_lora_rank=128, gpu_memory_utilization=0.9) 
    
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
    
    # Paper uses temperature 1.0 for E-step sampling
    params = SamplingParams(n=k, temperature=1.0, top_p=0.95, max_tokens=768, stop=["\nProblem:", "Problem:"])
    
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
             p_adapter_path = current_p_subfolder
         else:
             # Download adapter from HF if not available locally
             print(f"Downloading adapter from {HF_REPO_ID} subfolder {current_p_subfolder}...")
             try:
                 downloaded_path = snapshot_download(repo_id=HF_REPO_ID, allow_patterns=f"{current_p_subfolder}/*", token=HF_TOKEN)
                 p_adapter_path = os.path.join(downloaded_path, current_p_subfolder)
             except Exception as e:
                 print(f"Warning: Could not download {current_p_subfolder}: {e}")
                 raise
         
         p_model.load_adapter(p_adapter_path, adapter_name="p_adapter")
         p_model.set_adapter("p_adapter")
    except Exception as e:
         print(f"Error loading p_theta: {e}")
         raise

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
    """Run a single training step (SFT) for either p_theta or q_phi.
    
    Simplified approach: Just load base model and apply fresh LoRA (no merging).
    This avoids dtype mismatches and PEFT config conflicts.
    """
    # Load base model fresh each time
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH, dtype=DTYPE, load_in_4bit=LOAD_IN_4BIT
    )
    
    # Apply fresh LoRA directly on base model
    print("Applying fresh LoRA adapter on base model...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=False,  # Disable gradient checkpointing
        random_state=3407,
        use_rslora=False,
    )
    
    FastLanguageModel.for_training(model)
    
    ds = Dataset.from_dict({"text": texts})
    
    args = TrainingArguments(
        per_device_train_batch_size=64, # Optimized for H200 NVL (143GB VRAM) - increased from 32
        gradient_accumulation_steps=2,  # Effective batch size = 64 * 2 = 128
        num_train_epochs=1, # Plan requirement
        learning_rate=2e-6, # Paper uses 2e-6 for both E-step and M-step
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        output_dir=output_path,
        optim="adamw_8bit",
        report_to="wandb",
        run_name=run_name,
        dataloader_num_workers=8,  # Faster data loading for better GPU utilization
        gradient_checkpointing=False,  # Explicitly disable gradient checkpointing
    )
    
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds, dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH, packing=True, args=args
    )
    trainer.train()
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    # Upload
    if HF_TOKEN and not args.no_upload:
        iter_name = output_path.split('/')[-1] # iter-N
        hf_subpath = f"{HF_VERSION}/{'p' if 'p_' in run_name else 'q'}/{iter_name}"
        api = HfApi(token=HF_TOKEN)
        api.upload_folder(folder_path=output_path, repo_id=HF_REPO_ID, path_in_repo=hf_subpath, repo_type="model")
        
        # Also upload to latest
        hf_latest_path = f"{HF_VERSION}/{'p' if 'p_' in run_name else 'q'}/latest"
        api.upload_folder(folder_path=output_path, repo_id=HF_REPO_ID, path_in_repo=hf_latest_path, repo_type="model")
    elif args.no_upload:
        print(f"  Skipping HuggingFace upload (--no-upload flag): {output_path}")
    
    del model, trainer
    torch.cuda.empty_cache()

def run_e_step_update(new_triples, current_q_subfolder, iteration):
    """Train q_phi on the best selected rationales (SFT)."""
    print("E-Step: Updating q_phi...")
    # Prepare q_phi data: "Concepts: ... Problem: ... Rationale: {best_z}"
    # q_phi learns to produce the SELECTED best rationale given (c, x)
    q_texts = [f"Concepts: {t['concepts']}\nProblem: {t['problem']}\nRationale: {t['rationale']}" for t in new_triples]
    
    next_q_path = f"./models/{HF_VERSION}/q/iter-{iteration}"
    
    # Train SFT to update q_phi
    run_training_step(q_texts, current_q_subfolder, next_q_path, f"q_iter{iteration}")
    return next_q_path

def generate_m_step_data(triples, updated_q_subfolder):
    """Generate deterministic rationales for M-step using updated q_phi."""
    print("M-Step Prep: Generating deterministic rationales...")
    
    # Use the same generation logic but with temp=0
    # We need to load the just-updated q_phi
    
    # Prepare prompts
    vllm_prompts = [f"Concepts: {t['concepts']}\nProblem: {t['problem']}\nRationale:" for t in triples]
    
    llm = LLM(model=MODEL_NAME, enable_lora=True, max_lora_rank=128, gpu_memory_utilization=0.9)
    
    # Check if updated adapter exists locally (it should, we just trained it)
    # If not (e.g. remote only), download it
    if not os.path.exists(updated_q_subfolder):
         print(f"Downloading adapter from {HF_REPO_ID} subfolder {updated_q_subfolder}...")
         try:
             q_adapter_path = snapshot_download(repo_id=HF_REPO_ID, allow_patterns=f"{updated_q_subfolder}/*")
             q_adapter_path = os.path.join(q_adapter_path, updated_q_subfolder)
         except Exception as e:
             # Fallback to string path if it's actually a HF path
             q_adapter_path = updated_q_subfolder
    else:
         q_adapter_path = updated_q_subfolder
         
    lora_req = LoRARequest("q_adapter_new", 1, q_adapter_path)
    
    # Deterministic generation
    params = SamplingParams(n=1, temperature=0.0, max_tokens=768, stop=["\nProblem:", "Problem:"])
    
    outputs = llm.generate(vllm_prompts, params, lora_request=lora_req)
    
    m_step_triples = []
    for i, output in enumerate(outputs):
        z_det = output.outputs[0].text.strip()
        m_step_triples.append({
            "concepts": triples[i]['concepts'],
            "rationale": z_det,
            "problem": triples[i]['problem']
        })
        
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    
    return m_step_triples

def run_m_step_update(m_step_triples, current_p_subfolder, iteration):
    """Train p_theta on deterministic rationales."""
    print("M-Step: Updating p_theta...")
    
    # Prepare p_theta data: "Concepts: ... Rationale: {z_det} Problem: ..."
    # p_theta learns to generate the problem given (c, z_det)
    p_texts = [f"Concepts: {t['concepts']}\nRationale: {t['rationale']}\nProblem: {t['problem']}" for t in m_step_triples]
    
    next_p_path = f"./models/{HF_VERSION}/p/iter-{iteration}"
    
    run_training_step(p_texts, current_p_subfolder, next_p_path, f"p_iter{iteration}")
    return next_p_path

def find_latest_iteration():
    """Find the latest iteration number from HuggingFace repository."""
    if not HF_TOKEN:
        log.warning("HF_TOKEN missing — cannot check HuggingFace for latest iteration")
        return 0
    
    try:
        api = HfApi(token=HF_TOKEN)
        files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type="model", token=HF_TOKEN)
        
        iterations = []
        for file_path in files:
            if f"{HF_VERSION}/p/iter-" in file_path:
                try:
                    parts = file_path.split(f"{HF_VERSION}/p/iter-")
                    if len(parts) > 1:
                        iter_str = parts[1].split("/")[0]
                        iter_num = int(iter_str)
                        iterations.append(iter_num)
                except (ValueError, IndexError):
                    continue
        
        if iterations:
            return max(iterations)
        return 0
    except Exception as e:
        log.warning(f"Failed to check HuggingFace for latest iteration: {e}")
        return 0

# --- Main EM Loop ---
def main():
    # Load Triples
    triples = load_initial_triples()

    # Check for latest iteration to resume
    latest_iter = find_latest_iteration()
    start_iter = latest_iter + 1
    
    if latest_iter > 0:
        print(f"Resuming from iteration {start_iter}")
        # If we finished iter N, we want to start iter N+1
        # The starting models for iter N+1 are the outputs of iter N
        current_p_subfolder = f"{HF_VERSION}/p/iter-{latest_iter}"
        current_q_subfolder = f"{HF_VERSION}/q/iter-{latest_iter}"
    else:
        print("Starting from scratch (Iteration 1)")
        current_p_subfolder = HF_COLD_START_P_SUBFOLDER
        current_q_subfolder = HF_COLD_START_Q_SUBFOLDER
    
    for iteration in range(start_iter, EM_ITERS + 1):
        print(f"\n=== EM Iteration {iteration} ===")
        k = get_k_samples(iteration)
        print(f"Sampling k={k}")
        
        # 1. E-Step: Generate Rationales using current q_phi
        candidates = run_e_step_generation(triples, k, current_q_subfolder)
        
        # 2. E-Step: Reward & Selection (find best z*)
        best_triples = run_e_step_selection(triples, candidates, current_p_subfolder)
        
        # 3. E-Step Update: Train q_phi on best_triples (SFT)
        # This produces q_phi^{new}
        current_q_subfolder = run_e_step_update(best_triples, current_q_subfolder, iteration)
        
        # 4. M-Step Prep: Generate deterministic rationales using q_phi^{new}
        m_step_triples = generate_m_step_data(triples, current_q_subfolder)
        
        # 5. M-Step Update: Train p_theta on m_step_triples
        # This produces p_theta^{new}
        current_p_subfolder = run_m_step_update(m_step_triples, current_p_subfolder, iteration)
        
        # Cleanup old iterations
        cleanup_old_iterations(f"./models/{HF_VERSION}/p", keep_count=3)
        cleanup_old_iterations(f"./models/{HF_VERSION}/q", keep_count=3)
        
    print("EM Loop Complete!")

if __name__ == "__main__":
    main()
