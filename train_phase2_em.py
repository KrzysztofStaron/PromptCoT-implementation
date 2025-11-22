# train_phase2_em.py
# Phase 2: EM Loop with Structure Enforcement (14-16 hrs)
# Updates:
# - Sampling Schedule: k=3 (iter 1-2), k=6 (iter 3-4), k=10 (iter 5-6)
# - Reward: -loss + structure_penalty (missing fields/tags -> -10.0)
# - Generation: Force prefix "\nRationale:" for q_phi

import os
# Enable logits for Unsloth inference (needed for reward computation)
# MUST be set before importing Unsloth
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

from unsloth import FastLanguageModel
import json
import torch
import gc
import re
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
BASE_NUM_TRIPLETS = 100  # Base number of triples (used when k=3)

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

def get_num_triples_for_iteration(iteration):
    """Calculate number of triples for an iteration based on k.
    
    Decreases NUM_TRIPLETS as k increases to keep total computation roughly constant.
    Formula: num_triples = BASE_NUM_TRIPLETS * (3 / k)
    This keeps total candidates roughly constant: triples × k ≈ constant
    """
    k = get_k_samples(iteration)
    # Scale inversely with k: k=3 → full, k=6 → half, k=10 → 30% of base
    num_triples = int(BASE_NUM_TRIPLETS * (3 / k))
    return num_triples

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

def compute_rewards_batched(model, tokenizer, batch_data, device, batch_size=128):
    """Compute rewards for a batch of (concepts, problems, rationales) in parallel.
    
    Args:
        model: The model to use for scoring
        tokenizer: Tokenizer
        batch_data: List of tuples (c, x, z) where c=concepts, x=problem, z=rationale
        device: Device to run on
        batch_size: Batch size for processing
    
    Returns:
        List of rewards (same length as batch_data)
    """
    if not batch_data:
        return []
    
    # Separate valid and invalid (bad structure) items
    valid_items = []
    valid_indices = []
    
    for i, (c, x, z) in enumerate(batch_data):
        if check_structure_and_tags(z):
            valid_items.append((c, x, z))
            valid_indices.append(i)
    
    # Initialize rewards with -100.0 for invalid items
    rewards = [-100.0] * len(batch_data)
    
    if not valid_items:
        return rewards
    
    # Process in batches - each batch in its own try-except to handle OOM gracefully
    for batch_start in range(0, len(valid_items), batch_size):
        batch_end = min(batch_start + batch_size, len(valid_items))
        batch_valid = valid_items[batch_start:batch_end]
        batch_indices = valid_indices[batch_start:batch_end]
        
        try:
            # Prepare batched inputs for loss_x: "Concepts: {c}\nRationale: {z}\nProblem: {x}"
            texts_x = [f"Concepts: {c}\nRationale: {z}\nProblem: {x}" for c, x, z in batch_valid]
            inputs_x = tokenizer(texts_x, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH)
            inputs_x = {k: v.to(device) for k, v in inputs_x.items()}
            
            # Set pad_token_id if not already set
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            
            with torch.no_grad():
                outputs_x = model(**inputs_x, labels=inputs_x["input_ids"])
                # Get per-sample losses
                logits_x = outputs_x.logits
                labels_x = inputs_x["input_ids"]
                shift_logits_x = logits_x[..., :-1, :].contiguous()
                shift_labels_x = labels_x[..., 1:].contiguous()
                
                # Compute per-sample loss
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=pad_token_id)
                flat_shift_logits_x = shift_logits_x.view(-1, shift_logits_x.size(-1))
                flat_shift_labels_x = shift_labels_x.view(-1)
                flat_losses_x = loss_fct(flat_shift_logits_x, flat_shift_labels_x)
                
                # Reshape and compute mean per sequence (ignoring padding)
                losses_x = flat_losses_x.view(shift_labels_x.shape)
                mask_x = (shift_labels_x != pad_token_id).float()
                per_sample_loss_x = (losses_x * mask_x).sum(dim=1) / mask_x.sum(dim=1).clamp(min=1)
            
            # Move to CPU and delete immediately to free GPU memory
            per_sample_loss_x_cpu = per_sample_loss_x.cpu().clone()
            del inputs_x, outputs_x, logits_x, labels_x, shift_logits_x, shift_labels_x, flat_shift_logits_x, flat_shift_labels_x, losses_x, mask_x, per_sample_loss_x
            torch.cuda.empty_cache()
            
            # Prepare batched inputs for loss_z: "Concepts: {c}\nRationale: {z}"
            texts_z = [f"Concepts: {c}\nRationale: {z}" for c, x, z in batch_valid]
            inputs_z = tokenizer(texts_z, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH)
            inputs_z = {k: v.to(device) for k, v in inputs_z.items()}
            
            with torch.no_grad():
                outputs_z = model(**inputs_z, labels=inputs_z["input_ids"])
                logits_z = outputs_z.logits
                labels_z = inputs_z["input_ids"]
                shift_logits_z = logits_z[..., :-1, :].contiguous()
                shift_labels_z = labels_z[..., 1:].contiguous()
                
                # Compute per-sample loss
                flat_shift_logits_z = shift_logits_z.view(-1, shift_logits_z.size(-1))
                flat_shift_labels_z = shift_labels_z.view(-1)
                flat_losses_z = loss_fct(flat_shift_logits_z, flat_shift_labels_z)
                
                # Reshape and compute mean per sequence
                losses_z = flat_losses_z.view(shift_labels_z.shape)
                mask_z = (shift_labels_z != pad_token_id).float()
                per_sample_loss_z = (losses_z * mask_z).sum(dim=1) / mask_z.sum(dim=1).clamp(min=1)
            
            # Move to CPU and delete immediately
            per_sample_loss_z_cpu = per_sample_loss_z.cpu().clone()
            del inputs_z, outputs_z, logits_z, labels_z, shift_logits_z, shift_labels_z, flat_shift_logits_z, flat_shift_labels_z, losses_z, mask_z, per_sample_loss_z
            torch.cuda.empty_cache()
            
            # Verify we have valid tensors before computing rewards
            if not isinstance(per_sample_loss_x_cpu, torch.Tensor) or not isinstance(per_sample_loss_z_cpu, torch.Tensor):
                raise ValueError(f"Invalid tensor types: loss_x={type(per_sample_loss_x_cpu)}, loss_z={type(per_sample_loss_z_cpu)}")
            
            # Compute rewards: -(loss_x + loss_z) - already on CPU
            # Negate tensor first, then convert to list (fixes operator precedence issue)
            combined_loss = per_sample_loss_x_cpu + per_sample_loss_z_cpu
            batch_rewards = (-combined_loss).tolist()
            del per_sample_loss_x_cpu, per_sample_loss_z_cpu
            
            # Assign rewards to valid items
            for idx, reward in zip(batch_indices, batch_rewards):
                rewards[idx] = reward
                
        except torch.cuda.OutOfMemoryError as e:
            log.error(f"OOM in reward batch {batch_start//batch_size + 1}: {e}")
            # Clear cache and assign penalty to this batch
            torch.cuda.empty_cache()
            for idx in batch_indices:
                rewards[idx] = -100.0
        except Exception as e:
            log.error(f"Error in reward batch {batch_start//batch_size + 1}: {e}")
            # Assign penalty to this batch
            for idx in batch_indices:
                rewards[idx] = -100.0
    
    return rewards

# --- Data Loader ---
def load_initial_triples(num_triples=None):
    """Load triples from dataset.
    
    Args:
        num_triples: Number of triples to load. If None, uses BASE_NUM_TRIPLETS.
    """
    if num_triples is None:
        num_triples = BASE_NUM_TRIPLETS
    
    ds = load_dataset("xl-zhao/PromptCoT-Problem-Generation-Dataset", split=f"train[:{num_triples}]")
    
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
    return triples[:num_triples] # Limit to num_triples

# --- Helper Functions ---

def run_e_step_generation(triples, k, current_q_subfolder):
    """Run E-Step: Generate Rationales using q_phi (vLLM for fast inference)."""
    print("E-Step: Generating Rationales...")
    
    # Prepare prompts for vLLM: "Concepts: {c}\nProblem: {x}\nRationale:"
    vllm_prompts = [f"Concepts: {t['concepts']}\nProblem: {t['problem']}\nRationale:" for t in triples]
    
    print(f"  Initializing vLLM engine for {len(vllm_prompts)} prompts...")
    # Generate with vLLM (restart each iteration to pick up new LoRA adapters)
    llm = LLM(
        model=MODEL_NAME,
        enable_lora=True,
        max_lora_rank=128,
        gpu_memory_utilization=0.85,  # Reduced from 0.92 to leave more buffer (prevents OOM)
        max_num_batched_tokens=32768,  # Reduced from 49152 to prevent OOM
        max_num_seqs=512,  # Reduced from 1024 to prevent OOM
        enable_chunked_prefill=True,  # Better for large batches
        block_size=16,  # Memory efficiency optimization
    )
    print("  vLLM engine initialized")
    
    # Download adapter from HF if not available locally
    adapter_path = None
    if os.path.exists(current_q_subfolder):
        adapter_path = current_q_subfolder
        print(f"  Adapter found locally: {adapter_path}")
    else:
        print(f"  Downloading adapter from {HF_REPO_ID} subfolder {current_q_subfolder}...")
        try:
            downloaded_path = snapshot_download(repo_id=HF_REPO_ID, allow_patterns=f"{current_q_subfolder}/*", token=HF_TOKEN)
            adapter_path = os.path.join(downloaded_path, current_q_subfolder)
            print(f"  Adapter downloaded successfully")
        except Exception as e:
            print(f"  Warning: Could not download {current_q_subfolder}: {e}")
            adapter_path = current_q_subfolder
    
    lora_req = LoRARequest("q_adapter", 1, adapter_path)
    
    # Paper uses temperature 1.0 for E-step sampling
    print(f"  Generating {k} samples per prompt for {len(vllm_prompts)} prompts...")
    params = SamplingParams(n=k, temperature=1.0, top_p=0.95, max_tokens=768, stop=["\nProblem:", "Problem:"])
    
    outputs = llm.generate(vllm_prompts, params, lora_request=lora_req)
    
    # Collect candidates: triples[i] -> [z1, z2, ..., zk]
    candidates = []
    for output in outputs:
        z_list = [o.text.strip() for o in output.outputs]
        candidates.append(z_list)
    
    print(f"  Generation complete: {len(candidates)} triples, {sum(len(c) for c in candidates)} total candidates")
    
    # Clean up vLLM to free VRAM for M-step training
    print("  Cleaning up vLLM engine...")
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print("  Cleanup complete")
    
    return candidates

def run_e_step_selection(triples, candidates, current_p_subfolder):
    """Run E-Step: Select Best Rationales using p_theta."""
    print("E-Step: Selecting Best Rationales...")
    print(f"  Computing rewards for {len(triples)} triples with {sum(len(c) for c in candidates)} total candidates")
    
    # Load p_theta for scoring
    # Use Unsloth for efficient inference scoring
    print("  Loading p_theta model...")
    p_model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        token=HF_TOKEN
    )
    print("  p_theta model loaded")
    
    # Load adapter
    print(f"  Loading p_theta adapter from {HF_REPO_ID} subfolder {current_p_subfolder}")
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
    
    # Flatten all candidates for batched processing
    # Structure: (triple_idx, candidate_idx, concepts, problem, rationale)
    flat_candidates = []
    triple_indices = []  # Track which triple each candidate belongs to
    
    for i, t in enumerate(triples):
        c = t['concepts']
        x = t['problem']
        z_list = candidates[i]
        for z in z_list:
            flat_candidates.append((c, x, z))
            triple_indices.append(i)
    
    total_candidates = len(flat_candidates)
    print(f"  Computing rewards for {total_candidates} candidates in batches...")
    
    # Process in batches
    BATCH_SIZE_REWARD = 128  # Reduced from 256 to prevent OOM with seq_len=8192
    total_batches = (total_candidates + BATCH_SIZE_REWARD - 1) // BATCH_SIZE_REWARD
    all_rewards = []
    
    for batch_start in range(0, total_candidates, BATCH_SIZE_REWARD):
        batch_end = min(batch_start + BATCH_SIZE_REWARD, total_candidates)
        batch_data = flat_candidates[batch_start:batch_end]
        batch_num = batch_start // BATCH_SIZE_REWARD + 1
        
        # Progress logging
        print(f"    Processing reward batch {batch_num}/{total_batches} ({batch_end}/{total_candidates} candidates, {100 * batch_end / total_candidates:.1f}%)")
        
        batch_rewards = compute_rewards_batched(p_model, tokenizer, batch_data, p_model.device, BATCH_SIZE_REWARD)
        all_rewards.extend(batch_rewards)
        print(f"    Completed batch {batch_num}/{total_batches}")
    
    # Reconstruct scores per triple and select best
    new_triples = []
    reward_idx = 0
    
    for i, t in enumerate(triples):
        k = len(candidates[i])
        scores = all_rewards[reward_idx:reward_idx + k]
        reward_idx += k
        
        # Select best
        best_idx = scores.index(max(scores))
        best_z = candidates[i][best_idx]
        
        new_triples.append({
            "concepts": t['concepts'],
            "rationale": best_z,
            "problem": t['problem']
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
    print(f"  Training step: {run_name}")
    print(f"  Training on {len(texts)} examples")
    
    # Load base model fresh each time
    print("  Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH, dtype=DTYPE, load_in_4bit=LOAD_IN_4BIT
    )
    print("  Base model loaded")
    
    # Apply fresh LoRA directly on base model
    print("  Applying fresh LoRA adapter on base model...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Enable gradient checkpointing to save memory
        random_state=3407,
        use_rslora=False,
    )
    
    FastLanguageModel.for_training(model)
    print("  Model prepared for training")
    
    print("  Preparing dataset...")
    ds = Dataset.from_dict({"text": texts})
    print(f"  Dataset prepared: {len(ds)} examples")
    
    print("  Starting training...")
    training_args = TrainingArguments(
        per_device_train_batch_size=90, # Reduced from 192 to prevent OOM with seq_len=8192
        gradient_accumulation_steps=2,  # Effective batch size = 64 * 6 = 384 (same effective batch)
        num_train_epochs=1, # Plan requirement
        learning_rate=2e-6, # Paper uses 2e-6 for both E-step and M-step
        fp16=False,  # Disable fp16 - model is in bfloat16
        bf16=True,  # Use bf16 to match model dtype
        output_dir=output_path,
        optim="adamw_8bit",
        report_to="wandb",
        run_name=run_name,
        dataloader_num_workers=32,  # Maximize CPU utilization for data loading on H200
        gradient_checkpointing=True,  # Enable to save memory (trades compute for memory)
    )
    
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds, dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH, packing=True, args=training_args
    )
    trainer.train()
    print("  Training complete")
    
    print(f"  Saving model to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("  Model saved")
    
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
    """Generate deterministic rationales for M-step using updated q_phi (Unsloth)."""
    print("M-Step Prep: Generating deterministic rationales...")
    
    # Load model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        token=HF_TOKEN
    )
    
    # Load adapter
    adapter_path = None
    if os.path.exists(updated_q_subfolder):
        adapter_path = updated_q_subfolder
    else:
        print(f"Downloading adapter from {HF_REPO_ID} subfolder {updated_q_subfolder}...")
        try:
            downloaded_path = snapshot_download(repo_id=HF_REPO_ID, allow_patterns=f"{updated_q_subfolder}/*", token=HF_TOKEN)
            adapter_path = os.path.join(downloaded_path, updated_q_subfolder)
        except Exception as e:
            print(f"Warning: Could not download {updated_q_subfolder}: {e}")
            adapter_path = updated_q_subfolder
    
    if adapter_path and os.path.exists(adapter_path):
        model.load_adapter(adapter_path)
    
    FastLanguageModel.for_inference(model)
    
    # Generate deterministic rationales (temperature=0) in batches
    print(f"  Generating deterministic rationales for {len(triples)} triples...")
    prompts = [f"Concepts: {t['concepts']}\nProblem: {t['problem']}\nRationale:" for t in triples]
    
    m_step_triples = []
    batch_size_gen = 32  # Batch size for M-step generation
    
    for batch_start in range(0, len(prompts), batch_size_gen):
        batch_end = min(batch_start + batch_size_gen, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        batch_triples = triples[batch_start:batch_end]
        
        # Tokenize batch
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate for batch
        outputs = model.generate(
            **inputs,
            max_new_tokens=768,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        # Process outputs
        for i, output in enumerate(outputs):
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            rationale = generated_text.split("Rationale:")[-1].strip()
            # Stop at Problem: if it appears
            if "\nProblem:" in rationale:
                rationale = rationale.split("\nProblem:")[0].strip()
            
            m_step_triples.append({
                "concepts": batch_triples[i]['concepts'],
                "rationale": rationale,
                "problem": batch_triples[i]['problem']
            })
        
        del inputs, outputs
        torch.cuda.empty_cache()
    
    print(f"  Generated {len(m_step_triples)} deterministic rationales")
    
    # Cleanup
    del model
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
    # Load triples will be done per iteration with adjusted count based on k
    # For initial load (used in testm mode), use base number
    initial_triples = load_initial_triples(BASE_NUM_TRIPLETS)

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
        
        # Calculate k and adjust number of triples for this iteration
        k = get_k_samples(iteration)
        num_triples = get_num_triples_for_iteration(iteration)
        print(f"Sampling k={k}, using {num_triples} triples (scaled from base {BASE_NUM_TRIPLETS})")
        
        # Load triples for this iteration (with adjusted count)
        triples = load_initial_triples(num_triples)
        
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
