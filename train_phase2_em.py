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

# Enable memory fragmentation fix to prevent OOM
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- Config ---
MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-7B"
MAX_SEQ_LENGTH = 8192
DTYPE = None
LOAD_IN_4BIT = True

EM_ITERS = 2
BASE_NUM_TRIPLETS = 1000  # Base number of triples (used when k=3)

# Command-line arguments
parser = argparse.ArgumentParser(description="Train PromptCoT Phase 2 EM Loop")
parser.add_argument("--no-upload", action="store_true", help="Disable uploading to HuggingFace")
parser.add_argument("--k", type=int, default=5, help="Number of samples to generate per prompt (default: 10)")
args = parser.parse_args()


# Paths
# We load models from HuggingFace directly
HF_COLD_START_P_SUBFOLDER = f"{HF_VERSION}/p/cold-start"
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
    return args.k  # Use constant k from command line argument

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
            f"[CONCEPTS]\n{c}\n[/CONCEPTS]\n\n[RATIONALE]\n{z}\n[/RATIONALE]\n\n[PROBLEM]\n{x}\n[/PROBLEM]",
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
            # Prepare batched inputs for loss_x: "[CONCEPTS]\n{c}\n[/CONCEPTS]\n\n[RATIONALE]\n{z}\n[/RATIONALE]\n\n[PROBLEM]\n{x}\n[/PROBLEM]"
            texts_x = [f"[CONCEPTS]\n{c}\n[/CONCEPTS]\n\n[RATIONALE]\n{z}\n[/RATIONALE]\n\n[PROBLEM]\n{x}\n[/PROBLEM]" for c, x, z in batch_valid]
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

def get_gpu_memory_info():
    """Get current GPU memory usage information."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'free_gb': total - reserved
        }
    return None

def log_gpu_memory(stage=""):
    """Log current GPU memory usage."""
    mem_info = get_gpu_memory_info()
    if mem_info:
        print(f"  GPU Memory {stage}: {mem_info['allocated_gb']:.2f} GB allocated, "
              f"{mem_info['reserved_gb']:.2f} GB reserved, "
              f"{mem_info['free_gb']:.2f} GB free (of {mem_info['total_gb']:.2f} GB total)")

def cleanup_gpu_memory():
    """Aggressively clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        # Small delay to allow system to release memory
        import time
        time.sleep(0.5)

def run_e_step_generation(triples, k, current_q_subfolder):
    """Run E-Step: Generate Rationales using q_phi (vLLM for fast inference)."""
    print("E-Step: Generating Rationales...")
    
    # Prepare prompts for vLLM: "[CONCEPTS]\n{c}\n[/CONCEPTS]\n\n[PROBLEM]\n{x}\n[/PROBLEM]\n\n[RATIONALE]\n"
    vllm_prompts = [f"[CONCEPTS]\n{t['concepts']}\n[/CONCEPTS]\n\n[PROBLEM]\n{t['problem']}\n[/PROBLEM]\n\n[RATIONALE]\n" for t in triples]

    log_gpu_memory("before vLLM initialization")
    print(f"  Initializing vLLM engine for {len(vllm_prompts)} prompts...")
    # Generate with vLLM (restart each iteration to pick up new LoRA adapters)
    llm = LLM(
        model=MODEL_NAME,
        enable_lora=True,
        max_lora_rank=128,
        gpu_memory_utilization=0.8,  # Slightly increased for better throughput
        max_num_batched_tokens=24576,  # Balanced setting for performance
        max_num_seqs=256,  # Increased for better throughput
        enable_chunked_prefill=True,  # Better for large batches
        block_size=16,  # Memory efficiency optimization
    )
    print("  vLLM engine initialized")
    log_gpu_memory("after vLLM initialization")
    
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
    
    # Cleanup before creating LoRA request and starting generation
    cleanup_gpu_memory()
    log_gpu_memory("before LoRA adapter loading")
    
    lora_req = LoRARequest("q_adapter", 1, adapter_path)
    
    log_gpu_memory("after LoRA adapter loaded, before generation")
    
    # Paper uses temperature 1.0 for E-step sampling
    print(f"  Generating {k} samples per prompt for {len(vllm_prompts)} prompts...")
    params = SamplingParams(n=k, temperature=1.0, top_p=0.95, max_tokens=8192, stop=["[/RATIONALE]", "\n[PROBLEM]", "\n[CONCEPTS]"])
    
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
    cleanup_gpu_memory()
    log_gpu_memory("after cleanup")
    print("  Cleanup complete")
    
    return candidates

def run_e_step_selection(triples, candidates, current_p_subfolder, iteration):
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
    print("  Base model loaded")

    # Apply LoRA adapter structure (matches training configuration)
    print("  Applying LoRA adapter structure...")
    p_model = FastLanguageModel.get_peft_model(
        p_model,
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
    print(f"  Loading p_theta adapter from {HF_REPO_ID} subfolder {current_p_subfolder}")
    try:
        # Check if local or HF
        if os.path.exists(current_p_subfolder):
            p_adapter_path = current_p_subfolder
            print(f"  Adapter found locally: {p_adapter_path}")
        else:
            # Download adapter from HF if not available locally
            print(f"  Downloading adapter from {HF_REPO_ID} subfolder {current_p_subfolder}...")
            downloaded_path = snapshot_download(repo_id=HF_REPO_ID, allow_patterns=f"{current_p_subfolder}/*", token=HF_TOKEN)
            p_adapter_path = os.path.join(downloaded_path, current_p_subfolder)
            print("  Adapter downloaded successfully")

        p_model.load_adapter(p_adapter_path, adapter_name="p_adapter")
        p_model.set_adapter("p_adapter")
        print("  Adapter loaded successfully")
    except Exception as e:
        print(f"  Error loading p_theta: {e}")
        raise

    FastLanguageModel.for_inference(p_model)
    print("  p_theta model ready for inference")
    
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
    
    # Process in batches - balanced size that worked sometimes with 128
    BATCH_SIZE_REWARD = 96  # Increased for better throughput
    total_batches = (total_candidates + BATCH_SIZE_REWARD - 1) // BATCH_SIZE_REWARD
    all_rewards = []
    
    for batch_start in range(0, total_candidates, BATCH_SIZE_REWARD):
        batch_end = min(batch_start + BATCH_SIZE_REWARD, total_candidates)
        batch_data = flat_candidates[batch_start:batch_end]
        batch_num = batch_start // BATCH_SIZE_REWARD + 1
        
        # Progress logging
        print(f"    Processing reward batch {batch_num}/{total_batches} ({batch_end}/{total_candidates} candidates, {100 * batch_end / total_candidates:.1f}%)")
        
        # Use larger batch size for actual computation (64 sequences at a time)
        batch_rewards = compute_rewards_batched(p_model, tokenizer, batch_data, p_model.device, batch_size=48)
        all_rewards.extend(batch_rewards)
        print(f"    Completed batch {batch_num}/{total_batches}")

    # Log reward statistics to wandb
    valid_rewards = [r for r in all_rewards if r > -100.0]  # Exclude penalty rewards
    if valid_rewards:
        avg_reward = sum(valid_rewards) / len(valid_rewards)
        max_reward = max(valid_rewards)
        min_reward = min(valid_rewards)
        print(f"  Reward statistics: avg={avg_reward:.3f}, max={max_reward:.3f}, min={min_reward:.3f}")

        wandb.log({
            "iteration": iteration,
            "avg_reward": avg_reward,
            "max_reward": max_reward,
            "min_reward": min_reward,
            "num_valid_candidates": len(valid_rewards),
            "total_candidates": len(all_rewards)
        })

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

def run_training_step(texts, base_adapter_subfolder, output_path, iteration):
    """Run a single training step (SFT) for either p_theta or q_phi.

    Loads the previous adapter (or cold-start) and continues training from it.
    Does not create separate wandb runs - EM process manages logging.
    """
    print(f"  Training step: {output_path.split('/')[-1]}")
    print(f"  Training on {len(texts)} examples")
    
    # Load base model fresh each time
    print("  Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH, dtype=DTYPE, load_in_4bit=LOAD_IN_4BIT
    )
    print("  Base model loaded")
    
    # Apply LoRA adapter structure (matches Phase 0 configuration)
    print("  Applying LoRA adapter structure...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,  # Matches Phase 0 for correct adapter loading
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Enable gradient checkpointing to save memory
        random_state=3407,
        use_rslora=False,
    )
    
    # Load the pre-trained adapter (cold-start or previous iteration)
    print(f"  Loading adapter from {base_adapter_subfolder}...")
    try:
        # Check if local or HF
        if os.path.exists(base_adapter_subfolder):
            adapter_path = base_adapter_subfolder
            print(f"  Adapter found locally: {adapter_path}")
        else:
            # Download adapter from HF if not available locally
            print(f"  Downloading adapter from {HF_REPO_ID} subfolder {base_adapter_subfolder}...")
            downloaded_path = snapshot_download(repo_id=HF_REPO_ID, allow_patterns=f"{base_adapter_subfolder}/*", token=HF_TOKEN)
            adapter_path = os.path.join(downloaded_path, base_adapter_subfolder)
            print(f"  Adapter downloaded successfully")
        
        model.load_adapter(adapter_path, adapter_name="base_adapter")
        model.set_adapter("base_adapter")
        print("  Adapter loaded successfully")
    except Exception as e:
        print(f"  Warning: Could not load adapter {base_adapter_subfolder}: {e}")
        print("  Training from scratch (no adapter loaded)")
    
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
        report_to="none",  # Disable wandb reporting - EM run manages logging
        dataloader_num_workers=32,  # Maximize CPU utilization for data loading on H200
        gradient_checkpointing=True,  # Enable to save memory (trades compute for memory)
    )
    
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds, dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH, packing=True, args=training_args
    )
    train_result = trainer.train()
    print("  Training complete")

    # Log training loss to wandb
    final_loss = train_result.training_loss if hasattr(train_result, 'training_loss') else None
    if final_loss is not None:
        wandb.log({
            "iteration": iteration,
            "training_loss": final_loss,
            "model_type": "p" if "/p/" in output_path else "q"
        })

    # Upload to HuggingFace first
    if HF_TOKEN and not args.no_upload:
        print(f"  Uploading to HuggingFace first...")
        iter_name = output_path.split('/')[-1] # iter-N
        model_type = 'p' if '/p/' in output_path else 'q'
        hf_subpath = f"{HF_VERSION}/{model_type}/{iter_name}"
        api = HfApi(token=HF_TOKEN)
        api.upload_folder(folder_path=output_path, repo_id=HF_REPO_ID, path_in_repo=hf_subpath, repo_type="model")

        # Also upload to latest
        hf_latest_path = f"{HF_VERSION}/{model_type}/latest"
        api.upload_folder(folder_path=output_path, repo_id=HF_REPO_ID, path_in_repo=hf_latest_path, repo_type="model")
        print("  Upload to HuggingFace complete")
    elif args.no_upload:
        print(f"  Skipping HuggingFace upload (--no-upload flag): {output_path}")

    print(f"  Saving model locally to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("  Model saved locally")
    
    del model, trainer
    torch.cuda.empty_cache()

def run_e_step_update(new_triples, current_q_subfolder, iteration):
    """Train q_phi on the best selected rationales (SFT)."""
    print("E-Step: Updating q_phi...")
    # Prepare q_phi data: "[CONCEPTS]\n{c}\n[/CONCEPTS]\n\n[PROBLEM]\n{x}\n[/PROBLEM]\n\n[RATIONALE]\n{z}\n[/RATIONALE]"
    # q_phi learns to produce the SELECTED best rationale given (c, x)
    q_texts = [f"[CONCEPTS]\n{t['concepts']}\n[/CONCEPTS]\n\n[PROBLEM]\n{t['problem']}\n[/PROBLEM]\n\n[RATIONALE]\n{t['rationale']}\n[/RATIONALE]" for t in new_triples]

    next_q_path = f"./models/{HF_VERSION}/q/iter-{iteration}"

    # Train SFT to update q_phi
    run_training_step(q_texts, current_q_subfolder, next_q_path, iteration)
    return next_q_path

def generate_m_step_data(triples, updated_q_subfolder):
    """Generate deterministic rationales for M-step using updated q_phi (vLLM for fast inference)."""
    print("M-Step Prep: Generating deterministic rationales...")
    
    # Prepare prompts for vLLM: "[CONCEPTS]\n{c}\n[/CONCEPTS]\n\n[PROBLEM]\n{x}\n[/PROBLEM]\n\n[RATIONALE]\n"
    vllm_prompts = [f"[CONCEPTS]\n{t['concepts']}\n[/CONCEPTS]\n\n[PROBLEM]\n{t['problem']}\n[/PROBLEM]\n\n[RATIONALE]\n" for t in triples]

    log_gpu_memory("before vLLM initialization (M-step)")
    print(f"  Initializing vLLM engine for {len(vllm_prompts)} prompts...")
    # Generate with vLLM (much faster than Unsloth for batched inference)
    llm = LLM(
        model=MODEL_NAME,
        enable_lora=True,
        max_lora_rank=128,
        gpu_memory_utilization=0.8,  # Slightly increased for better throughput
        max_num_batched_tokens=24576,  # Balanced setting for performance
        max_num_seqs=320,  # Increased for better throughput
        enable_chunked_prefill=True,
        block_size=16,
    )
    print("  vLLM engine initialized")
    log_gpu_memory("after vLLM initialization (M-step)")
    
    # Download adapter from HF if not available locally
    adapter_path = None
    if os.path.exists(updated_q_subfolder):
        adapter_path = updated_q_subfolder
        print(f"  Adapter found locally: {adapter_path}")
    else:
        print(f"  Downloading adapter from {HF_REPO_ID} subfolder {updated_q_subfolder}...")
        try:
            downloaded_path = snapshot_download(repo_id=HF_REPO_ID, allow_patterns=f"{updated_q_subfolder}/*", token=HF_TOKEN)
            adapter_path = os.path.join(downloaded_path, updated_q_subfolder)
            print(f"  Adapter downloaded successfully")
        except Exception as e:
            print(f"  Warning: Could not download {updated_q_subfolder}: {e}")
            adapter_path = updated_q_subfolder
    
    # Cleanup before creating LoRA request and starting generation
    cleanup_gpu_memory()
    log_gpu_memory("before LoRA adapter loading (M-step)")
    
    lora_req = LoRARequest("q_adapter", 1, adapter_path)
    
    log_gpu_memory("after LoRA adapter loaded, before generation (M-step)")
    
    # Generate deterministic rationales (temperature=0.0 for deterministic)
    print(f"  Generating deterministic rationales for {len(vllm_prompts)} prompts...")
    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=8192, stop=["[/RATIONALE]", "\n[PROBLEM]", "\n[CONCEPTS]"])
    
    outputs = llm.generate(vllm_prompts, params, lora_request=lora_req)
    
    # Process outputs
    m_step_triples = []
    for i, output in enumerate(outputs):
        # Get the generated text (first output since n=1 by default)
        generated_text = output.outputs[0].text.strip()
        rationale = generated_text
        
        # Stop at Problem: if it appears (shouldn't happen with stop tokens, but just in case)
        if "\nProblem:" in rationale:
            rationale = rationale.split("\nProblem:")[0].strip()
        
        m_step_triples.append({
            "concepts": triples[i]['concepts'],
            "rationale": rationale,
            "problem": triples[i]['problem']
        })
    
    print(f"  Generated {len(m_step_triples)} deterministic rationales")
    
    # Clean up vLLM to free VRAM
    print("  Cleaning up vLLM engine...")
    del llm
    cleanup_gpu_memory()
    log_gpu_memory("after cleanup (M-step)")
    print("  Cleanup complete")
    
    return m_step_triples

def run_m_step_update(m_step_triples, current_p_subfolder, iteration):
    """Train p_theta on deterministic rationales."""
    print("M-Step: Updating p_theta...")

    # Prepare p_theta data: "[CONCEPTS]\n{c}\n[/CONCEPTS]\n\n[RATIONALE]\n{z}\n[/RATIONALE]\n\n[PROBLEM]\n{x}\n[/PROBLEM]"
    # p_theta learns to generate the problem given (c, z_det)
    p_texts = [f"[CONCEPTS]\n{t['concepts']}\n[/CONCEPTS]\n\n[RATIONALE]\n{t['rationale']}\n[/RATIONALE]\n\n[PROBLEM]\n{t['problem']}\n[/PROBLEM]" for t in m_step_triples]

    next_p_path = f"./models/{HF_VERSION}/p/iter-{iteration}"

    # Train SFT to update p_theta
    run_training_step(p_texts, current_p_subfolder, next_p_path, iteration)
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

    # Initialize single wandb run for entire EM process
    print("Initializing EM wandb run...")
    em_run_id = f"em_training_{EM_ITERS}iters"
    wandb.init(
        project="promptcot-em",
        id=em_run_id,
        resume="allow",
        config={
            "model": MODEL_NAME,
            "em_iterations": EM_ITERS,
            "max_seq_length": MAX_SEQ_LENGTH,
            "base_num_triplets": BASE_NUM_TRIPLETS,
        }
    )

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
        
        # Cleanup GPU memory at start of iteration to ensure clean state
        if iteration > start_iter:
            print("  Cleaning up GPU memory at start of iteration...")
            cleanup_gpu_memory()
            log_gpu_memory("at iteration start")
        
        # Calculate k and adjust number of triples for this iteration
        k = get_k_samples(iteration)
        num_triples = get_num_triples_for_iteration(iteration)
        print(f"Sampling k={k}, using {num_triples} triples (scaled from base {BASE_NUM_TRIPLETS})")

        # Log iteration start to wandb
        wandb.log({
            "iteration": iteration,
            "k_samples": k,
            "num_triples": num_triples,
        })

        # Load triples for this iteration (with adjusted count)
        triples = load_initial_triples(num_triples)
        
        # 1. E-Step: Generate Rationales using current q_phi
        candidates = run_e_step_generation(triples, k, current_q_subfolder)
        
        # 2. E-Step: Reward & Selection (find best z*)
        best_triples = run_e_step_selection(triples, candidates, current_p_subfolder, iteration)
        
        # 3. E-Step Update: Train q_phi on best_triples (SFT)
        # This produces q_phi^{new}
        current_q_subfolder = run_e_step_update(best_triples, current_q_subfolder, iteration)
        
        # 4. M-Step Prep: Generate deterministic rationales using q_phi^{new}
        m_step_triples = generate_m_step_data(triples, current_q_subfolder)
        
        # 5. M-Step Update: Train p_theta on m_step_triples
        # This produces p_theta^{new}
        current_p_subfolder = run_m_step_update(m_step_triples, current_p_subfolder, iteration)

        # Log iteration completion to wandb
        wandb.log({
            "iteration_completed": iteration,
        })

        # Cleanup old iterations
        cleanup_old_iterations(f"./models/{HF_VERSION}/p", keep_count=3)
        cleanup_old_iterations(f"./models/{HF_VERSION}/q", keep_count=3)
        
    print("EM Loop Complete!")

if __name__ == "__main__":
    main()
