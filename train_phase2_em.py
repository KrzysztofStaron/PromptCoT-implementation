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
from transformers import AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, create_repo, snapshot_download
from dotenv import load_dotenv
import wandb
from trl import SFTTrainer

# Set vLLM logging level before importing vLLM
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
os.environ["VLLM_USE_MODELSCOPE"] = "False"
# Suppress Gloo/NCCL verbose output
os.environ["GLOO_LOG_LEVEL"] = "WARN"
os.environ["NCCL_DEBUG"] = "WARN"

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from hf_config import HF_REPO_ID, HF_VERSION

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Configure logging levels for verbose libraries
import transformers
import datasets
import huggingface_hub

transformers.logging.set_verbosity_error()
datasets.logging.set_verbosity_error()
huggingface_hub.logging.set_verbosity_warning()

# Set main logger to WARNING to reduce noise
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)

# --- Config ---
MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-7B"
MAX_SEQ_LENGTH = 8192
DTYPE = None
LOAD_IN_4BIT = True

EM_ITERS = 6
NUM_TRIPLETS = 4000

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
    """Compute reward as log p_θ(x|z,c) + log p_θ(z|c).
    
    Reward = -(loss_x + loss_z) where:
    - loss_x = NLL of problem given concepts and rationale
    - loss_z = NLL of rationale given concepts
    
    Bad structure gets penalty of -100.0 to ensure it's always worse than valid generation.
    """
    if not check_structure_and_tags(z):
        return -100.0  # Large penalty to ensure bad structure is always worse than valid generation
        
    try:
        # Compute log p_θ(x | z, c)
        input_x = tokenizer(
            f"Concepts: {c}\nRationale: {z}\nProblem: {x}",
            return_tensors="pt"
        )
        input_x = {k: v.to(model.device) for k, v in input_x.items()}
        
        with torch.no_grad():
            loss_x = model(**input_x, labels=input_x["input_ids"]).loss
        del input_x  # Free memory immediately
        
        # Compute log p_θ(z | c)
        input_z = tokenizer(
            f"Concepts: {c}\nRationale: {z}",
            return_tensors="pt"
        )
        input_z = {k: v.to(model.device) for k, v in input_z.items()}
        
        with torch.no_grad():
            loss_z = model(**input_z, labels=input_z["input_ids"]).loss
        del input_z  # Free memory immediately
        
        # Reward = log p_θ(x|z,c) + log p_θ(z|c) = -(loss_x + loss_z)
        reward = -(loss_x.item() + loss_z.item())
        return reward
        
    except Exception as e:
        log.error(f"Reward comp error: {e}")
        return -100.0  # Same large penalty on error

def compute_rewards_batched(model, tokenizer, batch_data, device):
    """Compute rewards for a batch of (concepts, problems, rationales) in parallel.
    
    Args:
        model: The model to use for scoring
        tokenizer: Tokenizer
        batch_data: List of tuples (c, x, z) where c=concepts, x=problem, z=rationale
        device: Device to run on
    
    Returns:
        List of rewards (same length as batch_data)
    """
    if not batch_data:
        return []
    
    # Separate valid and invalid (bad structure) items
    valid_items = []
    valid_indices = []
    invalid_indices = []
    
    for i, (c, x, z) in enumerate(batch_data):
        if check_structure_and_tags(z):
            valid_items.append((c, x, z))
            valid_indices.append(i)
        else:
            invalid_indices.append(i)
    
    # Initialize rewards with -100.0 for invalid items
    rewards = [-100.0] * len(batch_data)
    
    if not valid_items:
        return rewards
    
    try:
        # Prepare batched inputs for loss_x: "Concepts: {c}\nRationale: {z}\nProblem: {x}"
        texts_x = [f"Concepts: {c}\nRationale: {z}\nProblem: {x}" for c, x, z in valid_items]
        inputs_x = tokenizer(texts_x, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH)
        inputs_x = {k: v.to(device) for k, v in inputs_x.items()}
        
        # Set pad_token_id if not already set
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        
        with torch.no_grad():
            outputs_x = model(**inputs_x, labels=inputs_x["input_ids"])
            # Get per-sample losses (reduction='none' then mean per sequence)
            # The model returns average loss, but we need per-sample
            # We'll compute it manually from logits
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
            # Mask out padding tokens
            mask_x = (shift_labels_x != pad_token_id).float()
            per_sample_loss_x = (losses_x * mask_x).sum(dim=1) / mask_x.sum(dim=1).clamp(min=1)
        
        del inputs_x, outputs_x, logits_x, labels_x
        
        # Prepare batched inputs for loss_z: "Concepts: {c}\nRationale: {z}"
        texts_z = [f"Concepts: {c}\nRationale: {z}" for c, x, z in valid_items]
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
        
        del inputs_z, outputs_z, logits_z, labels_z
        
        # Compute rewards: -(loss_x + loss_z)
        valid_rewards = -(per_sample_loss_x.cpu() + per_sample_loss_z.cpu()).tolist()
        
        # Assign rewards to valid items
        for idx, reward in zip(valid_indices, valid_rewards):
            rewards[idx] = reward
            
    except Exception as e:
        log.error(f"Batched reward comp error: {e}")
        # On error, all valid items get -100.0 penalty
        for idx in valid_indices:
            rewards[idx] = -100.0
    
    return rewards

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
    # Optimized for H200 NVL: higher memory utilization and better batching
    llm = LLM(
        model=MODEL_NAME,
        enable_lora=True,
        max_lora_rank=128,
        gpu_memory_utilization=0.95,  # Increased from 0.9 for better GPU utilization
        max_num_batched_tokens=32768,  # Allow larger batches for better throughput
        max_num_seqs=2048,  # More concurrent sequences
        enable_chunked_prefill=True,  # Better for large batches
        block_size=16,  # Memory efficiency optimization
    ) 
    
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
    BATCH_SIZE_REWARD = 64  # Batch size for reward computation
    all_rewards = []
    
    for batch_start in range(0, total_candidates, BATCH_SIZE_REWARD):
        batch_end = min(batch_start + BATCH_SIZE_REWARD, total_candidates)
        batch_data = flat_candidates[batch_start:batch_end]
        
        # Progress logging
        if (batch_start // BATCH_SIZE_REWARD + 1) % 10 == 0 or batch_end == total_candidates:
            print(f"    Processing batch {batch_start // BATCH_SIZE_REWARD + 1}/{(total_candidates + BATCH_SIZE_REWARD - 1) // BATCH_SIZE_REWARD} "
                  f"({batch_end}/{total_candidates} candidates, {100 * batch_end / total_candidates:.1f}%)")
        
        batch_rewards = compute_rewards_batched(p_model, tokenizer, batch_data, p_model.device)
        all_rewards.extend(batch_rewards)
    
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
    """Run a single training step (SFT) for either p_theta or q_phi."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH, dtype=DTYPE, load_in_4bit=LOAD_IN_4BIT
    )
    
    loaded_adapter = False
    try:
        # Try loading existing adapter
        if os.path.exists(base_adapter_subfolder):
            adapter_path = base_adapter_subfolder
            model.load_adapter(adapter_path)
            loaded_adapter = True
        else:
            # Download adapter from HF if not available locally
            print(f"Downloading adapter from {HF_REPO_ID} subfolder {base_adapter_subfolder}...")
            try:
                downloaded_path = snapshot_download(repo_id=HF_REPO_ID, allow_patterns=f"{base_adapter_subfolder}/*", token=HF_TOKEN)
                adapter_path = os.path.join(downloaded_path, base_adapter_subfolder)
                model.load_adapter(adapter_path)
                loaded_adapter = True
            except Exception as e:
                print(f"Adapter download/load failed: {e}")
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
    
    # Optimized for H200 NVL: higher memory utilization and better batching
    llm = LLM(
        model=MODEL_NAME,
        enable_lora=True,
        max_lora_rank=128,
        gpu_memory_utilization=0.95,  # Increased from 0.9 for better GPU utilization
        max_num_batched_tokens=32768,  # Allow larger batches for better throughput
        max_num_seqs=2048,  # More concurrent sequences
        enable_chunked_prefill=True,  # Better for large batches
        block_size=16,  # Memory efficiency optimization
    )
    
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
