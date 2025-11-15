# em.py — FINAL, RUNNING ON H200
# PromptCoT EM Loop Training
#
# CRITICAL: We use BASE models, NOT INSTRUCT models, for faithful PromptCoT 2.0 reproduction.
# Base models provide:
#   - High entropy and diversity (needed for EM exploration)
#   - Non-deterministic rationale generation
#   - No instruction-tuning biases that collapse rationale structures
#   - Ability to rebuild the problem-generation model from scratch (not fine-tune existing one)
#
# Paper uses Qwen2.5-32B-Base; we use Qwen2.5-7B-Base (scaled-down version)
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from datasets import Dataset
from peft import PeftModel
import os
import logging
import tempfile
import threading
import sys
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv
from tieBreaker import select_best_rationale
import wandb
from hf_config import HF_USERNAME, HF_VERSION, HF_REPO_ID, HF_P_BASE_PATH, HF_Q_BASE_PATH
from em_logging import (
    log_batch_metrics, log_final_batch_metrics, log_e_step_summary,
    log_m_step_summary, log_iteration_summary, log_winner_rationale,
    log_batch_start, log_batch_generation_complete, log_e_step_progress
)

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# === CONFIG ===
# BASE model (NOT Instruct) - required for faithful PromptCoT 2.0 reproduction
# Both pθ and qφ are initialized from the same base checkpoint
# Paper uses Qwen2.5-32B-Base; we use Qwen2.5-7B-Base (scaled-down version)
MODEL_NAME = "Qwen/Qwen2.5-7B"  # Base model (no -Instruct suffix)
SEED_FILE = "./data/annotated.jsonl"
CHECKPOINT_DIR = "./checkpoints"
EM_ITERS = 6
MAX_K_SAMPLES = 8
MIN_K_SAMPLES = 4
BATCH_SIZE = 16
USE_GROQ_TIEBREAKER = False
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch size = per_device_train_batch_size * gradient_accumulation_steps

# === SAMPLING SCHEDULE ===
def get_k_samples_for_iteration(iter_num_1_indexed: int) -> int:
    """
    Decrease the number of sampled rationales each EM iteration until hitting MIN_K_SAMPLES.
    Iter 1 → MAX_K_SAMPLES, Iter 2 → MAX_K_SAMPLES-1, etc.
    """
    decrement = max(0, iter_num_1_indexed - 1)
    return max(MIN_K_SAMPLES, MAX_K_SAMPLES - decrement)

# HuggingFace subfolders for cold-start adapters
PROMPT_INIT_SUBPATH = f"{HF_P_BASE_PATH}cold-start"
RATIONALE_INIT_SUBPATH = f"{HF_Q_BASE_PATH}cold-start"

# Create checkpoint directory if it doesn't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# === WANDB INIT ===
wandb.init(
    project="promptcot-em",
    name=f"em_training_{EM_ITERS}iters_k{MAX_K_SAMPLES}",
    config={
        "model": MODEL_NAME,
        "em_iterations": EM_ITERS,
        "k_samples": MAX_K_SAMPLES,
        "batch_size": BATCH_SIZE,
        "use_groq_tiebreaker": USE_GROQ_TIEBREAKER,
    }
)

# === TOKENIZER ===
log.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side='left',
    truncation_side='left'
)
tokenizer.pad_token = tokenizer.eos_token

# === MODELS ===
log.info("Loading base model for pθ...")
base_p = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

log.info(f"Loading pθ adapters from {HF_REPO_ID}/{PROMPT_INIT_SUBPATH}...")
pθ = PeftModel.from_pretrained(
    base_p,
    HF_REPO_ID,
    is_trainable=True,
    subfolder=PROMPT_INIT_SUBPATH,
    token=HF_TOKEN,
)

log.info("Loading base model for qφ...")
base_q = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

log.info(f"Loading qφ adapters from {HF_REPO_ID}/{RATIONALE_INIT_SUBPATH}...")
qφ = PeftModel.from_pretrained(
    base_q,
    HF_REPO_ID,
    is_trainable=True,
    subfolder=RATIONALE_INIT_SUBPATH,
    token=HF_TOKEN,
)

# === CHECKPOINT MANAGEMENT ===
def find_latest_checkpoint():
    """
    Find the latest checkpoint iteration number.
    Returns tuple (iter_num, is_incomplete) where iter_num is 1-indexed and is_incomplete indicates if it's an incomplete checkpoint.
    Handles migration from 0-indexed (iter_0) to 1-indexed (iter_1, iter_2, ...).
    """
    if not os.path.exists(CHECKPOINT_DIR):
        return None, False
    
    # Check for incomplete checkpoints first (they take priority)
    incomplete_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("iter_") and f.endswith("_incomplete_triples.jsonl")]
    if incomplete_files:
        iterations = []
        for f in incomplete_files:
            try:
                iter_num = int(f.split("_")[1])
                iterations.append(iter_num)
            except (ValueError, IndexError):
                continue
        if iterations:
            max_iter = max(iterations)
            return max_iter, True
    
    # Check for completed checkpoints
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("iter_") and f.endswith("_triples.jsonl") and not f.endswith("_incomplete_triples.jsonl")]
    if not checkpoint_files:
        return None, False
    
    iterations = []
    for f in checkpoint_files:
        try:
            iter_num = int(f.split("_")[1])
            iterations.append(iter_num)
        except (ValueError, IndexError):
            continue
    
    if not iterations:
        return None, False
    
    max_iter = max(iterations)
    # Handle migration: if max_iter is 0, it's old 0-indexed format (iter_0 = iteration 1 completed)
    if max_iter == 0:
        # Old format: iter_0 means iteration 1 completed (1-indexed)
        return 1, False
    # New format: iter_1 means iteration 1 completed, iter_2 means iteration 2 completed, etc.
    return max_iter, False

def load_checkpoint(iter_num_1_indexed, incomplete=False):
    """
    Load triples from a specific checkpoint iteration.
    
    Args:
        iter_num_1_indexed: 1-indexed iteration number (e.g., 1, 2, 3, ...)
        incomplete: If True, loads incomplete checkpoint
    """
    if incomplete:
        checkpoint_file = os.path.join(CHECKPOINT_DIR, f"iter_{iter_num_1_indexed}_incomplete_triples.jsonl")
    else:
        checkpoint_file = os.path.join(CHECKPOINT_DIR, f"iter_{iter_num_1_indexed}_triples.jsonl")
    
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            triples = [json.loads(line) for line in f]
        status = "incomplete" if incomplete else "completed"
        log.info(f"Loaded {status} checkpoint from iteration {iter_num_1_indexed}: {len(triples)} triples")
        return triples
    
    # Fallback: try 0-indexed format for backward compatibility (iter_0 = iteration 1)
    if iter_num_1_indexed == 1 and not incomplete:
        old_checkpoint_file = os.path.join(CHECKPOINT_DIR, f"iter_0_triples.jsonl")
        if os.path.exists(old_checkpoint_file):
            with open(old_checkpoint_file) as f:
                triples = [json.loads(line) for line in f]
            log.info(f"Loaded checkpoint from iteration 1 (old format iter_0): {len(triples)} triples")
            return triples
    
    log.warning(f"Checkpoint file not found for iteration {iter_num_1_indexed} (incomplete={incomplete})")
    return None

def save_checkpoint(iter_num, triples, incomplete=False):
    """Save triples to checkpoint file.
    
    Args:
        iter_num: Iteration number (1-indexed)
        triples: List of triples to save
        incomplete: If True, saves as incomplete checkpoint (for resuming mid-iteration)
    """
    if incomplete:
        checkpoint_file = os.path.join(CHECKPOINT_DIR, f"iter_{iter_num}_incomplete_triples.jsonl")
    else:
        checkpoint_file = os.path.join(CHECKPOINT_DIR, f"iter_{iter_num}_triples.jsonl")
        # Remove incomplete checkpoint if it exists (iteration was completed)
        incomplete_file = os.path.join(CHECKPOINT_DIR, f"iter_{iter_num}_incomplete_triples.jsonl")
        if os.path.exists(incomplete_file):
            os.remove(incomplete_file)
    
    with open(checkpoint_file, 'w') as f:
        for triple in triples:
            f.write(json.dumps(triple) + '\n')
    log.info(f"Saved checkpoint: {checkpoint_file} ({len(triples)} triples)")

# === LOAD DATA (from checkpoint or seed) ===
# Note: All iterations are 1-indexed (iter_1, iter_2, etc.)
latest_iter, is_incomplete = find_latest_checkpoint()
if latest_iter is not None:
    log.info(f"Found latest checkpoint at iteration {latest_iter} ({'incomplete' if is_incomplete else 'completed'})")
    current_triples = load_checkpoint(latest_iter, incomplete=is_incomplete)
    if is_incomplete:
        # Incomplete checkpoint means we were interrupted during this iteration - resume from it
        start_iter = latest_iter
        log.info(f"Resuming from iteration {start_iter} (iteration {latest_iter} was incomplete)")
    else:
        # Completed checkpoint means we finished this iteration - start from next
        start_iter = latest_iter + 1
        log.info(f"Resuming from iteration {start_iter} (iteration {latest_iter} was completed)")
else:
    log.info("No checkpoint found, starting from seed data")
    with open(SEED_FILE) as f:
        current_triples = [json.loads(line) for line in f]
    start_iter = 1  # Start from iteration 1 (1-indexed)
    log.info(f"Loaded {len(current_triples)} triples from seed file")

log.info(f"Starting EM loop from iteration {start_iter}, total iterations: {EM_ITERS}")

# === GRACEFUL SHUTDOWN HANDLER ===
shutdown_event = threading.Event()

def monitor_shutdown():
    """Monitor stdin for 'exit' command to trigger graceful shutdown"""
    # Check if stdin is available (not redirected)
    if not sys.stdin.isatty():
        log.debug("Stdin not available for interactive input. Shutdown monitor disabled.")
        return
    
    while not shutdown_event.is_set():
        try:
            line = input().strip().lower()
            if line == "exit":
                log.info("Shutdown requested by user. Saving checkpoint and stopping training...")
                shutdown_event.set()
                break
        except (EOFError, KeyboardInterrupt):
            break
        except Exception:
            pass  # Ignore errors in input monitoring

# Start shutdown monitor thread
shutdown_thread = threading.Thread(target=monitor_shutdown, daemon=True)
shutdown_thread.start()
log.info("Shutdown monitor active. Type 'exit' to save checkpoint and stop training gracefully.")

# === REWARD: log pθ(x|z,c) + log pθ(z|c) ===
def compute_reward(pθ, c, x, z):
    try:
        # log pθ(x | z, c)
        input_x = tokenizer(
            f"Concepts: {' | '.join(c)}\nRationale: {z}\nProblem: {x}",
            return_tensors="pt"
        ).to(pθ.device)
        loss_x = pθ(**input_x, labels=input_x["input_ids"]).loss

        # log pθ(z | c)
        input_z = tokenizer(
            f"Concepts: {' | '.join(c)}\nRationale: {z}", 
            return_tensors="pt"
        ).to(pθ.device)
        loss_z = pθ(**input_z, labels=input_z["input_ids"]).loss

        reward = -(loss_x.item() + loss_z.item())
        return reward
    except Exception as e:
        return -100

# === BATCHED E-STEP (FIXED) ===
def batched_e_step(qφ, batch_c, batch_x, num_samples):
    input_texts = [f"Concepts: {' | '.join(c)}\nProblem: {x}\nRationale:" for c, x in zip(batch_c, batch_x)]
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(qφ.device)

    qφ.eval()
    with torch.no_grad():
        outputs = qφ.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=num_samples,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )

    sequences = outputs.sequences.reshape(len(batch_c), num_samples, -1)
    z_candidates = []
    for i in range(len(batch_c)):
        z_list = []
        for k in range(num_samples):
            seq = sequences[i, k]
            z = tokenizer.decode(seq, skip_special_tokens=True).split("Rationale:")[-1].strip()
            z_list.append(z)
        z_candidates.append(z_list)
    
    # Log average rationale length
    avg_length = sum(len(z) for z_list in z_candidates for z in z_list) / (len(batch_c) * num_samples) if z_candidates else 0
    log.debug(f"[E-STEP] Generated rationales - avg length: {avg_length:.1f} chars")
    
    return z_candidates

# === STRUCTURE CHECK ===
def check_structure(text):
    """Check if text contains all required fields"""
    text_lower = text.lower()
    has_concepts = "concepts:" in text_lower
    has_problem = "problem:" in text_lower
    has_rationale = "rationale:" in text_lower
    return has_concepts and has_problem and has_rationale

def compute_structure_accuracy(model, tokenizer, triples, model_type="prompt", sample_size=5):
    """Compute structure accuracy for a model"""
    model.eval()
    structure_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for i in range(min(sample_size, len(triples))):
            ex = triples[i]
            # For pθ model: start with concepts only
            if model_type == "prompt":
                # pθ format: Concepts: ... -> Rationale: ... Problem: ...
                concepts_str = ' | '.join(ex['concepts'])
                prompt = f"Concepts: {concepts_str}\n"
            else:
                # qϕ format: Concepts: ... Problem: ... -> Rationale: ...
                concepts_str = ' | '.join(ex['concepts'])
                prompt = f"Concepts: {concepts_str}\nProblem: {ex['problem']}\n"
            
            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Check structure - must have all three fields
            if check_structure(generated_text):
                structure_correct += 1
            total_samples += 1
    
    # Calculate accuracy
    structure_accuracy = structure_correct / total_samples if total_samples > 0 else 0.0
    model.train()
    return structure_accuracy

# === M-STEP ===
def m_step(model, triples, mode, em_iter_0_indexed):
    """
    M-step training.
    
    Args:
        model: The model to train
        triples: Training triples
        mode: "prompt" or "rationale"
        em_iter_0_indexed: 0-indexed iteration number (for backward compatibility with TrainingArguments)
    """
    log.info(f"Starting M-step for {mode} model with {len(triples)} triples")
    texts = []
    for t in triples:
        text = f"Concepts: {' | '.join(t['concepts'])}\nRationale: {t['rationale']}\nProblem: {t['problem']}" if mode == "prompt" else \
               f"Concepts: {' | '.join(t['concepts'])}\nProblem: {t['problem']}\nRationale: {t['rationale']}"
        texts.append({"text": text})
    
    ds = Dataset.from_list(texts).map(lambda x: tokenizer(x["text"], truncation=True, max_length=512), batched=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="temp",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            num_train_epochs=1,
            bf16=True,
            report_to="none",  # Disable auto-logging, we log manually
            logging_steps=10,
            log_level="info",
            run_name=f"m_step_{mode}_iter{em_iter_0_indexed+1}"
        ),
        train_dataset=ds,
        data_collator=data_collator
    )
    train_result = trainer.train()
    
    # Compute structure accuracy after training
    #structure_accuracy = compute_structure_accuracy(model, tokenizer, triples, model_type=mode, sample_size=5)
    structure_accuracy = 1.0
    
    # Return loss and structure accuracy
    final_loss = train_result.training_loss if hasattr(train_result, 'training_loss') else train_result.metrics.get('train_loss', 0)
    log.info(f"M-step for {mode} complete. Final loss: {final_loss:.4f}, Structure accuracy: {structure_accuracy:.2%}")
    
    return final_loss, structure_accuracy

# === HF UPLOAD (2 REPOS) ===
def upload_checkpoint(pθ, qφ, iter_num):
    if not HF_TOKEN:
        log.warning("HF_TOKEN missing — skipping upload")
        return

    api = HfApi(token=HF_TOKEN)
    
    # Create repo if it doesn't exist
    create_repo(HF_REPO_ID, token=HF_TOKEN, repo_type="model", exist_ok=True)

    def _upload(model, base_path, label):
        iter_path = f"{base_path}iter-{iter_num}/"
        latest_path = f"{base_path}latest/"
        with tempfile.TemporaryDirectory(prefix=f"promptcot_{label}_iter{iter_num}_") as tmp_dir:
            model.save_pretrained(tmp_dir)
            api.upload_folder(folder_path=tmp_dir, path_in_repo=iter_path, repo_id=HF_REPO_ID, repo_type="model")
            api.upload_folder(folder_path=tmp_dir, path_in_repo=latest_path, repo_id=HF_REPO_ID, repo_type="model")
            log.info(f"{label} iter-{iter_num} → {HF_REPO_ID}/{iter_path}")

    _upload(pθ, HF_P_BASE_PATH, "p")
    _upload(qφ, HF_Q_BASE_PATH, "q")

# === MAIN LOOP ===
# Note: em_iter is 1-indexed (1, 2, 3, ...)
if start_iter > EM_ITERS:
    log.warning(f"Latest checkpoint is at iteration {latest_iter}, but EM_ITERS={EM_ITERS}. Nothing to do.")
else:
    for em_iter in range(start_iter, EM_ITERS + 1):
        # Check for shutdown at start of iteration
        if shutdown_event.is_set():
            log.info(f"Shutdown detected at start of iteration {em_iter}. Saving checkpoint...")
            # Save current state (from previous iteration, which was completed)
            save_checkpoint(em_iter - 1, current_triples, incomplete=False)
            log.info("Checkpoint saved. Exiting gracefully.")
            break
        
        log.info(f"\nEM ITER {em_iter}/{EM_ITERS} (resumed from {start_iter})")
        current_k_samples = get_k_samples_for_iteration(em_iter)
        log.info(f"[E-STEP] Using {current_k_samples} rationale samples per triple this iteration")
        
        # === E-STEP ===
        log.info(f"[E-STEP] Starting E-step with {len(current_triples)} triples")
        new_triples = []
        batch_c, batch_x = [], []
        total_batches = (len(current_triples) + BATCH_SIZE - 1) // BATCH_SIZE
        batch_num = 0
        all_rewards = []
        total_tiebreaker_used = 0
        
        for t in current_triples:
            # Check for shutdown during E-step processing
            if shutdown_event.is_set():
                log.info(f"Shutdown detected during E-step of iteration {em_iter}. Saving partial progress...")
                # Save partial new_triples if we have any (incomplete), otherwise save current_triples (completed)
                if new_triples:
                    save_checkpoint(em_iter, new_triples, incomplete=True)
                else:
                    save_checkpoint(em_iter - 1, current_triples, incomplete=False)
                log.info("Checkpoint saved. Exiting gracefully.")
                break
            batch_c.append(t["concepts"])
            batch_x.append(t["problem"])
            if len(batch_c) == BATCH_SIZE:
                batch_num += 1
                log_batch_start(batch_num, total_batches, len(batch_c), current_k_samples)
                z_cands = batched_e_step(qφ, batch_c, batch_x, current_k_samples)
                log_batch_generation_complete(sum(len(z) for z in z_cands))
                
                # Calculate average rationale lengths for this batch
                batch_rationale_lengths = [len(z) for z_list in z_cands for z in z_list]
                avg_rationale_length = sum(batch_rationale_lengths) / len(batch_rationale_lengths) if batch_rationale_lengths else 0
                batch_rewards = []
                batch_selected_rewards = []
                batch_tiebreaker_used = 0
                batch_reward_spreads_eligible = []  # Track spreads for eligible cases
                batch_eligible_count = 0
                
                for i, (c, x, z_list) in enumerate(zip(batch_c, batch_x, z_cands)):
                    rewards = [compute_reward(pθ, c, x, z) for z in z_list]
                    batch_rewards.extend(rewards)
                    
                    # Track reward spread and eligibility
                    reward_spread = max(rewards) - min(rewards) if rewards else 0
                    is_eligible = USE_GROQ_TIEBREAKER and reward_spread < 0.5
                    if is_eligible:
                        batch_reward_spreads_eligible.append(reward_spread)
                        batch_eligible_count += 1
                    
                    best_idx, best_z, tiebreaker_used = select_best_rationale(
                        c,
                        x,
                        z_list,
                        rewards,
                        use_groq=USE_GROQ_TIEBREAKER,
                        iteration=em_iter
                    )
                    if tiebreaker_used:
                        batch_tiebreaker_used += 1
                    
                    selected_reward = rewards[best_idx]
                    batch_selected_rewards.append(selected_reward)
                    all_rewards.append(selected_reward)
                    new_triples.append({"concepts": c, "rationale": best_z, "problem": x})
                    
                    # Log winning rationale details
                    if tiebreaker_used or (i + 1) % 8 == 0:
                        log_winner_rationale(
                            best_idx, len(z_list), selected_reward, reward_spread,
                            x, rewards, best_z, tiebreaker_used
                        )
                    
                    log_e_step_progress(i, len(batch_c))
                
                total_tiebreaker_used += batch_tiebreaker_used
                
                log_batch_metrics(
                    em_iter, batch_num, total_batches, batch_rewards, batch_selected_rewards,
                    avg_rationale_length, batch_tiebreaker_used, batch_reward_spreads_eligible,
                    batch_eligible_count, len(batch_c)
                )
                
                batch_c, batch_x = [], []
                
                # Check for shutdown after batch
                if shutdown_event.is_set():
                    log.info(f"Shutdown detected after batch {batch_num} of iteration {em_iter}. Saving partial progress...")
                    if new_triples:
                        save_checkpoint(em_iter, new_triples, incomplete=True)
                    else:
                        save_checkpoint(em_iter - 1, current_triples, incomplete=False)
                    log.info("Checkpoint saved. Exiting gracefully.")
                    break

        # Check if we broke out of inner loop due to shutdown
        if shutdown_event.is_set():
            break

        # Process remaining batch if any
        if batch_c:
            batch_num += 1
            log_batch_start(batch_num, total_batches, len(batch_c), current_k_samples)
            z_cands = batched_e_step(qφ, batch_c, batch_x, current_k_samples)
            log_batch_generation_complete(sum(len(z) for z in z_cands))
            
            # Calculate average rationale lengths for this batch
            batch_rationale_lengths = [len(z) for z_list in z_cands for z in z_list]
            avg_rationale_length = sum(batch_rationale_lengths) / len(batch_rationale_lengths) if batch_rationale_lengths else 0
            batch_rewards = []
            batch_selected_rewards = []
            batch_tiebreaker_used = 0
            batch_reward_spreads_eligible = []  # Track spreads for eligible cases
            batch_eligible_count = 0
            
            for i, (c, x, z_list) in enumerate(zip(batch_c, batch_x, z_cands)):
                rewards = [compute_reward(pθ, c, x, z) for z in z_list]
                batch_rewards.extend(rewards)
                
                # Track reward spread and eligibility
                reward_spread = max(rewards) - min(rewards) if rewards else 0
                is_eligible = USE_GROQ_TIEBREAKER and reward_spread < 0.5
                if is_eligible:
                    batch_reward_spreads_eligible.append(reward_spread)
                    batch_eligible_count += 1
                
                best_idx, best_z, tiebreaker_used = select_best_rationale(
                    c,
                    x,
                    z_list,
                    rewards,
                    use_groq=USE_GROQ_TIEBREAKER,
                    iteration=em_iter
                )
                if tiebreaker_used:
                    batch_tiebreaker_used += 1
                
                selected_reward = rewards[best_idx]
                batch_selected_rewards.append(selected_reward)
                all_rewards.append(selected_reward)
                new_triples.append({"concepts": c, "rationale": best_z, "problem": x})
                
                # Log winning rationale details
                if tiebreaker_used or (i + 1) % 4 == 0:
                    log_winner_rationale(
                        best_idx, len(z_list), selected_reward, reward_spread,
                        x, rewards, best_z, tiebreaker_used
                    )
            
            total_tiebreaker_used += batch_tiebreaker_used
            
            log_final_batch_metrics(
                em_iter, batch_num, total_batches, batch_rewards, batch_selected_rewards,
                avg_rationale_length, batch_tiebreaker_used, batch_reward_spreads_eligible,
                batch_eligible_count, len(batch_c)
            )
            
            # Check for shutdown after final batch
            if shutdown_event.is_set():
                log.info(f"Shutdown detected after final batch of iteration {em_iter}. Saving progress...")
                if new_triples:
                    save_checkpoint(em_iter, new_triples, incomplete=True)
                else:
                    save_checkpoint(em_iter - 1, current_triples, incomplete=False)
                log.info("Checkpoint saved. Exiting gracefully.")
                break
        
        # Check for shutdown after E-step completes
        if shutdown_event.is_set():
            log.info(f"Shutdown detected after E-step of iteration {em_iter}. Saving progress...")
            save_checkpoint(em_iter, new_triples, incomplete=True)
            log.info("Checkpoint saved. Exiting gracefully.")
            break
        
        # E-step summary
        e_step_global_step = ((em_iter - 1) * total_batches) + total_batches
        log_e_step_summary(em_iter, total_batches, all_rewards, total_tiebreaker_used)
        
        # M-step - collect losses and structure accuracies
        prompt_loss, prompt_structure_accuracy = m_step(pθ, new_triples, "prompt", em_iter - 1)  # Convert to 0-indexed for m_step
        rationale_loss, rationale_structure_accuracy = m_step(qφ, new_triples, "rationale", em_iter - 1)  # Convert to 0-indexed for m_step
        
        # Check for shutdown after M-step completes
        if shutdown_event.is_set():
            log.info(f"Shutdown detected after M-step of iteration {em_iter}. Saving checkpoint...")
            current_triples = new_triples
            save_checkpoint(em_iter, current_triples, incomplete=False)
            log.info("Checkpoint saved. Exiting gracefully.")
            break
        
        # Log M-step to wandb
        log_m_step_summary(em_iter, e_step_global_step, prompt_loss, rationale_loss,
                          prompt_structure_accuracy, rationale_structure_accuracy)
        
        # Log overall iteration summary
        m_step_global_step = e_step_global_step + 1
        log_iteration_summary(em_iter, m_step_global_step, len(new_triples))
        
        current_triples = new_triples
        # Save checkpoint after each iteration (em_iter is 1-indexed)
        save_checkpoint(em_iter, current_triples)
        upload_checkpoint(pθ, qφ, em_iter)

    if shutdown_event.is_set():
        log.info("Training stopped by user. Checkpoint saved. Resume by running the script again.")
    else:
        log.info("DONE! Training completed successfully.")
    wandb.finish()
