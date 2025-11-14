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
import sys
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv
from tieBreaker import select_best_rationale
import wandb
from hf_config import HF_USERNAME, HF_VERSION, HF_REPO_ID, HF_P_BASE_PATH, HF_Q_BASE_PATH

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Configure logging with explicit formatting and force flush
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

# Force stdout to be unbuffered
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# === CONFIG ===
# BASE model (NOT Instruct) - required for faithful PromptCoT 2.0 reproduction
# Both pθ and qφ are initialized from the same base checkpoint
# Paper uses Qwen2.5-32B-Base; we use Qwen2.5-14B-Base (scaled-down version)
MODEL_NAME = "Qwen/Qwen2.5-14B"  # Base model (no -Instruct suffix)
SEED_FILE = "./data/annotated.jsonl"
CHECKPOINT_DIR = "./checkpoints"
EM_ITERS = 10
K_SAMPLES = 4
BATCH_SIZE = 5
USE_GROQ_TIEBREAKER = True

# Create checkpoint directory if it doesn't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# === WANDB INIT ===
wandb.init(
    project="promptcot-em",
    name=f"em_training_{EM_ITERS}iters_k{K_SAMPLES}",
    config={
        "model": MODEL_NAME,
        "em_iterations": EM_ITERS,
        "k_samples": K_SAMPLES,
        "batch_size": BATCH_SIZE,
        "use_groq_tiebreaker": USE_GROQ_TIEBREAKER,
    },
    settings=wandb.Settings(
        _disable_stats=False,
        _disable_meta=False,
    )
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

log.info("Loading pθ...")
pθ = PeftModel.from_pretrained(base_p, "./models/prompt_model", is_trainable=True)

log.info("Loading base model for qφ...")
base_q = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

log.info("Loading qφ...")
qφ = PeftModel.from_pretrained(base_q, "./models/rationale_model", is_trainable=True)

# === CHECKPOINT MANAGEMENT ===
def find_latest_checkpoint():
    """
    Find the latest checkpoint iteration number.
    Returns 1-indexed iteration number (e.g., 1, 2, 3, ...)
    Handles migration from 0-indexed (iter_0) to 1-indexed (iter_1, iter_2, ...).
    """
    if not os.path.exists(CHECKPOINT_DIR):
        return None
    
    checkpoint_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("iter_") and f.endswith("_triples.jsonl")]
    if not checkpoint_files:
        return None
    
    iterations = []
    for f in checkpoint_files:
        try:
            iter_num = int(f.split("_")[1])
            iterations.append(iter_num)
        except (ValueError, IndexError):
            continue
    
    if not iterations:
        return None
    
    max_iter = max(iterations)
    # Handle migration: if max_iter is 0, it's old 0-indexed format (iter_0 = iteration 1 completed)
    # Otherwise, assume 1-indexed format (iter_1 = iteration 1 completed, iter_2 = iteration 2 completed, etc.)
    if max_iter == 0:
        # Old format: iter_0 means iteration 1 completed (1-indexed)
        return 1
    # New format: iter_1 means iteration 1 completed, iter_2 means iteration 2 completed, etc.
    return max_iter

def load_checkpoint(iter_num_1_indexed):
    """
    Load triples from a specific checkpoint iteration.
    
    Args:
        iter_num_1_indexed: 1-indexed iteration number (e.g., 1, 2, 3, ...)
    """
    # Try 1-indexed format first (iter_1, iter_2, ...)
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"iter_{iter_num_1_indexed}_triples.jsonl")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            triples = [json.loads(line) for line in f]
        log.info(f"Loaded checkpoint from iteration {iter_num_1_indexed}: {len(triples)} triples")
        return triples
    
    # Fallback: try 0-indexed format for backward compatibility (iter_0 = iteration 1)
    if iter_num_1_indexed == 1:
        old_checkpoint_file = os.path.join(CHECKPOINT_DIR, f"iter_0_triples.jsonl")
        if os.path.exists(old_checkpoint_file):
            with open(old_checkpoint_file) as f:
                triples = [json.loads(line) for line in f]
            log.info(f"Loaded checkpoint from iteration 1 (old format iter_0): {len(triples)} triples")
            return triples
    
    log.warning(f"Checkpoint file not found for iteration {iter_num_1_indexed}")
    return None

def save_checkpoint(iter_num, triples):
    """Save triples to checkpoint file."""
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"iter_{iter_num}_triples.jsonl")
    with open(checkpoint_file, 'w') as f:
        for triple in triples:
            f.write(json.dumps(triple) + '\n')
    log.info(f"Saved checkpoint: {checkpoint_file} ({len(triples)} triples)")

# === LOAD DATA (from checkpoint or seed) ===
# Note: All iterations are 1-indexed (iter_1, iter_2, etc.)
latest_iter = find_latest_checkpoint()
if latest_iter is not None:
    log.info(f"Found latest checkpoint at iteration {latest_iter}")
    current_triples = load_checkpoint(latest_iter)
    # latest_iter is the completed iteration, so start from the next one
    start_iter = latest_iter + 1
    log.info(f"Resuming from iteration {start_iter}")
else:
    log.info("No checkpoint found, starting from seed data")
    with open(SEED_FILE) as f:
        current_triples = [json.loads(line) for line in f]
    start_iter = 1  # Start from iteration 1 (1-indexed)
    log.info(f"Loaded {len(current_triples)} triples from seed file")

log.info(f"Starting EM loop from iteration {start_iter}, total iterations: {EM_ITERS}")

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
def batched_e_step(qφ, batch_c, batch_x):
    input_texts = [f"Concepts: {' | '.join(c)}\nProblem: {x}\nRationale:" for c, x in zip(batch_c, batch_x)]
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(qφ.device)

    qφ.eval()
    with torch.no_grad():
        outputs = qφ.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            num_return_sequences=K_SAMPLES,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )

    sequences = outputs.sequences.reshape(len(batch_c), K_SAMPLES, -1)
    z_candidates = []
    for i in range(len(batch_c)):
        z_list = []
        for k in range(K_SAMPLES):
            seq = sequences[i, k]
            z = tokenizer.decode(seq, skip_special_tokens=True).split("Rationale:")[-1].strip()
            z_list.append(z)
        z_candidates.append(z_list)
    
    # Log average rationale length
    avg_length = sum(len(z) for z_list in z_candidates for z in z_list) / (len(batch_c) * K_SAMPLES) if z_candidates else 0
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
                max_new_tokens=1024,
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
    for idx, t in enumerate(triples):
        if idx % 50 == 0:
            log.info(f"  Formatting triple {idx}/{len(triples)}")
        text = f"Concepts: {' | '.join(t['concepts'])}\nRationale: {t['rationale']}\nProblem: {t['problem']}" if mode == "prompt" else \
               f"Concepts: {' | '.join(t['concepts'])}\nProblem: {t['problem']}\nRationale: {t['rationale']}"
        texts.append({"text": text})
    
    log.info("Creating dataset...")
    ds = Dataset.from_list(texts).map(lambda x: tokenizer(x["text"], truncation=True), batched=True)
    log.info("Dataset ready. Starting training...")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="temp",
            per_device_train_batch_size=2,
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
    structure_accuracy = compute_structure_accuracy(model, tokenizer, triples, model_type=mode, sample_size=5)
    
    # Return loss and structure accuracy
    final_loss = train_result.training_loss if hasattr(train_result, 'training_loss') else train_result.metrics.get('train_loss', 0)
    log.info(f"M-step for {mode} complete. Final loss: {final_loss:.4f}, Structure accuracy: {structure_accuracy:.2%}")
    
    # Clean up trainer to free VRAM
    del trainer
    del ds
    torch.cuda.empty_cache()
    
    return final_loss, structure_accuracy

# === HF UPLOAD (2 REPOS) ===
def upload_checkpoint(pθ, qφ, iter_num):
    if not HF_TOKEN:
        log.warning("HF_TOKEN missing — skipping upload")
        return

    api = HfApi(token=HF_TOKEN)
    
    # Create repo if it doesn't exist
    create_repo(HF_REPO_ID, token=HF_TOKEN, repo_type="model", exist_ok=True)
    
    # pθ → <HF_VERSION>/p/iter-<iter_num>/ and <HF_VERSION>/p/latest/
    pθ.save_pretrained(f"./temp_p_iter{iter_num}")
    p_iter_path = f"{HF_P_BASE_PATH}iter-{iter_num}/"
    p_latest_path = f"{HF_P_BASE_PATH}latest/"
    api.upload_folder(folder_path=f"./temp_p_iter{iter_num}", path_in_repo=p_iter_path, repo_id=HF_REPO_ID, repo_type="model")
    api.upload_folder(folder_path=f"./temp_p_iter{iter_num}", path_in_repo=p_latest_path, repo_id=HF_REPO_ID, repo_type="model")
    log.info(f"p iter-{iter_num} → {HF_REPO_ID}/{p_iter_path}")

    # qφ → <HF_VERSION>/q/iter-<iter_num>/ and <HF_VERSION>/q/latest/
    qφ.save_pretrained(f"./temp_q_iter{iter_num}")
    q_iter_path = f"{HF_Q_BASE_PATH}iter-{iter_num}/"
    q_latest_path = f"{HF_Q_BASE_PATH}latest/"
    api.upload_folder(folder_path=f"./temp_q_iter{iter_num}", path_in_repo=q_iter_path, repo_id=HF_REPO_ID, repo_type="model")
    api.upload_folder(folder_path=f"./temp_q_iter{iter_num}", path_in_repo=q_latest_path, repo_id=HF_REPO_ID, repo_type="model")
    log.info(f"q iter-{iter_num} → {HF_REPO_ID}/{q_iter_path}")

# === MAIN LOOP ===
# Note: em_iter is 1-indexed (1, 2, 3, ...)
if start_iter > EM_ITERS:
    log.warning(f"Latest checkpoint is at iteration {latest_iter}, but EM_ITERS={EM_ITERS}. Nothing to do.")
else:
    for em_iter in range(start_iter, EM_ITERS + 1):
        log.info(f"\nEM ITER {em_iter}/{EM_ITERS} (resumed from {start_iter})")
        
        # === E-STEP ===
        log.info(f"[E-STEP] Starting E-step with {len(current_triples)} triples")
        new_triples = []
        batch_c, batch_x = [], []
        total_batches = (len(current_triples) + BATCH_SIZE - 1) // BATCH_SIZE
        batch_num = 0
        all_rewards = []
        total_tiebreaker_used = 0
        
        for t in current_triples:
            batch_c.append(t["concepts"])
            batch_x.append(t["problem"])
            if len(batch_c) == BATCH_SIZE:
                batch_num += 1
                log.info(f"\n{'='*80}")
                log.info(f"[E-STEP] Processing batch {batch_num}/{total_batches} ({len(batch_c)} samples)")
                log.info(f"{'='*80}")
                sys.stdout.flush()
                
                log.info(f"[E-STEP] Generating {K_SAMPLES} rationale candidates per sample...")
                sys.stdout.flush()
                z_cands = batched_e_step(qφ, batch_c, batch_x)
                log.info(f"[E-STEP] Generated {sum(len(z) for z in z_cands)} total candidates")
                
                # Calculate average rationale lengths for this batch
                batch_rationale_lengths = [len(z) for z_list in z_cands for z in z_list]
                avg_rationale_length = sum(batch_rationale_lengths) / len(batch_rationale_lengths) if batch_rationale_lengths else 0
                
                log.info(f"[E-STEP] Computing rewards and selecting best rationale...")
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
                    if tiebreaker_used:
                        log.info(f"[WINNER] 🎯 Tiebreaker selected rationale {best_idx+1}/{len(z_list)} (reward={selected_reward:.2f}, spread={reward_spread:.3f})")
                        log.info(f"[WINNER] Problem: {x[:100]}..." if len(x) > 100 else f"[WINNER] Problem: {x}")
                        log.info(f"[WINNER] All candidate rewards: {[f'{r:.3f}' for r in rewards]}")
                        log.info(f"[WINNER] Rationale: {best_z[:150]}..." if len(best_z) > 150 else f"[WINNER] Rationale: {best_z}")
                    elif (i + 1) % 8 == 0:
                        # Log every 8th winning rationale when not using tiebreaker
                        log.info(f"[WINNER] Selected rationale {best_idx+1}/{len(z_list)} (reward={selected_reward:.2f}, spread={reward_spread:.3f})")
                        log.info(f"[WINNER] Rationale: {best_z[:150]}..." if len(best_z) > 150 else f"[WINNER] Rationale: {best_z}")
                    
                    if (i + 1) % 4 == 0:
                        log.info(f"[E-STEP]   Processed {i+1}/{len(batch_c)} samples in batch")
                
                total_tiebreaker_used += batch_tiebreaker_used
                
                # Compute batch statistics
                avg_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0
                max_reward = max(batch_rewards) if batch_rewards else 0
                min_reward = min(batch_rewards) if batch_rewards else 0
                std_reward = (sum((r - avg_reward) ** 2 for r in batch_rewards) / len(batch_rewards)) ** 0.5 if len(batch_rewards) > 1 else 0.0
                avg_selected_reward = sum(batch_selected_rewards) / len(batch_selected_rewards) if batch_selected_rewards else 0
                
                # Compute reward spread statistics for eligible cases
                reward_spread_avg = sum(batch_reward_spreads_eligible) / len(batch_reward_spreads_eligible) if batch_reward_spreads_eligible else 0.0
                reward_spread_min = min(batch_reward_spreads_eligible) if batch_reward_spreads_eligible else 0.0
                reward_spread_max = max(batch_reward_spreads_eligible) if batch_reward_spreads_eligible else 0.0
                
                log.info(f"[E-STEP] Batch {batch_num} complete. Avg reward: {avg_reward:.2f}, Best: {max_reward:.2f}")
                log.info(f"[E-STEP] Batch {batch_num} summary: {batch_tiebreaker_used}/{len(batch_c)} actually used tiebreaker ({batch_tiebreaker_used/len(batch_c)*100:.1f}%), {batch_eligible_count} eligible")
                sys.stdout.flush()
                
                # Log batch metrics to wandb
                global_step = ((em_iter - 1) * total_batches) + batch_num
                log.info(f"[WANDB] Logging batch {batch_num} metrics at step {global_step}")
                wandb.log({
                    "batch/iteration": em_iter,
                    "batch/batch_num": batch_num,
                    "batch/reward_avg_all": avg_reward,
                    "batch/reward_avg_selected": avg_selected_reward,
                    "batch/reward_max": max_reward,
                    "batch/reward_min": min_reward,
                    "batch/reward_std": std_reward,
                    "batch/avg_rationale_length": avg_rationale_length,
                    "batch/tiebreaker_used": batch_tiebreaker_used,
                    "batch/reward_spread_avg": reward_spread_avg,
                    "batch/reward_spread_min": reward_spread_min,
                    "batch/reward_spread_max": reward_spread_max,
                    "batch/reward_spread_eligible_count": batch_eligible_count,
                }, step=global_step)
                log.info(f"[WANDB] Batch {batch_num} logged successfully")
                sys.stdout.flush()
                
                batch_c, batch_x = [], []

        # Process remaining batch if any
        if batch_c:
            batch_num += 1
            log.info(f"\n{'='*80}")
            log.info(f"[E-STEP] Processing final batch {batch_num}/{total_batches} ({len(batch_c)} samples)")
            log.info(f"{'='*80}")
            sys.stdout.flush()
            log.info(f"[E-STEP] Generating {K_SAMPLES} rationale candidates per sample...")
            sys.stdout.flush()
            z_cands = batched_e_step(qφ, batch_c, batch_x)
            log.info(f"[E-STEP] Generated {sum(len(z) for z in z_cands)} total candidates")
            
            # Calculate average rationale lengths for this batch
            batch_rationale_lengths = [len(z) for z_list in z_cands for z in z_list]
            avg_rationale_length = sum(batch_rationale_lengths) / len(batch_rationale_lengths) if batch_rationale_lengths else 0
            
            log.info(f"[E-STEP] Computing rewards and selecting best rationale...")
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
                if tiebreaker_used:
                    log.info(f"[WINNER] 🎯 Tiebreaker selected rationale {best_idx+1}/{len(z_list)} (reward={selected_reward:.2f}, spread={reward_spread:.3f})")
                    log.info(f"[WINNER] Problem: {x[:100]}..." if len(x) > 100 else f"[WINNER] Problem: {x}")
                    log.info(f"[WINNER] All candidate rewards: {[f'{r:.3f}' for r in rewards]}")
                    log.info(f"[WINNER] Rationale: {best_z[:150]}..." if len(best_z) > 150 else f"[WINNER] Rationale: {best_z}")
                elif (i + 1) % 4 == 0:
                    # Log every 4th winning rationale in final batch
                    log.info(f"[WINNER] Selected rationale {best_idx+1}/{len(z_list)} (reward={selected_reward:.2f}, spread={reward_spread:.3f})")
                    log.info(f"[WINNER] Rationale: {best_z[:150]}..." if len(best_z) > 150 else f"[WINNER] Rationale: {best_z}")
            
            total_tiebreaker_used += batch_tiebreaker_used
            
            # Compute batch statistics
            avg_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0
            max_reward = max(batch_rewards) if batch_rewards else 0
            min_reward = min(batch_rewards) if batch_rewards else 0
            std_reward = (sum((r - avg_reward) ** 2 for r in batch_rewards) / len(batch_rewards)) ** 0.5 if len(batch_rewards) > 1 else 0.0
            avg_selected_reward = sum(batch_selected_rewards) / len(batch_selected_rewards) if batch_selected_rewards else 0
            
            # Compute reward spread statistics for eligible cases
            reward_spread_avg = sum(batch_reward_spreads_eligible) / len(batch_reward_spreads_eligible) if batch_reward_spreads_eligible else 0.0
            reward_spread_min = min(batch_reward_spreads_eligible) if batch_reward_spreads_eligible else 0.0
            reward_spread_max = max(batch_reward_spreads_eligible) if batch_reward_spreads_eligible else 0.0
            
            log.info(f"[E-STEP] Final batch complete. Avg reward: {avg_reward:.2f}, Best: {max_reward:.2f}")
            log.info(f"[E-STEP] Final batch summary: {batch_tiebreaker_used}/{len(batch_c)} actually used tiebreaker ({batch_tiebreaker_used/len(batch_c)*100:.1f}%), {batch_eligible_count} eligible")
            sys.stdout.flush()
            
            # Log batch metrics to wandb
            global_step = ((em_iter - 1) * total_batches) + batch_num
            log.info(f"[WANDB] Logging final batch {batch_num} metrics at step {global_step}")
            wandb.log({
                "batch/iteration": em_iter,
                "batch/batch_num": batch_num,
                "batch/reward_avg_all": avg_reward,
                "batch/reward_avg_selected": avg_selected_reward,
                "batch/reward_max": max_reward,
                "batch/reward_min": min_reward,
                "batch/reward_std": std_reward,
                "batch/avg_rationale_length": avg_rationale_length,
                "batch/tiebreaker_used": batch_tiebreaker_used,
                "batch/reward_spread_avg": reward_spread_avg,
                "batch/reward_spread_min": reward_spread_min,
                "batch/reward_spread_max": reward_spread_max,
                "batch/reward_spread_eligible_count": batch_eligible_count,
            }, step=global_step)
            log.info(f"[WANDB] Final batch {batch_num} logged successfully")
            sys.stdout.flush()
        
        # E-step summary
        if all_rewards:
            avg_reward = sum(all_rewards) / len(all_rewards)
            max_reward = max(all_rewards)
            min_reward = min(all_rewards)
            std_reward = (sum((r - avg_reward) ** 2 for r in all_rewards) / len(all_rewards)) ** 0.5 if len(all_rewards) > 1 else 0.0
            log.info(f"[E-STEP] Complete! Selected {len(new_triples)} triples")
            log.info(f"[E-STEP] Reward stats - Avg: {avg_reward:.2f}, Max: {max_reward:.2f}, Min: {min_reward:.2f}, Std: {std_reward:.2f}")
        else:
            avg_reward = 0.0
            max_reward = 0.0
            min_reward = 0.0
            std_reward = 0.0
        
        # Log E-step summary to wandb
        e_step_global_step = ((em_iter - 1) * total_batches) + total_batches
        wandb.log({
            "e_step/iteration": em_iter,
            "e_step/reward_avg": avg_reward,
            "e_step/reward_max": max_reward,
            "e_step/reward_min": min_reward,
            "e_step/reward_std": std_reward,
            "e_step/tiebreaker_used_total": total_tiebreaker_used,
        }, step=e_step_global_step)
        log.info(f"[WANDB] Logged E-step summary at step {em_iter}: reward_avg={avg_reward:.2f}, reward_max={max_reward:.2f}")
        
        # M-step - collect losses and structure accuracies
        # Move qφ to CPU to free VRAM while training pθ
        log.info("[M-STEP] Moving qφ to CPU to free VRAM for pθ training...")
        qφ_device = qφ.device
        qφ.to('cpu')
        torch.cuda.empty_cache()
        
        prompt_loss, prompt_structure_accuracy = m_step(pθ, new_triples, "prompt", em_iter - 1)  # Convert to 0-indexed for m_step
        
        # Move qφ back to GPU and move pθ to CPU for qφ training
        log.info("[M-STEP] Moving qφ back to GPU and moving pθ to CPU...")
        qφ.to(qφ_device)
        pθ_device = pθ.device
        pθ.to('cpu')
        torch.cuda.empty_cache()
        
        rationale_loss, rationale_structure_accuracy = m_step(qφ, new_triples, "rationale", em_iter - 1)  # Convert to 0-indexed for m_step
        
        # Move pθ back to GPU
        log.info("[M-STEP] Moving pθ back to GPU...")
        pθ.to(pθ_device)
        torch.cuda.empty_cache()
        
        # Log M-step to wandb
        m_step_global_step = e_step_global_step + 1
        wandb.log({
            "m_step/iteration": em_iter,
            "m_step/prompt_loss": prompt_loss,
            "m_step/rationale_loss": rationale_loss,
            "m_step/combined_loss": prompt_loss + rationale_loss,
            "m_step/prompt_structure_accuracy": prompt_structure_accuracy,
            "m_step/rationale_structure_accuracy": rationale_structure_accuracy,
        }, step=m_step_global_step)
        log.info(f"[WANDB] Logged M-step at step {m_step_global_step}: prompt_loss={prompt_loss:.4f}, rationale_loss={rationale_loss:.4f}")
        log.info(f"[WANDB] Structure accuracy - prompt: {prompt_structure_accuracy:.2%}, rationale: {rationale_structure_accuracy:.2%}")
        
        # Log overall iteration summary
        iter_summary_step = m_step_global_step + 1
        wandb.log({
            "iteration/num": em_iter,
        }, step=iter_summary_step)
        log.info(f"[WANDB] Logged iteration {em_iter} complete: {len(new_triples)} triples")
        
        current_triples = new_triples
        # Save checkpoint after each iteration (em_iter is 1-indexed)
        save_checkpoint(em_iter, current_triples)
        upload_checkpoint(pθ, qφ, em_iter)

    log.info("DONE!")
    wandb.finish()
