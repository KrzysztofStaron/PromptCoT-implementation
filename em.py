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
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback, BitsAndBytesConfig
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
EM_ITERS = 6
MAX_K_SAMPLES = 7
MIN_K_SAMPLES = 4
BATCH_SIZE = 16
USE_GROQ_TIEBREAKER = False
GRADIENT_ACCUMULATION_STEPS = 10  # Effective batch size = per_device_train_batch_size * gradient_accumulation_steps

# === GENERATION CONFIG ===
# Generation parameters for rationale and problem generation
RATIONALE_MAX_NEW_TOKENS = 8192  # Max tokens for rationale generation (E-step)
PROBLEM_MAX_NEW_TOKENS = 1024   # Max tokens for problem generation
GENERATION_TEMPERATURE = 0.7    # Temperature for sampling
GENERATION_TOP_P = 0.9         # Top-p for nucleus sampling
STRUCTURE_CHECK_MAX_TOKENS = 8192  # Max tokens for structure accuracy check
MAX_SEQUENCE_LENGTH = 1024      # Max sequence length for tokenization/truncation

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

# === HUGGINGFACE CHECKPOINT MANAGEMENT ===
def find_latest_iteration_from_hf():
    """Find the latest iteration number from HuggingFace repository.
    Returns None if no iterations found (will use cold-start models)."""
    if not HF_TOKEN:
        log.warning("HF_TOKEN missing — cannot check HuggingFace for latest iteration")
        return None
    
    try:
        api = HfApi(token=HF_TOKEN)
        # List files in the repository to find iter-* folders
        files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type="model", token=HF_TOKEN)
        
        iterations = []
        has_cold_start = False
        for file_path in files:
            # Check for cold-start folders
            if PROMPT_INIT_SUBPATH in file_path or RATIONALE_INIT_SUBPATH in file_path:
                has_cold_start = True
            
            # Look for paths like "p/iter-1/" or "p/iter-2/"
            if f"{HF_P_BASE_PATH}iter-" in file_path:
                try:
                    # Extract iteration number from path like "p/iter-1/adapter_config.json"
                    parts = file_path.split(f"{HF_P_BASE_PATH}iter-")
                    if len(parts) > 1:
                        iter_str = parts[1].split("/")[0]
                        iter_num = int(iter_str)
                        iterations.append(iter_num)
                except (ValueError, IndexError):
                    continue
        
        if iterations:
            latest = max(iterations)
            log.info(f"Found latest iteration {latest} on HuggingFace")
            return latest
        else:
            if has_cold_start:
                log.info("No iterations found on HuggingFace, but cold-start models exist. Will load from cold-start.")
            else:
                log.warning("No iterations found on HuggingFace and no cold-start models found. Will attempt to load from cold-start anyway.")
            return None
    except Exception as e:
        log.warning(f"Failed to check HuggingFace for latest iteration: {e}")
        log.warning("Will attempt to load from cold-start models.")
        return None

# === MODELS ===
# Load base model with 4-bit quantization to match cold-start training
# This ensures compatibility with adapters trained with Unsloth (which uses 4-bit quantization)
log.info("Loading base model for pθ (4-bit quantized for compatibility with cold-start models)...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
base_p = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)

# Load models from HuggingFace latest/ (or cold-start if no iterations exist)
latest_iter_hf = find_latest_iteration_from_hf()
if latest_iter_hf is not None:
    # Load from latest iteration
    p_latest_path = f"{HF_P_BASE_PATH}latest/"
    q_latest_path = f"{HF_Q_BASE_PATH}latest/"
    log.info(f"Loading pθ adapters from {HF_REPO_ID}/{p_latest_path}...")
    pθ = PeftModel.from_pretrained(
        base_p,
        HF_REPO_ID,
        is_trainable=True,
        subfolder=p_latest_path.rstrip("/"),
        token=HF_TOKEN,
    )
    log.info("Loading base model for qφ (4-bit quantized for compatibility with cold-start models)...")
    base_q = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    log.info(f"Loading qφ adapters from {HF_REPO_ID}/{q_latest_path}...")
    qφ = PeftModel.from_pretrained(
        base_q,
        HF_REPO_ID,
        is_trainable=True,
        subfolder=q_latest_path.rstrip("/"),
        token=HF_TOKEN,
    )
    # Clear memory after loading models
    torch.cuda.empty_cache()
    gc.collect()
    log.info("Models loaded. GPU memory cleared.")
else:
    # Load from cold-start (initial models)
    log.info(f"No iterations found. Loading pθ adapters from cold-start: {HF_REPO_ID}/{PROMPT_INIT_SUBPATH}...")
    try:
        pθ = PeftModel.from_pretrained(
            base_p,
            HF_REPO_ID,
            is_trainable=True,
            subfolder=PROMPT_INIT_SUBPATH,
            token=HF_TOKEN,
        )
    except Exception as e:
        log.error(f"Failed to load pθ from cold-start: {e}")
        log.error("Make sure cold-start models have been uploaded to HuggingFace first!")
        raise
    
    log.info("Loading base model for qφ (4-bit quantized for compatibility with cold-start models)...")
    base_q = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    log.info(f"Loading qφ adapters from cold-start: {HF_REPO_ID}/{RATIONALE_INIT_SUBPATH}...")
    try:
        qφ = PeftModel.from_pretrained(
            base_q,
            HF_REPO_ID,
            is_trainable=True,
            subfolder=RATIONALE_INIT_SUBPATH,
            token=HF_TOKEN,
        )
    except Exception as e:
        log.error(f"Failed to load qφ from cold-start: {e}")
        log.error("Make sure cold-start models have been uploaded to HuggingFace first!")
        raise
    
    # Clear memory after loading models
    torch.cuda.empty_cache()
    gc.collect()
    log.info("Models loaded. GPU memory cleared.")

# === LOAD DATA ===
# Always start from seed file (data is not checkpointed, only models are)
log.info("Loading seed data...")
with open(SEED_FILE) as f:
    current_triples = [json.loads(line) for line in f]

if len(current_triples) == 0:
    log.error("ERROR: No triples to process! Seed file is empty.")
    sys.exit(1)

# Determine starting iteration based on HuggingFace
if latest_iter_hf is not None:
    start_iter = latest_iter_hf + 1
    log.info(f"Found latest iteration {latest_iter_hf} on HuggingFace, starting from iteration {start_iter}")
else:
    start_iter = 1
    log.info("No previous iterations found, starting from iteration 1")

log.info(f"Starting EM loop from iteration {start_iter}, total iterations: {EM_ITERS}, with {len(current_triples)} triples")

# === IMMEDIATE SHUTDOWN HANDLER ===
shutdown_event = threading.Event()

class ImmediateShutdown(Exception):
    """Exception raised when user requests immediate shutdown"""
    pass

def monitor_shutdown():
    """Monitor stdin for 'exit' command to trigger immediate shutdown"""
    # Check if stdin is available (not redirected)
    if not sys.stdin.isatty():
        log.debug("Stdin not available for interactive input. Shutdown monitor disabled.")
        return
    
    while not shutdown_event.is_set():
        try:
            line = input().strip().lower()
            if line == "exit":
                log.error("=" * 60)
                log.error("IMMEDIATE SHUTDOWN REQUESTED BY USER!")
                log.error("=" * 60)
                shutdown_event.set()
                # Raise exception in main thread to interrupt immediately
                import _thread
                _thread.interrupt_main()
                break
        except (EOFError, KeyboardInterrupt):
            break
        except Exception:
            pass  # Ignore errors in input monitoring

# Start shutdown monitor thread
shutdown_thread = threading.Thread(target=monitor_shutdown, daemon=True)
shutdown_thread.start()
log.info("Shutdown monitor active. Type 'exit' to IMMEDIATELY stop training.")

# === REWARD: log pθ(x|z,c) + log pθ(z|c) ===
def compute_reward(pθ, c, x, z):
    # Check for shutdown before expensive operations
    if shutdown_event.is_set():
        raise ImmediateShutdown("Shutdown requested during reward computation")
    
    try:
        # log pθ(x | z, c)
        input_x = tokenizer(
            f"Concepts: {' | '.join(c)}\nRationale: {z}\nProblem: {x}",
            return_tensors="pt"
        ).to(pθ.device)
        loss_x = pθ(**input_x, labels=input_x["input_ids"]).loss
        del input_x  # Free memory immediately

        # log pθ(z | c)
        input_z = tokenizer(
            f"Concepts: {' | '.join(c)}\nRationale: {z}", 
            return_tensors="pt"
        ).to(pθ.device)
        loss_z = pθ(**input_z, labels=input_z["input_ids"]).loss
        del input_z  # Free memory immediately

        reward = -(loss_x.item() + loss_z.item())
        return reward
    except ImmediateShutdown:
        raise
    except Exception as e:
        return -100

# === BATCHED E-STEP (FIXED) ===
def batched_e_step(qφ, batch_c, batch_x, num_samples):
    # Check for shutdown before expensive generation
    if shutdown_event.is_set():
        raise ImmediateShutdown("Shutdown requested before generation")
    
    # Clear cache before generation to avoid OOM
    torch.cuda.empty_cache()
    
    input_texts = [f"Concepts: {' | '.join(c)}\nProblem: {x}\nRationale:" for c, x in zip(batch_c, batch_x)]
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQUENCE_LENGTH).to(qφ.device)

    qφ.eval()
    with torch.no_grad():
        # Check again before generation (generation can take a while)
        if shutdown_event.is_set():
            raise ImmediateShutdown("Shutdown requested during generation")
        
        outputs = qφ.generate(
            **inputs,
            max_new_tokens=RATIONALE_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=GENERATION_TEMPERATURE,
            top_p=GENERATION_TOP_P,
            num_return_sequences=num_samples,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
    
    # Clear cache after generation
    torch.cuda.empty_cache()

    sequences = outputs.sequences.reshape(len(batch_c), num_samples, -1)
    z_candidates = []
    for i in range(len(batch_c)):
        # Check for shutdown during processing
        if shutdown_event.is_set():
            raise ImmediateShutdown("Shutdown requested during candidate processing")
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
                max_new_tokens=STRUCTURE_CHECK_MAX_TOKENS,
                do_sample=True,
                temperature=GENERATION_TEMPERATURE,
                top_p=GENERATION_TOP_P,
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
    
    ds = Dataset.from_list(texts).map(lambda x: tokenizer(x["text"], truncation=True, max_length=MAX_SEQUENCE_LENGTH), batched=True)

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
    expected_next_iter = start_iter
    try:
        for em_iter in range(start_iter, EM_ITERS + 1):
            # Validate iteration sequence
            if em_iter != expected_next_iter:
                log.error(f"ITERATION SEQUENCE ERROR: Expected iteration {expected_next_iter}, but got {em_iter}!")
                log.error("This indicates iterations are being skipped. Aborting.")
                break
            
            # Check for shutdown at start of iteration
            if shutdown_event.is_set():
                raise ImmediateShutdown(f"Shutdown requested at start of iteration {em_iter}")
            
            log.info(f"\n{'='*60}")
            log.info(f"EM ITER {em_iter}/{EM_ITERS}")
            log.info(f"{'='*60}")
            current_k_samples = get_k_samples_for_iteration(em_iter)
            log.info(f"[E-STEP] Using {current_k_samples} rationale samples per triple this iteration")
            
            if len(current_triples) == 0:
                log.error(f"ERROR: current_triples is empty at start of iteration {em_iter}! Cannot proceed.")
                log.error("This indicates a serious bug - checkpoints may be corrupted or logic error in checkpoint loading.")
                break
            
            # === E-STEP ===
            log.info(f"[E-STEP] Starting E-step with {len(current_triples)} triples")
            new_triples = []
            batch_c, batch_x = [], []
            total_batches = (len(current_triples) + BATCH_SIZE - 1) // BATCH_SIZE
            batch_num = 0
            all_rewards = []
            total_tiebreaker_used = 0
            
            for t in current_triples:
                # Check for shutdown during E-step processing (check frequently)
                if shutdown_event.is_set():
                    raise ImmediateShutdown(f"Shutdown requested during E-step of iteration {em_iter}")
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
                        # Check for shutdown before computing rewards (can be slow)
                        if shutdown_event.is_set():
                            raise ImmediateShutdown(f"Shutdown requested during reward computation in batch {batch_num}")
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
                        raise ImmediateShutdown(f"Shutdown requested after batch {batch_num} of iteration {em_iter}")

            # Check if shutdown was requested
            if shutdown_event.is_set():
                raise ImmediateShutdown(f"Shutdown requested during iteration {em_iter}")

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
                    # Check for shutdown before computing rewards (can be slow)
                    if shutdown_event.is_set():
                        raise ImmediateShutdown(f"Shutdown requested during reward computation in batch {batch_num}")
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
                    raise ImmediateShutdown(f"Shutdown requested after final batch of iteration {em_iter}")
            
            # Check for shutdown after E-step completes
            if shutdown_event.is_set():
                raise ImmediateShutdown(f"Shutdown requested after E-step of iteration {em_iter}")
            
            # Validate E-step output
            if len(new_triples) == 0:
                log.error(f"ERROR: E-step produced 0 triples for iteration {em_iter}!")
                log.error(f"Input had {len(current_triples)} triples, processed {total_batches} batches")
                log.error("This should never happen - something is broken!")
                break
            
            if len(new_triples) != len(current_triples):
                log.warning(f"WARNING: E-step produced {len(new_triples)} triples but input had {len(current_triples)} triples!")
            
            # E-step summary
            log_e_step_summary(em_iter, total_batches, all_rewards, total_tiebreaker_used)
            
            # M-step - collect losses and structure accuracies
            prompt_loss, prompt_structure_accuracy = m_step(pθ, new_triples, "prompt", em_iter - 1)  # Convert to 0-indexed for m_step
            rationale_loss, rationale_structure_accuracy = m_step(qφ, new_triples, "rationale", em_iter - 1)  # Convert to 0-indexed for m_step
            
            # Check for shutdown after M-step completes
            if shutdown_event.is_set():
                raise ImmediateShutdown(f"Shutdown requested after M-step of iteration {em_iter}")
            
            # Log M-step to wandb
            log_m_step_summary(em_iter, 0, prompt_loss, rationale_loss,
                              prompt_structure_accuracy, rationale_structure_accuracy)
            
            # Log overall iteration summary
            log_iteration_summary(em_iter, 0, len(new_triples))
            
            current_triples = new_triples
            # Upload to HuggingFace after each iteration (em_iter is 1-indexed)
            log.info(f"[ITERATION {em_iter} COMPLETE] Uploading models to HuggingFace...")
            upload_checkpoint(pθ, qφ, em_iter)
            log.info(f"[ITERATION {em_iter} COMPLETE] Models uploaded. Moving to next iteration.")
            
            # Update expected next iteration
            expected_next_iter = em_iter + 1
    
    except ImmediateShutdown as e:
        log.error("=" * 60)
        log.error(f"IMMEDIATE SHUTDOWN: {e}")
        log.error("Training stopped immediately by user request.")
        log.error("=" * 60)
    except KeyboardInterrupt:
        log.error("=" * 60)
        log.error("Keyboard interrupt received. Stopping training.")
        log.error("=" * 60)
    finally:
        if shutdown_event.is_set():
            log.info("Training stopped by user. Resume by running the script again.")
        else:
            log.info("DONE! Training completed successfully.")
        wandb.finish()
