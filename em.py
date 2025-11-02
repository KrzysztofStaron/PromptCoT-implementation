# em.py 
import json
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
import os
import logging
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = "PanzerBread"

# === CONFIG ===
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
SEED_FILE = "./data/annotated.jsonl"
EM_ITERS = 10
K_SAMPLES = 10  # H200 can handle more samples
BATCH_SIZE = 16  # Larger batches for H200
MAX_LENGTH = 4096  # Full 4k context for H200 (model supports up to 8192)
REPO_PREFIX = "promptcot-"
USE_BEAM_SEARCH = True  # More stable than sampling for H200
ENABLE_WANDB = True  # Enable monitoring for H200 runs

torch.manual_seed(42)
random.seed(42)


if not ENABLE_WANDB:
    os.environ["WANDB_DISABLED"] = "true"
else:
    os.environ["WANDB_DISABLED"] = "false"
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# === TOKENIZER ===
log.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side='left',
    truncation_side='left'
)
tokenizer.pad_token = tokenizer.eos_token
# Set max length to use full context
if hasattr(tokenizer, 'model_max_length'):
    tokenizer.model_max_length = min(tokenizer.model_max_length, MAX_LENGTH)
log.info(f"Using max_length={MAX_LENGTH}")

# === MODELS ===
log.info("Loading base...")
base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# LoRA config for creating new adapters if needed
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

log.info("Loading prompt_model...")
prompt_model_path = "./models/prompt_model"
if os.path.exists(prompt_model_path) and os.path.exists(os.path.join(prompt_model_path, "adapter_config.json")):
    prompt_model = PeftModel.from_pretrained(base, prompt_model_path, is_trainable=True)
    log.info("Loaded existing prompt_model adapter")
else:
    log.warning(f"Prompt model not found at {prompt_model_path}, creating new adapter")
    os.makedirs(prompt_model_path, exist_ok=True)
    prompt_model = get_peft_model(base, lora_config)
    prompt_model.save_pretrained(prompt_model_path)
    log.info(f"Created new prompt_model adapter at {prompt_model_path}")

log.info("Loading rationale_model...")
rationale_model_path = "./models/rationale_model"
if os.path.exists(rationale_model_path) and os.path.exists(os.path.join(rationale_model_path, "adapter_config.json")):
    rationale_model = PeftModel.from_pretrained(base, rationale_model_path, is_trainable=True)
    log.info("Loaded existing rationale_model adapter")
else:
    log.warning(f"Rationale model not found at {rationale_model_path}, creating new adapter")
    os.makedirs(rationale_model_path, exist_ok=True)
    rationale_model = get_peft_model(base, lora_config)
    rationale_model.save_pretrained(rationale_model_path)
    log.info(f"Created new rationale_model adapter at {rationale_model_path}")

# Log GPU memory
if torch.cuda.is_available():
    log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    log.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# === SEED ===
with open(SEED_FILE) as f:
    current_triples = [json.loads(line) for line in f]
log.info(f"Loaded {len(current_triples)} triples")

# === REWARD ===
def compute_reward_batch(prompt_model, batch_c, batch_x, batch_z_list):
    """
    Batch-optimized reward computation for H200.
    Computes rewards for all z candidates in parallel.
    Returns: List of rewards for each (c, x, z) combination
    """
    was_training = prompt_model.training
    prompt_model.eval()
    
    try:
        with torch.no_grad():
            # Prepare batch inputs for prompt_x: Concepts | Rationale | Problem
            prompt_x_texts = []
            prompt_z_texts = []
            batch_indices = []  # Track which z belongs to which (c, x)
            
            for idx, ((c, x), z_list) in enumerate(zip(zip(batch_c, batch_x), batch_z_list)):
                for z in z_list:
                    prompt_x_texts.append(f"Concepts: {' | '.join(c)}\nRationale: {z}\nProblem: {x}")
                    prompt_z_texts.append(f"Concepts: {' | '.join(c)}\nRationale: {z}")
                    batch_indices.append(idx)
            
            # Batch tokenize and compute
            if prompt_x_texts:
                inputs_x = tokenizer(prompt_x_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(prompt_model.device)
                inputs_z = tokenizer(prompt_z_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(prompt_model.device)
                
                # Compute losses in batch (need logits for per-sample computation)
                outputs_x = prompt_model(**inputs_x, labels=inputs_x["input_ids"], return_dict=True)
                outputs_z = prompt_model(**inputs_z, labels=inputs_z["input_ids"], return_dict=True)
                
                # Extract per-sample losses using shift_logits
                batch_size = inputs_x["input_ids"].shape[0]
                logp_x_list = []
                logp_z_list = []
                
                # Compute per-sample log probabilities more accurately
                # Shift logits and compute cross-entropy per sample
                logits_x = outputs_x.logits
                logits_z = outputs_z.logits
                
                for i in range(batch_size):
                    # Shift logits and labels for next-token prediction
                    shift_logits_x = logits_x[i, :-1, :].contiguous()
                    shift_labels_x = inputs_x["input_ids"][i, 1:].contiguous()
                    shift_logits_z = logits_z[i, :-1, :].contiguous()
                    shift_labels_z = inputs_z["input_ids"][i, 1:].contiguous()
                    
                    # Mask padding tokens
                    mask_x = (shift_labels_x != tokenizer.pad_token_id)
                    mask_z = (shift_labels_z != tokenizer.pad_token_id)
                    
                    if mask_x.sum() > 0:
                        # Compute per-sample cross-entropy
                        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
                        logp_x = -loss_fct(shift_logits_x[mask_x], shift_labels_x[mask_x]).item() * mask_x.sum().item()
                    else:
                        logp_x = -100
                    
                    if mask_z.sum() > 0:
                        logp_z = -loss_fct(shift_logits_z[mask_z], shift_labels_z[mask_z]).item() * mask_z.sum().item()
                    else:
                        logp_z = -100
                    
                    logp_x_list.append(logp_x)
                    logp_z_list.append(logp_z)
            else:
                logp_x_list = [-100] * len(batch_indices)
                logp_z_list = [-100] * len(batch_indices)
            
            # Group rewards by (c, x) pair
            rewards_by_pair = {}
            for i, idx in enumerate(batch_indices):
                if idx not in rewards_by_pair:
                    rewards_by_pair[idx] = []
                rewards_by_pair[idx].append(logp_x_list[i] + logp_z_list[i])
            
            # Return as list of lists (one list per (c, x) pair)
            result = [rewards_by_pair.get(i, [-100]) for i in range(len(batch_c))]
            
        if was_training:
            prompt_model.train()
        
        return result
    except Exception as e:
        log.error(f"Error in compute_reward_batch: {e}")
        if was_training:
            prompt_model.train()
        # Return invalid rewards
        return [[-100] * len(z_list) for z_list in batch_z_list]

def compute_reward(prompt_model, c, x, z):
    """
    Single reward computation (fallback/compatibility).
    """
    return compute_reward_batch(prompt_model, [c], [x], [[z]])[0][0]

# === BATCHED E-STEP (H200-optimized) ===
def batched_e_step(rationale_model, batch_c, batch_x):
    input_texts = [f"Concepts: {' | '.join(c)}\nProblem: {x}\nRationale:" for c, x in zip(batch_c, batch_x)]
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(rationale_model.device)

    was_training = rationale_model.training
    rationale_model.eval()
    
    with torch.no_grad():
        if USE_BEAM_SEARCH:
            # Beam search for more stable/coherent rationale generation
            outputs = rationale_model.generate(
                **inputs,
                max_new_tokens=128,  # Longer rationale for H200
                num_beams=K_SAMPLES,
                num_return_sequences=K_SAMPLES,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                early_stopping=True
            )
        else:
            # Sampling fallback
            outputs = rationale_model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=K_SAMPLES,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True
            )
    
    if was_training:
        rationale_model.train()

    sequences = outputs.sequences.reshape(len(batch_c), K_SAMPLES, -1)
    z_candidates = []
    for i in range(len(batch_c)):
        z_list = []
        for k in range(K_SAMPLES):
            seq = sequences[i, k]
            z = tokenizer.decode(seq, skip_special_tokens=True).split("Rationale:")[-1].strip()
            z_list.append(z)
        z_candidates.append(z_list)
    return z_candidates

# === M-STEP ===
def tokenize_with_mask(examples, mask_keyword):
    """
    Tokenize text and create labels where only tokens after mask_keyword contribute to loss.
    For causal LM: labels[i] = input_ids[i+1], with -100 to ignore prefix tokens.
    """
    texts = examples["text"]
    encoded = tokenizer(texts, truncation=True, max_length=MAX_LENGTH, padding=False)
    
    # Tokenize the keyword to find it in the sequence
    keyword_tokens = tokenizer.encode(mask_keyword, add_special_tokens=False)
    
    labels = []
    for i, text in enumerate(texts):
        input_ids = encoded["input_ids"][i]
        # Find keyword tokens in the actual sequence (more reliable than encoding prefix separately)
        mask_start_idx = None
        for j in range(len(input_ids) - len(keyword_tokens) + 1):
            if input_ids[j:j+len(keyword_tokens)] == keyword_tokens:
                mask_start_idx = j + len(keyword_tokens)  # Start predicting after keyword
                break
        
        # If keyword not found, fallback: encode prefix to find position
        if mask_start_idx is None:
            prefix = text.split(mask_keyword)[0] + mask_keyword
            prefix_encoded = tokenizer(prefix, add_special_tokens=True, truncation=False)
            mask_start_idx = len(prefix_encoded["input_ids"])
        
        # Create labels: -100 for prefix (including keyword itself), input_ids[i+1] for target tokens
        label = [-100] * len(input_ids)
        # For causal LM: at position j, we predict input_ids[j+1]
        for j in range(mask_start_idx - 1, len(input_ids) - 1):
            label[j] = input_ids[j + 1]
        labels.append(label)
    
    encoded["labels"] = labels
    return encoded

def m_step(model, triples, mode):
    log.info(f"Starting M-step for {mode} model with {len(triples)} triples")
    texts = []
    for idx, t in enumerate(triples):
        if idx % 50 == 0:
            log.info(f"  Formatting triple {idx}/{len(triples)}")
        text = f"Concepts: {' | '.join(t['concepts'])}\nRationale: {t['rationale']}\nProblem: {t['problem']}" if mode == "prompt" else \
               f"Concepts: {' | '.join(t['concepts'])}\nProblem: {t['problem']}\nRationale: {t['rationale']}"
        texts.append({"text": text})
    
    log.info("Creating dataset...")
    mask_keyword = "Problem:" if mode == "prompt" else "Rationale:"
    ds = Dataset.from_list(texts).map(lambda examples: tokenize_with_mask(examples, mask_keyword), batched=True)
    log.info("Dataset ready. Starting training...")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="temp",
            per_device_train_batch_size=8,  # H200 can handle larger batches
            gradient_accumulation_steps=2,  # Effective batch size = 16
            num_train_epochs=1,
            bf16=True,
            optim="adamw_torch_fused",  # Fused optimizer for H200
            learning_rate=2e-5,
            lr_scheduler_type="cosine",
            warmup_steps=50,
            report_to="wandb" if ENABLE_WANDB else "none",
            logging_steps=50,
            save_strategy="epoch",
            save_total_limit=2,  # Keep only last 2 checkpoints
            fp16_full_eval=True,  # Faster evaluation
            dataloader_num_workers=4,  # Parallel data loading
            dataloader_pin_memory=True,
        ),
        train_dataset=ds,
        data_collator=data_collator
    )
    trainer.train()
    log.info(f"M-step for {mode} complete")

# === HF UPLOAD (2 REPOS) ===
def upload_checkpoint(prompt_model, rationale_model, iter_num):
    if not HF_TOKEN:
        log.warning("HF_TOKEN missing — skipping upload")
        return

    api = HfApi(token=HF_TOKEN)
    
    # prompt_model → promptcot-p
    prompt_model.save_pretrained(f"./temp_p_iter{iter_num}")
    p_repo = f"{HF_USERNAME}/{REPO_PREFIX}p"
    create_repo(p_repo, token=HF_TOKEN, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=f"./temp_p_iter{iter_num}", path_in_repo=f"iter-{iter_num}", repo_id=p_repo, repo_type="model")
    api.upload_folder(folder_path=f"./temp_p_iter{iter_num}", path_in_repo="latest", repo_id=p_repo, repo_type="model")
    log.info(f"p iter-{iter_num} → {p_repo}/iter-{iter_num}")

    # rationale_model → promptcot-q
    rationale_model.save_pretrained(f"./temp_q_iter{iter_num}")
    q_repo = f"{HF_USERNAME}/{REPO_PREFIX}q"
    create_repo(q_repo, token=HF_TOKEN, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=f"./temp_q_iter{iter_num}", path_in_repo=f"iter-{iter_num}", repo_id=q_repo, repo_type="model")
    api.upload_folder(folder_path=f"./temp_q_iter{iter_num}", path_in_repo="latest", repo_id=q_repo, repo_type="model")
    log.info(f"q iter-{iter_num} → {q_repo}/iter-{iter_num}")

# === HELPER: Softmax-weighted sampling for stability ===
def select_best_z(z_list, rewards, use_softmax=True, temperature=1.0):
    """
    Select rationale using softmax-weighted sampling for EM stability.
    Falls back to max if all rewards are invalid.
    """
    valid_indices = [i for i, r in enumerate(rewards) if r > -99]
    if not valid_indices:
        return z_list[0]  # Fallback
    
    if not use_softmax or len(valid_indices) == 1:
        # Use max for single valid candidate
        best_idx = max(valid_indices, key=lambda i: rewards[i])
        return z_list[best_idx]
    
    # Softmax-weighted sampling
    valid_rewards = torch.tensor([rewards[i] for i in valid_indices])
    probs = torch.softmax(valid_rewards / temperature, dim=0)
    selected_idx = torch.multinomial(probs, 1).item()
    return z_list[valid_indices[selected_idx]]

# === MAIN LOOP ===
for em_iter in range(EM_ITERS):
    log.info(f"\n{'='*60}")
    log.info(f"EM ITER {em_iter+1}/{EM_ITERS}")
    log.info(f"{'='*60}")
    new_triples = []
    batch_c, batch_x = [], []
    
    # Compute rewards for all z candidates
    for t in current_triples:
        batch_c.append(t["concepts"])
        batch_x.append(t["problem"])
        if len(batch_c) == BATCH_SIZE:
            z_cands = batched_e_step(rationale_model, batch_c, batch_x)
            # Batch-optimized reward computation
            batch_rewards = compute_reward_batch(prompt_model, batch_c, batch_x, z_cands)
            for i, (c, x, z_list) in enumerate(zip(batch_c, batch_x, z_cands)):
                rewards = batch_rewards[i]
                best_z = select_best_z(z_list, rewards, use_softmax=(em_iter > 0))
                new_triples.append({"concepts": c, "rationale": best_z, "problem": x})
            batch_c, batch_x = [], []

    # Process remaining batch if any
    if batch_c:
        z_cands = batched_e_step(rationale_model, batch_c, batch_x)
        # Batch-optimized reward computation
        batch_rewards = compute_reward_batch(prompt_model, batch_c, batch_x, z_cands)
        for i, (c, x, z_list) in enumerate(zip(batch_c, batch_x, z_cands)):
            rewards = batch_rewards[i]
            best_z = select_best_z(z_list, rewards, use_softmax=(em_iter > 0))
            new_triples.append({"concepts": c, "rationale": best_z, "problem": x})
    
    log.info(f"Selected {len(new_triples)} triples for M-step")
    
    # M-step: train both models
    m_step(prompt_model, new_triples, "prompt")
    m_step(rationale_model, new_triples, "rationale")
    
    # Update current triples (could optionally mix with previous for stability)
    current_triples = new_triples
    
    # Upload checkpoint (less frequent for H200)
    if (em_iter + 1) % 2 == 0 or em_iter == EM_ITERS - 1:  # Upload every 2 iters + final
        upload_checkpoint(prompt_model, rationale_model, em_iter)
    
    # Simple validation: log average reward on a sample
    if (em_iter + 1) % 2 == 0:
        sample_size = min(10, len(current_triples))
        sample = random.sample(current_triples, sample_size)
        sample_rewards = []
        for t in sample:
            # Use the rationale from the triple
            reward = compute_reward(prompt_model, t["concepts"], t["problem"], t["rationale"])
            sample_rewards.append(reward)
        avg_reward = sum(sample_rewards) / len(sample_rewards) if sample_rewards else -100
        log.info(f"Validation (iter {em_iter+1}): avg reward = {avg_reward:.4f} (sample size: {sample_size})")

log.info("DONE!")
