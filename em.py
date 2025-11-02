# em.py — FINAL, RUNNING ON H200
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from peft import PeftModel
import os
import logging
import wandb
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = "PanzerBread/promptcot-"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# === CONFIG ===
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
SEED_FILE = "./data/annotated.jsonl"
EM_ITERS = 10
K_SAMPLES = 8
BATCH_SIZE = 8

# Initialize wandb
try:
    wandb.init(
        project="promptcot-2.0",
        name="em-training",
        config={
            "model": MODEL_NAME,
            "k_samples": K_SAMPLES,
            "batch_size": BATCH_SIZE,
            "em_iters": EM_ITERS,
        }
    )
    log.info("✅ Wandb initialized successfully. View logs at: https://wandb.ai")
except Exception as e:
    log.warning(f"⚠️  Failed to initialize wandb: {e}. Continuing without wandb logging.")
    wandb = None

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
log.info("Loading base...")
base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

log.info("Loading pθ...")
pθ = PeftModel.from_pretrained(base, "./models/prompt_model", is_trainable=True)
log.info("Loading qφ...")
qφ = PeftModel.from_pretrained(base, "./models/rationale_model", is_trainable=True)

# === SEED ===
with open(SEED_FILE) as f:
    current_triples = [json.loads(line) for line in f]
log.info(f"Loaded {len(current_triples)} triples")

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
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(qφ.device)

    qφ.eval()
    with torch.no_grad():
        outputs = qφ.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
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

# === M-STEP ===
def m_step(model, triples, mode, em_iter):
    log.info(f"Starting M-step for {mode} model with {len(triples)} triples")
    texts = []
    for idx, t in enumerate(triples):
        if idx % 50 == 0:
            log.info(f"  Formatting triple {idx}/{len(triples)}")
        text = f"Concepts: {' | '.join(t['concepts'])}\nRationale: {t['rationale']}\nProblem: {t['problem']}" if mode == "prompt" else \
               f"Concepts: {' | '.join(t['concepts'])}\nProblem: {t['problem']}\nRationale: {t['rationale']}"
        texts.append({"text": text})
    
    log.info("Creating dataset...")
    ds = Dataset.from_list(texts).map(lambda x: tokenizer(x["text"], truncation=True, max_length=512), batched=True)
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
            report_to="wandb" if wandb else "none",
            logging_steps=10,
            log_level="info",
            run_name=f"m_step_{mode}_iter{em_iter+1}"
        ),
        train_dataset=ds,
        data_collator=data_collator  # ← CRITICAL FIX: Pad dynamically for LM
    )
    train_result = trainer.train()
    
    # Log final training loss to wandb with custom metric name
    final_loss = train_result.training_loss if hasattr(train_result, 'training_loss') else train_result.metrics.get('train_loss', 0)
    if wandb:
        wandb.log({
            f"m_step/{mode}_loss": final_loss,
            f"m_step/{mode}_num_samples": len(triples),
            f"em_iteration": em_iter + 1
        })
    
    log.info(f"M-step for {mode} complete. Final loss: {final_loss:.4f}")

# === HF UPLOAD (2 REPOS) ===
def upload_checkpoint(pθ, qφ, iter_num):
    if not HF_TOKEN:
        log.warning("HF_TOKEN missing — skipping upload")
        return

    api = HfApi(token=HF_TOKEN)
    
    # pθ → promptcot-p
    pθ.save_pretrained(f"./temp_p_iter{iter_num}")
    p_repo = f"{HF_USERNAME}p"
    create_repo(p_repo, token=HF_TOKEN, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=f"./temp_p_iter{iter_num}", path_in_repo=f"iter-{iter_num}", repo_id=p_repo, repo_type="model")
    api.upload_folder(folder_path=f"./temp_p_iter{iter_num}", path_in_repo="latest", repo_id=p_repo, repo_type="model")
    log.info(f"p iter-{iter_num} → {p_repo}/iter-{iter_num}")

    # qφ → promptcot-q
    qφ.save_pretrained(f"./temp_q_iter{iter_num}")
    q_repo = f"{HF_USERNAME}q"
    create_repo(q_repo, token=HF_TOKEN, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=f"./temp_q_iter{iter_num}", path_in_repo=f"iter-{iter_num}", repo_id=q_repo, repo_type="model")
    api.upload_folder(folder_path=f"./temp_q_iter{iter_num}", path_in_repo="latest", repo_id=q_repo, repo_type="model")
    log.info(f"q iter-{iter_num} → {q_repo}/iter-{iter_num}")

# === MAIN LOOP ===
for em_iter in range(EM_ITERS):
    log.info(f"\nEM ITER {em_iter+1}/{EM_ITERS}")
    
    # === E-STEP ===
    log.info(f"[E-STEP] Starting E-step with {len(current_triples)} triples")
    new_triples = []
    batch_c, batch_x = [], []
    total_batches = (len(current_triples) + BATCH_SIZE - 1) // BATCH_SIZE
    batch_num = 0
    all_rewards = []
    
    for t in current_triples:
        batch_c.append(t["concepts"])
        batch_x.append(t["problem"])
        if len(batch_c) == BATCH_SIZE:
            batch_num += 1
            log.info(f"[E-STEP] Processing batch {batch_num}/{total_batches} ({len(batch_c)} samples)")
            
            log.info(f"[E-STEP] Generating {K_SAMPLES} rationale candidates per sample...")
            z_cands = batched_e_step(qφ, batch_c, batch_x)
            log.info(f"[E-STEP] Generated {sum(len(z) for z in z_cands)} total candidates")
            
            log.info(f"[E-STEP] Computing rewards and selecting best rationale...")
            batch_rewards = []
            for i, (c, x, z_list) in enumerate(zip(batch_c, batch_x, z_cands)):
                rewards = [compute_reward(pθ, c, x, z) for z in z_list]
                batch_rewards.extend(rewards)
                all_rewards.append(max(rewards))
                best_z = z_list[rewards.index(max(rewards))]
                new_triples.append({"concepts": c, "rationale": best_z, "problem": x})
                
                if (i + 1) % 4 == 0:
                    log.info(f"[E-STEP]   Processed {i+1}/{len(batch_c)} samples in batch")
            
            avg_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0
            log.info(f"[E-STEP] Batch {batch_num} complete. Avg reward: {avg_reward:.2f}, Best: {max(batch_rewards):.2f}")
            
            # Log batch metrics to wandb
            if wandb:
                wandb.log({
                    f"e_step/batch_reward_avg": avg_reward,
                    f"e_step/batch_reward_max": max(batch_rewards) if batch_rewards else 0,
                    f"e_step/batch_num": batch_num,
                    f"em_iteration": em_iter + 1
                })
            
            batch_c, batch_x = [], []

    # Process remaining batch if any
    if batch_c:
        batch_num += 1
        log.info(f"[E-STEP] Processing final batch {batch_num}/{total_batches} ({len(batch_c)} samples)")
        log.info(f"[E-STEP] Generating {K_SAMPLES} rationale candidates per sample...")
        z_cands = batched_e_step(qφ, batch_c, batch_x)
        log.info(f"[E-STEP] Generated {sum(len(z) for z in z_cands)} total candidates")
        
        log.info(f"[E-STEP] Computing rewards and selecting best rationale...")
        batch_rewards = []
        for i, (c, x, z_list) in enumerate(zip(batch_c, batch_x, z_cands)):
            rewards = [compute_reward(pθ, c, x, z) for z in z_list]
            batch_rewards.extend(rewards)
            all_rewards.append(max(rewards))
            best_z = z_list[rewards.index(max(rewards))]
            new_triples.append({"concepts": c, "rationale": best_z, "problem": x})
        avg_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0
        log.info(f"[E-STEP] Final batch complete. Avg reward: {avg_reward:.2f}, Best: {max(batch_rewards):.2f}")
        
        # Log final batch metrics to wandb
        if wandb:
            wandb.log({
                f"e_step/batch_reward_avg": avg_reward,
                f"e_step/batch_reward_max": max(batch_rewards) if batch_rewards else 0,
                f"e_step/batch_num": batch_num,
                f"em_iteration": em_iter + 1
            })
    
    # E-step summary
    if all_rewards:
        avg_reward = sum(all_rewards) / len(all_rewards)
        max_reward = max(all_rewards)
        min_reward = min(all_rewards)
        log.info(f"[E-STEP] Complete! Selected {len(new_triples)} triples")
        log.info(f"[E-STEP] Reward stats - Avg: {avg_reward:.2f}, Max: {max_reward:.2f}, Min: {min_reward:.2f}")
        
        # Log to wandb
        if wandb:
            wandb.log({
                f"e_step/reward_avg": avg_reward,
                f"e_step/reward_max": max_reward,
                f"e_step/reward_min": min_reward,
                f"e_step/triples_selected": len(new_triples),
                f"em_iteration": em_iter + 1
            })
    
    # M-step with wandb logging
    m_step(pθ, new_triples, "prompt", em_iter)
    m_step(qφ, new_triples, "rationale", em_iter)
    
    current_triples = new_triples
    upload_checkpoint(pθ, qφ, em_iter)
    
    # Log iteration complete
    if wandb:
        wandb.log({
            f"em_iteration/complete": 1,
            f"em_iteration/num_triples": len(current_triples)
        })

log.info("DONE!")
if wandb:
    wandb.finish()
