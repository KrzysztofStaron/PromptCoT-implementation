# em.py — FINAL, RUNNING ON H200
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from peft import PeftModel
import os
import logging
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = "PanzerBread/promptcot-"

os.environ["WANDB_DISABLED"] = "true"
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# === CONFIG ===
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
SEED_FILE = "./data/annotated.jsonl"
EM_ITERS = 10
K_SAMPLES = 4
BATCH_SIZE = 8

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

# === REWARD ===
def compute_reward(pθ, c, x, z):
    try:
        input_x = tokenizer(f"Concepts: {' | '.join(c)}\nRationale: {z}\nProblem: {x}", return_tensors="pt").to(pθ.device)
        loss = pθ(**input_x, labels=input_x["input_ids"]).loss
        return -loss.item()
    except:
        return -100

# === BATCHED E-STEP (FIXED) ===
def batched_e_step(qφ, batch_c, batch_x):
    input_texts = [f"Concepts: {' | '.join(c)}\nProblem: {x}\nRationale:" for c, x in zip(batch_c, batch_x)]
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(qφ.device)

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
    return z_candidates

# === M-STEP ===
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
            report_to="none"
        ),
        train_dataset=ds,
        data_collator=data_collator  # ← CRITICAL FIX: Pad dynamically for LM
    )
    trainer.train()
    log.info(f"M-step for {mode} complete")

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
    new_triples = []
    batch_c, batch_x = [], []
    
    for t in current_triples:
        batch_c.append(t["concepts"])
        batch_x.append(t["problem"])
        if len(batch_c) == BATCH_SIZE:
            z_cands = batched_e_step(qφ, batch_c, batch_x)
            for i, (c, x, z_list) in enumerate(zip(batch_c, batch_x, z_cands)):
                best_z = max(z_list, key=lambda z: compute_reward(pθ, c, x, z))
                new_triples.append({"concepts": c, "rationale": best_z, "problem": x})
            batch_c, batch_x = [], []

    # Process remaining batch if any
    if batch_c:
        z_cands = batched_e_step(qφ, batch_c, batch_x)
        for i, (c, x, z_list) in enumerate(zip(batch_c, batch_x, z_cands)):
            best_z = max(z_list, key=lambda z: compute_reward(pθ, c, x, z))
            new_triples.append({"concepts": c, "rationale": best_z, "problem": x})
    
    m_step(pθ, new_triples, "prompt")
    m_step(qφ, new_triples, "rationale")
    current_triples = new_triples
    upload_checkpoint(pθ, qφ, em_iter)

log.info("DONE!")
