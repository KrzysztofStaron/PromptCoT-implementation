# cold_start.py
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import HfApi, create_repo
import os
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

os.environ["WANDB_DISABLED"] = "true"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
SEED_FILE = "./data/annotated.jsonl"
OUTPUT_DIR_P = "./models/prompt_model"
OUTPUT_DIR_Q = "./models/rationale_model"
HF_USERNAME = "PanzerBread/promptCoT-"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load seed
with open(SEED_FILE) as f:
    seed_data = [json.loads(line) for line in f]

# Format functions
def format_prompt(ex):
    return f"Concepts: {' | '.join(ex['concepts'])}\nRationale: {ex['rationale']}\nProblem: {ex['problem']}"

def format_rationale(ex):
    return f"Concepts: {' | '.join(ex['concepts'])}\nProblem: {ex['problem']}\nRationale: {ex['rationale']}"

# Tokenize
def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding=False)

# Prepare datasets
prompt_texts = [format_prompt(ex) for ex in seed_data]
rationale_texts = [format_rationale(ex) for ex in seed_data]

prompt_ds = Dataset.from_dict({"text": prompt_texts}).map(tokenize, batched=True)
rationale_ds = Dataset.from_dict({"text": rationale_texts}).map(tokenize, batched=True)

# LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# Training args
args = TrainingArguments(
    output_dir="temp",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    logging_steps=10,
    save_steps=200,
    bf16=True,  # bf16 is better for A100 GPUs
    report_to="none",
)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False 
)

# Upload models to HuggingFace Hub
def upload_to_hf(folder_path, repo_name):
    api = HfApi(token=os.getenv("HF_TOKEN"))
    repo_id = f"${hf_username}/{repo_name}"
    
    # Create repo if it doesn't exist
    create_repo(
        repo_id=repo_id,
        token=os.getenv("HF_TOKEN"),
        repo_type="model",
        exist_ok=True
    )
    
    # Upload folder
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Uploaded {repo_name} to HuggingFace Hub")

# Train & Save - loads and trains one model at a time
def train_and_save(dataset, path, repo_name):
    # Load model only when needed
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = get_peft_model(base, lora_config)
    
    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=data_collator)
    trainer.train()
    model.save_pretrained(path)
    print(f"Saved to {path}")
    
    # Upload to HuggingFace immediately if token is set
    if os.getenv("HF_TOKEN"):
        print(f"Uploading {repo_name} to HuggingFace Hub...")
        upload_to_hf(path, repo_name)
    
    # Clean up GPU memory
    del model
    del base
    del trainer
    torch.cuda.empty_cache()

print("Cold-start: Training pθ...")
train_and_save(prompt_ds, OUTPUT_DIR_P, "prompt")

print("Cold-start: Training qφ...")
train_and_save(rationale_ds, OUTPUT_DIR_Q, "rationale")

print("Cold-start complete! Models saved.")


    