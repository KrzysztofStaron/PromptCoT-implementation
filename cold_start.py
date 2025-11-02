# cold_start.py
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

os.environ["WANDB_DISABLED"] = "true"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
SEED_FILE = "./data/seed_triples.json"
OUTPUT_DIR_P = "./models/prompt_model"
OUTPUT_DIR_Q = "./models/rationale_model"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load seed
with open(SEED_FILE) as f:
    seed_data = json.load(f)

# Format functions
def format_prompt(ex):
    return f"Concepts: {' | '.join(ex['concepts'])}\nRationale: {ex['rationale']}\nProblem: {ex['problem']}"

def format_rationale(ex):
    return f"Concepts: {' | '.join(ex['concepts'])}\nProblem: {ex['problem']}\nRationale: {ex['rationale']}"

# Tokenize
def tokenize(texts):
    return tokenizer(texts, truncation=True, max_length=512, padding=False)

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

# Load base + LoRA
base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
pθ = get_peft_model(base, lora_config)
qφ = get_peft_model(base, lora_config)

# Training args
args = TrainingArguments(
    output_dir="temp",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    logging_steps=10,
    save_steps=200,
    fp16=True,
    report_to="none",
)

# Train & Save
def train_and_save(model, dataset, path):
    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=lambda x: x)
    trainer.train()
    model.save_pretrained(path)
    print(f"Saved to {path}")

print("Cold-start: Training pθ...")
train_and_save(pθ, prompt_ds, OUTPUT_DIR_P)

print("Cold-start: Training qφ...")
train_and_save(qφ, rationale_ds, OUTPUT_DIR_Q)

print("Cold-start complete! Models saved.")

# Upload models to HuggingFace Hub
def upload_to_hf(folder_path, repo_name):
    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.upload_folder(
        folder_path=folder_path,
        repo_id=f"PanzerBread/promptCoT-{repo_name}",
        repo_type="model",
    )
    print(f"Uploaded {repo_name} to HuggingFace Hub")

# Upload both models if HF_TOKEN is set
if os.getenv("HF_TOKEN"):
    print("\nUploading models to HuggingFace Hub...")
    upload_to_hf(OUTPUT_DIR_P, "prompt")
    upload_to_hf(OUTPUT_DIR_Q, "rationale")
    print("All models uploaded!")
else:
    print("\nSet HF_TOKEN environment variable to upload models to HuggingFace Hub")


    