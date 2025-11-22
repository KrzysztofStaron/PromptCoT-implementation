# train_phase1_q.py
# Phase 1: Rationale Model Training (q_phi only) (2 hrs)
# Goal: Specialize q_phi from Phase 0 model.
# Action:
#   - q_phi (Rationale Model): Train on Concepts + Problem -> Rationale

import os
import re
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi
from hf_config import HF_REPO_ID, HF_VERSION

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# --- Config ---
# Load from HuggingFace Phase 0 output
BASE_MODEL_PATH = f"{HF_REPO_ID}/{HF_VERSION}/joint" 

MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-7B" # The base model
MAX_SEQ_LENGTH = 8192
DTYPE = None
LOAD_IN_4BIT = True

OUTPUT_DIR_Q = f"./models/{HF_VERSION}/q/cold-start"
HF_PATH_Q = f"{HF_VERSION}/q/cold-start"

# --- Dataset Parsing ---
def parse_q_phi(examples):
    # q_phi: Concepts + Problem -> Rationale
    # "Concepts: {c}\nProblem: {p}\nRationale: {r}"
    prompts = examples['prompt']
    completions = examples['completion']
    texts = []
    for p, c in zip(prompts, completions):
        concepts_match = re.search(r"Foundational Concepts:(.*?)Difficulty Level:", p, re.DOTALL)
        if concepts_match:
            concepts_cleaned = re.sub(r"\d+\.\s*", "", concepts_match.group(1).strip())
            concepts_cleaned = " | ".join([line.strip() for line in concepts_cleaned.split('\n') if line.strip()])
        else:
            concepts_cleaned = p
            
        # Robust regex for Rationale
        rationale_match = re.search(r"<!-- BEGIN RATIONALE -->(.*?)(?:<!-- END RATIONALE -->|(?=<!-- BEGIN PROBLEM -->))", c, re.DOTALL)
        problem_match = re.search(r"<!-- BEGIN PROBLEM -->(.*?)<!-- END PROBLEM -->", c, re.DOTALL)
        
        if rationale_match and problem_match:
            text = f"Concepts: {concepts_cleaned}\nProblem: {problem_match.group(1).strip()}\nRationale: {rationale_match.group(1).strip()}"
            texts.append(text)
    return {"text": texts}

def train_model(mode, output_dir, hf_path, dataset):
    print(f"--- Training {mode} Model ---")
    
    # Load Base Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        token=HF_TOKEN,
    )
    
    # Add LoRA config (same as Phase 0)
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                        "gate_proj", "up_proj", "down_proj",
                        "embed_tokens", "lm_head"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
    )
    
    # Load Phase 0 Adapters from HuggingFace
    print(f"Loading Phase 0 adapters from {BASE_MODEL_PATH}")
    try:
        model.load_adapter(BASE_MODEL_PATH)
    except Exception as e:
        print(f"Error loading adapter from HF: {e}")
        print("Ensure Phase 0 has finished uploading and the path is correct.")
        return

    # Ensure model is in training mode
    FastLanguageModel.for_training(model)
    
    # Training Args
    training_args = TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        num_train_epochs=3, # Phase 1 requirement
        learning_rate=1e-4, # Slightly lower LR for refinement
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        output_dir=output_dir,
        report_to="wandb",
        run_name=f"phase1_{mode}_{HF_VERSION}",
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=True,
        args=training_args,
    )
    
    trainer.train()
    
    print(f"Saving {mode} to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Upload
    if HF_TOKEN:
        print(f"Uploading {mode} to {HF_REPO_ID}/{hf_path}")
        api = HfApi(token=HF_TOKEN)
        api.upload_folder(
            folder_path=output_dir,
            repo_id=HF_REPO_ID,
            path_in_repo=hf_path,
            repo_type="model"
        )
        
    # Cleanup
    del model, trainer
    torch.cuda.empty_cache()

def main():
    print("🚀 Starting Phase 1: Rationale Model Training (q_phi)")
    
    # Load Data
    print("Loading dataset...")
    ds = load_dataset("xl-zhao/PromptCoT-Problem-Generation-Dataset", split="train")
    
    # Prepare Dataset
    print("Formatting dataset for q_phi...")
    ds_q = ds.map(parse_q_phi, batched=True, remove_columns=ds.column_names)
    
    # Train q_phi
    train_model("q_phi", OUTPUT_DIR_Q, HF_PATH_Q, ds_q)
    
    print("Phase 1 Complete!")

if __name__ == "__main__":
    main()

