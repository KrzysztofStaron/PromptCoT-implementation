# train_phase0_joint.py
# Phase 0: Joint Supervised Pre-training (1 hr)
# Goal: Teach the format Concepts -> Rationale -> Problem perfectly.
# Model: DeepSeek-R1-Distill-Qwen-14B-Base (Unsloth Optimized)

import os
import re
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
import torch
from dotenv import load_dotenv
import wandb
from hf_config import HF_REPO_ID, HF_VERSION, HF_TAGS

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# --- Config ---
MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-14B" # Or Qwen2.5-Coder-14B-Base if preferred
MAX_SEQ_LENGTH = 8192 # DeepSeek context length
DTYPE = None # Auto-detect (BF16 on H100)
LOAD_IN_4BIT = True

OUTPUT_DIR = f"./models/{HF_VERSION}/joint"
HF_OUTPUT_PATH = f"{HF_VERSION}/joint"

# --- Dataset Parsing ---
def parse_promptcot_dataset(examples):
    # Format:
    # Prompt (Concepts) -> Completion (Rationale + Problem)
    # We want: "Concepts: {concepts}\nRationale: {rationale}\nProblem: {problem}"
    
    prompts = examples['prompt']
    completions = examples['completion']
    
    texts = []
    for p, c in zip(prompts, completions):
        # Extract Concepts from Prompt
        # The prompt usually starts with "Given foundational concepts... Foundational Concepts:\n1. ..."
        # We want to extract the list of concepts or just use the text after "Foundational Concepts:"
        
        concepts_match = re.search(r"Foundational Concepts:(.*?)Difficulty Level:", p, re.DOTALL)
        if concepts_match:
            concepts_text = concepts_match.group(1).strip()
            # Clean up numbering "1. ", "2. " etc if needed, or keep as raw list text
            # Simple cleanup: replace newlines with " | "
            concepts_cleaned = re.sub(r"\d+\.\s*", "", concepts_text) # Remove numbers
            concepts_cleaned = " | ".join([line.strip() for line in concepts_cleaned.split('\n') if line.strip()])
        else:
            # Fallback if regex fails
            concepts_cleaned = p
            
        # Extract Rationale and Problem from Completion
        # Tags: <!-- BEGIN RATIONALE --> ... <!-- END RATIONALE --><!-- BEGIN PROBLEM --> ... <!-- END PROBLEM -->
        # Robust regex to handle missing END RATIONALE tag (capture until BEGIN PROBLEM)
        
        rationale_match = re.search(r"<!-- BEGIN RATIONALE -->(.*?)(?:<!-- END RATIONALE -->|(?=<!-- BEGIN PROBLEM -->))", c, re.DOTALL)
        problem_match = re.search(r"<!-- BEGIN PROBLEM -->(.*?)<!-- END PROBLEM -->", c, re.DOTALL)
        
        if rationale_match and problem_match:
            rationale = rationale_match.group(1).strip()
            problem = problem_match.group(1).strip()
            
            # Construct Training Format
            # "Concepts: ...\nRationale: ...\nProblem: ..."
            text = f"Concepts: {concepts_cleaned}\nRationale: {rationale}\nProblem: {problem}"
            texts.append(text)
            
    return {"text": texts}

def main():
    print(f"🚀 Starting Phase 0: Joint Training on {MODEL_NAME}")
    
    # 1. Load Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        token=HF_TOKEN,
    )
    
    # 2. Add LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=64, # Grok-4.1 Recipe: r=128
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    # 3. Load and Format Dataset
    print("Loading xl-zhao/PromptCoT-Problem-Generation-Dataset (First 10k)...")
    dataset = load_dataset("xl-zhao/PromptCoT-Problem-Generation-Dataset", split="train[:10000]")
    print(f"Original size: {len(dataset)}")
    
    dataset = dataset.map(parse_promptcot_dataset, batched=True, remove_columns=dataset.column_names)
    print(f"Formatted size: {len(dataset)}")
    
    # 4. Training Arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=16, # Grok-4.1 Recipe: batch 16
        gradient_accumulation_steps=8,  # Grok-4.1 Recipe: grad accum 8 -> effective 128
        warmup_steps=100,
        max_steps=0, # Use epochs
        num_train_epochs=4, # Phase 0 requirement
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="wandb",
        run_name=f"phase0_joint_{HF_VERSION}",
    )
    
    # 5. Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=True, # Pack for efficiency
        args=training_args,
    )
    
    # 6. Train
    print("Training...")
    trainer.train()
    
    # 7. Save
    print(f"Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # 8. Upload to HF
    if HF_TOKEN:
        print(f"Uploading to HuggingFace: {HF_REPO_ID}/{HF_OUTPUT_PATH}")
        model.push_to_hub(f"{HF_REPO_ID}", token=HF_TOKEN, private=True) # Pushing adapter to root repo? Need subfolder support in unsloth/peft usually requires generic upload
        # Unsloth push_to_hub might push to root. Better to use HfApi for specific subfolders if needed, 
        # or just let it push as a distinct model repo if that's easier, but we stick to the plan's path structure.
        # We'll use HfApi to be safe and precise with the subfolder.
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.upload_folder(
            folder_path=OUTPUT_DIR,
            repo_id=HF_REPO_ID,
            path_in_repo=HF_OUTPUT_PATH,
            repo_type="model"
        )
        
    print("Phase 0 Complete!")

if __name__ == "__main__":
    main()

