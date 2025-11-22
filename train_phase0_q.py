# train_phase0_q.py
# Phase 0: Q Model Pre-training (q_phi)
# Goal: Teach the q_phi model the format Concepts -> Problem -> Rationale.
# Model: DeepSeek-R1-Distill-Qwen-7B-Base (Unsloth Optimized)

import os
import re
import sys
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
MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-7B" # 7B for faster training
MAX_SEQ_LENGTH = 8192 # DeepSeek context length
DTYPE = None # Auto-detect (BF16 on H100)
LOAD_IN_4BIT = True

HF_OUTPUT_PATH = f"{HF_VERSION}/q/cold-start"
OUTPUT_DIR = f"./models/{HF_VERSION}/q/cold-start"

# TEST MODE FLAG
TEST_DATASET_ONLY = "--test-dataset" in sys.argv

# --- Dataset Parsing ---
def parse_promptcot_dataset(examples):
    # Format:
    # Concepts -> Problem -> Rationale
    # We want: "Concepts: {concepts}\nProblem: {problem}\nRationale: {rationale}"
    
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
            # "Concepts: ...\nProblem: ...\nRationale: ..."
            text = f"Concepts: {concepts_cleaned}\nProblem: {problem}\nRationale: {rationale}"
            texts.append(text)
            
    return {"text": texts}

def main():
    print(f"🚀 Starting Phase 0: Joint Training on {MODEL_NAME}")
    
    # TEST MODE: Skip model loading, just process and log dataset
    if TEST_DATASET_ONLY:
        print("\n" + "="*60)
        print("TEST MODE: Dataset Processing Only")
        print("="*60 + "\n")
        
        # Load and Format Dataset (small sample)
        print("Loading xl-zhao/PromptCoT-Problem-Generation-Dataset (first 100 samples)...")
        dataset = load_dataset("xl-zhao/PromptCoT-Problem-Generation-Dataset", split="train[:100]")
        print(f"Original size: {len(dataset)}")
        
        dataset = dataset.map(parse_promptcot_dataset, batched=True, remove_columns=dataset.column_names)
        print(f"Formatted size: {len(dataset)}")
        
        # Log first 3 examples
        print("\n" + "="*60)
        print("SAMPLE PROCESSED EXAMPLES:")
        print("="*60)
        for i in range(min(3, len(dataset))):
            print(f"\n--- Example {i+1} ---")
            print(dataset[i]['text'])
            print("\n" + "-"*60)
        
        # Log statistics
        print("\n" + "="*60)
        print("DATASET STATISTICS:")
        print("="*60)
        lengths = [len(text) for text in dataset['text']]
        print(f"Total examples: {len(dataset)}")
        print(f"Avg length (chars): {sum(lengths)/len(lengths):.1f}")
        print(f"Min length: {min(lengths)}")
        print(f"Max length: {max(lengths)}")
        
        print("\n✅ Dataset test complete! Remove --test-dataset flag to train.")
        return
    
    # NORMAL MODE: Full training
    # 1. Load Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        token=HF_TOKEN,
    )
    
    # 2. Add LoRA Adapters (QLoRA with 4-bit base + FULL training of embeddings/head)
    model = FastLanguageModel.get_peft_model(
        model,
        r=64, # Higher rank = more capacity (closer to full fine-tuning)
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                        "gate_proj", "up_proj", "down_proj"],  # LoRA on transformer layers
        lora_alpha=64,  # Doubled to match higher rank
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        modules_to_save=["lm_head"],
    )
    
    # 3. Load and Format Dataset
    print("Loading xl-zhao/PromptCoT-Problem-Generation-Dataset...")
    dataset = load_dataset("xl-zhao/PromptCoT-Problem-Generation-Dataset", split="train")
    print(f"Original size: {len(dataset)}")
    
    dataset = dataset.map(parse_promptcot_dataset, batched=True, remove_columns=dataset.column_names)
    print(f"Formatted size: {len(dataset)}")
    
    # 4. Training Arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=192, 
        gradient_accumulation_steps=1, 
        warmup_steps=100,
        # max_steps=0, # Use epochs (defaults to -1 which means use num_train_epochs)
        num_train_epochs=1, # Single epoch over 40k samples (same compute as 4 epochs × 10k)
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

