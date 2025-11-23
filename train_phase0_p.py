# train_phase0_p.py
import os
import re
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
import torch
from dotenv import load_dotenv
from hf_config import HF_REPO_ID, HF_VERSION

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# --- Config ---
MODEL_NAME = "unsloth/DeepSeek-R1-Distill-Qwen-7B"
MAX_SEQ_LENGTH = 8192
DTYPE = None
LOAD_IN_4BIT = True

HF_OUTPUT_PATH = f"{HF_VERSION}/p/cold-start"
OUTPUT_DIR = f"./models/{HF_VERSION}/p/cold-start"

def parse_promptcot_dataset(examples):
    prompts = examples['prompt']
    completions = examples['completion']
    texts = []
    for p, c in zip(prompts, completions):
        concepts_match = re.search(r"Foundational Concepts:(.*?)Difficulty Level:", p, re.DOTALL)
        if concepts_match:
            concepts_text = concepts_match.group(1).strip()
            concepts_cleaned = re.sub(r"\d+\.\s*", "", concepts_text)
            concepts_cleaned = " | ".join([line.strip() for line in concepts_cleaned.split('\n') if line.strip()])
        else:
            concepts_cleaned = p

        rationale_match = re.search(r"<!-- BEGIN RATIONALE -->(.*?)(?:<!-- END RATIONALE -->|(?=<!-- BEGIN PROBLEM -->))", c, re.DOTALL)
        problem_match = re.search(r"<!-- BEGIN PROBLEM -->(.*?)<!-- END PROBLEM -->", c, re.DOTALL)

        if rationale_match and problem_match:
            rationale = rationale_match.group(1).strip()
            problem = problem_match.group(1).strip()
            text = f"[CONCEPTS]\n{concepts_cleaned}\n[/CONCEPTS]\n\n[RATIONALE]\n{rationale}\n[/RATIONALE]\n\n[PROBLEM]\n{problem}\n[/PROBLEM]"
            texts.append(text)

    return {"text": texts}

def main():
    print(f"Starting Phase 0: Cold-start on {MODEL_NAME} pθ")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        token=HF_TOKEN,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0,
        use_gradient_checkpointing="unsloth",
    )

    dataset = load_dataset("xl-zhao/PromptCoT-Problem-Generation-Dataset", split="train")
    dataset = dataset.map(parse_promptcot_dataset, batched=True, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        per_device_train_batch_size=80,
        gradient_accumulation_steps=2,
        warmup_steps=20,
        num_train_epochs=1,
        output_dir=OUTPUT_DIR,
        report_to="wandb",
        run_name=f"phase0_coldstart_p_{HF_VERSION}",
        logging_steps=5,
        logging_first_step=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        args=training_args,
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    if HF_TOKEN:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.upload_folder(
            folder_path=OUTPUT_DIR,
            repo_id=HF_REPO_ID,
            path_in_repo=HF_OUTPUT_PATH,
            repo_type="model"
        )

if __name__ == "__main__":
    main()

