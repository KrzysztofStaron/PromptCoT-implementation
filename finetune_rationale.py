
import json
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

# Disable wandb logging
os.environ["WANDB_DISABLED"] = "true"

# Load tokenizer and set pad token
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# qφ(z|c,x)
rationale_model = AutoModelForCausalLM.from_pretrained("gpt2")

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

# Wrap model with LoRA
print("\nApplying LoRA to rationale model...")
rationale_model = get_peft_model(rationale_model, lora_config)
rationale_model.print_trainable_parameters()

# Load data
def load_jsonl(filename):
    with open(filename, 'r') as f:
        return [json.loads(line) for line in f]

print("\nLoading training data...")
rationale_data = load_jsonl("./data/rationale_finetune_data.jsonl")

print(f"Loaded {len(rationale_data)} rationale examples")

# Format data for training
def format_data(data):
    """Format data into prompt + output for causal LM"""
    formatted = []
    for item in data:
        # Combine prompt and output
        text = item["prompt"] + " " + item["output"]
        formatted.append({"text": text})
    return formatted

rationale_formatted = format_data(rationale_data)

# Tokenize data
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding=False
    )

print("Tokenizing datasets...")
rationale_dataset = Dataset.from_list(rationale_formatted)
rationale_dataset = rationale_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT-2 is causal LM, not masked LM
)

# Training arguments for rationale model
training_args_rationale = TrainingArguments(
    output_dir="./models/rationale_model",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    eval_strategy="no",
    save_total_limit=2,
    prediction_loss_only=True,
    remove_unused_columns=False,
    report_to="none",
)

# Fine-tune rationale model
print("\nFine-tuning rationale model qφ(z|c,x)...")
trainer_rationale = Trainer(
    model=rationale_model,
    args=training_args_rationale,
    data_collator=data_collator,
    train_dataset=rationale_dataset,
)

trainer_rationale.train()
trainer_rationale.save_model()
print("\n✓ Rationale model fine-tuning complete!")
print("Model saved to ./models/rationale_model")

