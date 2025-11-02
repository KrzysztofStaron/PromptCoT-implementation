
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

# pθ(x|z,c)
prompt_model = AutoModelForCausalLM.from_pretrained("gpt2")

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

# Wrap model with LoRA
print("\nApplying LoRA to prompt model...")
prompt_model = get_peft_model(prompt_model, lora_config)
prompt_model.print_trainable_parameters()

# Load data
def load_jsonl(filename):
    with open(filename, 'r') as f:
        return [json.loads(line) for line in f]

print("\nLoading training data...")
prompt_data = load_jsonl("./data/prompt_finetune_data.jsonl")

print(f"Loaded {len(prompt_data)} prompt examples")

# Format data for training
def format_data(data):
    """Format data into prompt + output for causal LM"""
    formatted = []
    for item in data:
        # Combine prompt and output
        text = item["prompt"] + " " + item["output"]
        formatted.append({"text": text})
    return formatted

prompt_formatted = format_data(prompt_data)

# Tokenize data
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding=False
    )

print("Tokenizing datasets...")
prompt_dataset = Dataset.from_list(prompt_formatted)
prompt_dataset = prompt_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT-2 is causal LM, not masked LM
)

# Training arguments for prompt model
training_args_prompt = TrainingArguments(
    output_dir="./models/prompt_model",
    num_train_epochs=3,
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

# Fine-tune prompt model
print("\nFine-tuning prompt model pθ(x|z,c)...")
trainer_prompt = Trainer(
    model=prompt_model,
    args=training_args_prompt,
    data_collator=data_collator,
    train_dataset=prompt_dataset,
)

trainer_prompt.train()
trainer_prompt.save_model()
print("\n✓ Prompt model fine-tuning complete!")
print("Model saved to ./models/prompt_model")

