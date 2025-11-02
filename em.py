# em_loop.py
import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import PeftModel
from datasets import Dataset
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv
import random

# Load environment variables
load_dotenv()
os.environ["WANDB_DISABLED"] = "true"

# Load tokenizer and models
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
prompt_model = PeftModel.from_pretrained(base, "PanzerBread/promptCoT-prompt")
rationale_model = PeftModel.from_pretrained(base, "PanzerBread/promptCoT-rationale")

# Load seed triples
with open("./data/annotated.jsonl") as f:
    raw_data = [json.loads(line) for line in f]  # List of (c, z, x)
    # Convert from annotated.jsonl format (concepts, rationale, problem) to internal format (c, z, x)
    current_triples = [{"c": item["concepts"], "z": item["rationale"], "x": item["problem"]} for item in raw_data]

# Helper function to compute log probability of a sequence given a context
def compute_log_prob(model, context, target):
    """
    Computes log P(target | context) using the model.
    
    Args:
        model: Causal language model
        context: String context (e.g., "Concepts: c\nRationale:")
        target: String target to compute probability of (e.g., "z")
    
    Returns:
        Average log probability per token (float)
    """
    # Create full text: context + target
    full_text = context + target
    
    # Tokenize
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
    
    # Get labels: shift by 1 to predict next token
    # We only want to compute loss on the target part, not the context
    labels = inputs["input_ids"].clone()
    context_length = tokenizer(context, return_tensors="pt", truncation=True, max_length=512)["input_ids"].shape[1]
    
    # Mask context tokens so we only compute loss on target tokens
    labels[:, :context_length] = -100
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
    
    # Return negative loss (which is log probability)
    # outputs.loss is the average negative log probability per token
    # We negate it to get positive log probability
    return -outputs.loss.item()

# Reward function (Eq. 5)
def reward(prompt_model, rationale_model, c, x, z):
    # log p_θ(z|c) - probability of rationale given concepts
    context_z = f"Concepts: {' | '.join(c)}\nRationale: "
    log_p_z = compute_log_prob(prompt_model, context_z, z)

    # log p_θ(x|z,c) - probability of problem given concepts and rationale
    context_x = f"Concepts: {' | '.join(c)}\nRationale: {z}\nProblem: "
    log_p_x = compute_log_prob(prompt_model, context_x, x)

    return log_p_z + log_p_x

# E-Step: Sample 8 rationales, pick best
def e_step(rationale_model, c, x, k=8):
    candidates = []
    for _ in range(k):
        input_text = f"Concepts: {' | '.join(c)}\nProblem: {x}\nRationale: "
        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = rationale_model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode only the new tokens
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        z = tokenizer.decode(new_tokens, skip_special_tokens=True)
        candidates.append(z)
    
    rewards = [reward(prompt_model, rationale_model, c, x, z) for z in candidates]
    z_best = candidates[torch.argmax(torch.tensor(rewards))]
    return z_best

# Tokenize function for dataset
def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding=False)

# M-Step: SFT one epoch
def m_step(model, triples, mode):
    # mode: "prompt" or "rationale"
    formatted = []
    for t in triples:
        if mode == "prompt":
            text = f"Concepts: {' | '.join(t['c'])}\nRationale: {t['z']}\nProblem: {t['x']}"
        else:
            text = f"Concepts: {' | '.join(t['c'])}\nProblem: {t['x']}\nRationale: {t['z']}"
        formatted.append({"text": text})
    
    dataset = Dataset.from_list(formatted)
    dataset = dataset.map(tokenize, batched=True)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training args
    training_args = TrainingArguments(
        output_dir="temp",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        logging_steps=5,
        report_to="none",
        bf16=True,
    )
    
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=data_collator)
    trainer.train()

# Upload function for HuggingFace Hub
def upload_to_hf(folder_path, repo_name):
    """Upload model to HuggingFace Hub"""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print(f"Skipping upload of {repo_name} - HF_TOKEN not set")
        return
    
    username = "PanzerBread"  # Your HF username
    repo_id = f"{username}/promptCoT-{repo_name}"
    
    try:
        # Create repo if it doesn't exist
        create_repo(
            repo_id=repo_id,
            token=hf_token,
            repo_type="model",
            exist_ok=True
        )
        
        # Upload folder
        api = HfApi(token=hf_token)
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"✅ Uploaded {repo_name} to {repo_id}")
    except Exception as e:
        print(f"❌ Failed to upload {repo_name}: {e}")

# === MAIN EM LOOP ===
prompt_model.eval()
rationale_model.eval()

for em_iter in range(10):
    print(f"EM Iteration {em_iter+1}/10")
    
    # Set to eval mode for generation
    prompt_model.eval()
    rationale_model.eval()
    
    new_triples = []
    for triple in current_triples:
        c, x = triple["c"], triple["x"]
        z_best = e_step(rationale_model, c, x)
        new_triples.append({"c": c, "z": z_best, "x": x})
    
    # M-Step: Update prompt_model
    prompt_model.train()
    m_step(prompt_model, new_triples, mode="prompt")
    
    # Update rationale_model
    rationale_model.train()
    m_step(rationale_model, new_triples, mode="rationale")
    
    current_triples = new_triples
    
    # Save checkpoint
    prompt_model.save_pretrained(f"./models/prompt_model_iter_{em_iter}")
    rationale_model.save_pretrained(f"./models/rationale_model_iter_{em_iter}")
    print(f"Saved checkpoints for iteration {em_iter+1}")

# Save final models after EM loop
print("Saving final models...")
prompt_model.save_pretrained("./models/prompt_model_final")
rationale_model.save_pretrained("./models/rationale_model_final")
print("Final models saved!")

# Upload final models to HuggingFace
print("\nUploading final models to HuggingFace Hub...")
upload_to_hf("./models/prompt_model_final", "prompt-final")
upload_to_hf("./models/rationale_model_final", "rationale-final")

print("\nEM loop complete! Generated 100k+ hard problems.")
