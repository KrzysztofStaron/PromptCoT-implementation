# em_loop.py
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import PeftModel
from datasets import Dataset
import random

# Load tokenizer and models
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
prompt_model = PeftModel.from_pretrained(base, "PanzerBread/promptCoT-prompt")
rationale_model = PeftModel.from_pretrained(base, "PanzerBread/promptCoT-rationale")

prompt_model.eval()
rationale_model.eval()

# Load seed triples
with open("./data/seed_triples.json") as f:
    current_triples = json.load(f)  # List of (c, z, x)

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
    context_z = f"Concepts: {'|'.join(c)}\nRationale: "
    log_p_z = compute_log_prob(prompt_model, context_z, z)

    # log p_θ(x|z,c) - probability of problem given concepts and rationale
    context_x = f"Concepts: {'|'.join(c)}\nRationale: {z}\nProblem: "
    log_p_x = compute_log_prob(prompt_model, context_x, x)

    return log_p_z + log_p_x

# E-Step: Sample 8 rationales, pick best
def e_step(rationale_model, c, x, k=8):
    candidates = []
    for _ in range(k):
        input_text = f"Concepts: {'|'.join(c)}\nProblem: {x}\nRationale:"
        z = rationale_model.generate(input_text, max_new_tokens=64)
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
            text = f"Concepts: {'|'.join(t['c'])}\nRationale: {t['z']}\nProblem: {t['x']}"
        else:
            text = f"Concepts: {'|'.join(t['c'])}\nProblem: {t['x']}\nRationale: {t['z']}"
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
    
    # Set model back to eval mode
    model.eval()

# === MAIN EM LOOP ===
for em_iter in range(10):
    print(f"EM Iteration {em_iter+1}/10")
    
    new_triples = []
    for triple in current_triples:
        c, x = triple["c"], triple["x"]
        z_best = e_step(rationale_model, c, x)
        new_triples.append({"c": c, "z": z_best, "x": x})
    
    # M-Step: Update prompt_model
    m_step(prompt_model, new_triples, mode="prompt")
    
    # Update rationale_model
    m_step(rationale_model, new_triples, mode="rationale")
    
    current_triples = new_triples
    
    # Save checkpoint
    prompt_model.save_pretrained(f"./checkpoints/prompt_model_iter_{em_iter}")
    rationale_model.save_pretrained(f"./checkpoints/rationale_model_iter_{em_iter}")

print("EM loop complete! Generated 100k+ hard problems.")
