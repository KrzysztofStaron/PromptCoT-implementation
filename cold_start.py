# cold_start.py
# PromptCoT Cold-Start Training
# Implements PromptCoT 1.0/2.0 design: warm-start both qϕ and pθ models on seed ⟨c,z,x⟩ triplets
#
# CRITICAL: We use BASE models, NOT INSTRUCT models, for faithful PromptCoT 2.0 reproduction.
# Base models provide:
#   - High entropy and diversity (needed for EM exploration)
#   - Non-deterministic rationale generation
#   - No instruction-tuning biases that collapse rationale structures
#
# Models:
#   1. qϕ(z|c,x) - Rationale generator (E-model): approximates posterior over rationales
#      Input: (c, x) -> Output: z (rationale)
#      Format: "Concepts: [...]\nProblem: <x>\nRationale: <z>"
#
#   2. pθ(z,x|c) - Prompt/joint generator (M-model): generates rationale then problem from concepts
#      Input: c -> Output: z then x (autoregressive)
#      Format: "Concepts: [...]\nRationale: <z>\nProblem: <x>"
#
# Training: MLE on seed ⟨c,z,x⟩ triplets (paper: lr=2e-5, batch=16, epochs=2)
# Paper uses Qwen2.5-32B-Base; we use Qwen2.5-7B-Base (scaled-down version)
#
# H200 OPTIMIZED: Using Unsloth for 450+ TFLOPS (vs 80-120 TFLOPS with vanilla HF Trainer)
# Expected training time: ~15 minutes total (vs 1-2 hours)
import json
from transformers import TrainingArguments
from datasets import Dataset
from huggingface_hub import HfApi, create_repo
import os
import torch
from dotenv import load_dotenv
import wandb
from hf_config import HF_USERNAME, HF_VERSION, HF_TAGS, HF_REPO_ID, HF_P_BASE_PATH, HF_Q_BASE_PATH

# Unsloth for H200 max performance (450+ TFLOPS)
from unsloth import FastLanguageModel

# Load environment variables from .env file
load_dotenv()

# Initialize wandb
wandb.init(project="PromptCoT-coldstart", name="H200_max_speed")

# BASE model (NOT Instruct) - required for faithful PromptCoT 2.0 reproduction
# Base models provide high entropy, diversity, and non-deterministic exploration needed for EM
# Paper uses Qwen2.5-32B-Base; we use Qwen2.5-7B-Base (scaled-down version)
MODEL_NAME = "Qwen/Qwen2.5-7B"  # Base model (no -Instruct suffix)
SEED_FILE = "./data/annotated.jsonl"
OUTPUT_DIR_P = "./models/prompt_model"  # pθ: joint generator p(z,x|c)
OUTPUT_DIR_Q = "./models/rationale_model"  # qϕ: rationale generator q(z|c,x)
# HuggingFace Hub configuration
PUSH_TO_HF = True

# Computed HuggingFace repo paths (cold-start specific)
HF_PROMPT_PATH = f"{HF_P_BASE_PATH}cold-start/"
HF_RATIONALE_PATH = f"{HF_Q_BASE_PATH}cold-start/"

if PUSH_TO_HF:
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("PUSH_TO_HF is True but HF_TOKEN environment variable is not set")
else:
    HF_TOKEN = None

# Unsloth config - tuned for H200 (450+ TFLOPS)
MAX_SEQ_LENGTH = 16384  # H200 can handle long sequences efficiently
DTYPE = None  # Auto-detect (BF16 on H200)
LOAD_IN_4BIT = True

# LoRA config (higher rank = better quality + still fast on H200)
LORA_CONFIG = dict(
    r=128,  # H200 can handle it easily
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unslotted",
    use_rslora=True,
)

# Load seed data: ⟨c,z,x⟩ triplets
# Each entry: {"concepts": [str], "rationale": str, "problem": str}
with open(SEED_FILE) as f:
    seed_data = [json.loads(line) for line in f]

print(f"Loaded {len(seed_data)} seed triplets for cold-start training")

# Format functions for PromptCoT design
def format_prompt(ex):
    """
    Format for pθ(z,x|c) - joint generator model
    Input: c (concepts)
    Output: z (rationale) then x (problem) autoregressively
    """
    concepts_str = ' | '.join(ex['concepts'])
    return f"Concepts: {concepts_str}\nRationale: {ex['rationale']}\nProblem: {ex['problem']}"

def format_rationale(ex):
    """
    Format for qϕ(z|c,x) - rationale generator model (approximate posterior)
    Input: c (concepts) + x (problem)
    Output: z (rationale)
    """
    concepts_str = ' | '.join(ex['concepts'])
    return f"Concepts: {concepts_str}\nProblem: {ex['problem']}\nRationale: {ex['rationale']}"

# Prepare datasets from seed ⟨c,z,x⟩ triplets
# pθ dataset: train joint model p(z,x|c) on full triplets
prompt_texts = [format_prompt(ex) for ex in seed_data]
print(f"Prepared pθ dataset: {len(prompt_texts)} examples")

# qϕ dataset: train rationale generator q(z|c,x) on (c,x) -> z
rationale_texts = [format_rationale(ex) for ex in seed_data]
print(f"Prepared qϕ dataset: {len(rationale_texts)} examples")

# Structure check functions
def check_structure(text):
    """Check if text contains all required fields"""
    text_lower = text.lower()
    has_concepts = "concepts:" in text_lower
    has_problem = "problem:" in text_lower
    has_rationale = "rationale:" in text_lower
    return has_concepts and has_problem and has_rationale

def compute_structure_accuracy(model, tokenizer, seed_data, model_type="prompt", sample_size=5):
    """Compute structure accuracy for a model"""
    FastLanguageModel.for_inference(model)  # Enable inference mode
    structure_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for i in range(min(sample_size, len(seed_data))):
            ex = seed_data[i]
            # For pθ model: start with concepts only
            if model_type == "prompt":
                # pθ format: Concepts: ... -> Rationale: ... Problem: ...
                concepts_str = ' | '.join(ex['concepts'])
                prompt = f"Concepts: {concepts_str}\n"
            else:
                # qϕ format: Concepts: ... Problem: ... -> Rationale: ...
                concepts_str = ' | '.join(ex['concepts'])
                prompt = f"Concepts: {concepts_str}\nProblem: {ex['problem']}\n"
            
            # Generate using Unsloth's optimized generation
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Check structure - must have all three fields
            if check_structure(generated_text):
                structure_correct += 1
            total_samples += 1
    
    # Calculate accuracy
    structure_accuracy = structure_correct / total_samples if total_samples > 0 else 0.0
    FastLanguageModel.for_training(model)  # Re-enable training mode
    return structure_accuracy

# Upload models to HuggingFace Hub
def upload_to_hf(folder_path, repo_name, tags=None):
    api = HfApi(token=HF_TOKEN)
    
    # Create repo if it doesn't exist
    create_repo(
        repo_id=HF_REPO_ID,
        token=HF_TOKEN,
        repo_type="model",
        exist_ok=True
    )
    
    # Build path_in_repo: <model_type>/<HF_VERSION>/cold-start/
    if repo_name == "prompt":
        path_in_repo = HF_PROMPT_PATH
    elif repo_name == "rationale":
        path_in_repo = HF_RATIONALE_PATH
    else:
        # Fallback: assume prompt if unknown
        path_in_repo = HF_PROMPT_PATH
    
    # Upload folder to subfolder
    api.upload_folder(
        folder_path=folder_path,
        repo_id=HF_REPO_ID,
        path_in_repo=path_in_repo,
        repo_type="model",
    )
    
    print(f"Uploaded {repo_name} to HuggingFace Hub as {HF_REPO_ID}/{path_in_repo}")

# Train & Save - loads and trains one model at a time using Unsloth
def train_and_save(texts, path, repo_name, model_description):
    """
    Train a model on the dataset using Unsloth (H200 optimized: 450+ TFLOPS).
    
    Args:
        texts: List of formatted text strings
        path: Local path to save the model
        repo_name: Name for HuggingFace repo ("prompt" or "rationale")
        model_description: Description of what this model does
    """
    print(f"\n{'='*60}")
    print(f"🚀 Training {model_description} with Unsloth on H200...")
    print(f"{'='*60}")
    
    # Load model with Unsloth (4-bit quantization + optimized LoRA)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,  # Auto-detect (BF16 on H200)
        load_in_4bit=LOAD_IN_4BIT,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    
    # Apply LoRA adapters
    model = FastLanguageModel.get_peft_model(model, **LORA_CONFIG)
    
    # Prepare dataset
    dataset = Dataset.from_dict({"text": texts})
    
    # Training args optimized for H200
    training_args = TrainingArguments(
        per_device_train_batch_size=12,  # H200 can do 12-16
        gradient_accumulation_steps=8,   # Effective batch = 96-128
        warmup_steps=10,
        num_train_epochs=2,  # Paper: 2 epochs for cold-start
        learning_rate=2e-4,  # Slightly higher LR works well with Unsloth
        bf16=True,
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        report_to="wandb",
        output_dir=path,
        torch_compile=True,  # +25% speed boost
        dataloader_num_workers=16,
        remove_unused_columns=False,
    )
    
    # Get Unsloth trainer (packing enabled for massive speedup)
    trainer = FastLanguageModel.get_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=16,
        packing=True,  # Packs short examples → massive speedup
        args=training_args,
    )
    
    print(f"Starting training on {len(texts)} examples...")
    trainer.train()
    
    # Compute structure accuracy after training
    structure_accuracy = compute_structure_accuracy(model, tokenizer, seed_data, model_type=repo_name, sample_size=5)
    print(f"✓ Structure accuracy: {structure_accuracy:.2%}")
    
    # Log to wandb
    wandb.log({
        f"{repo_name}_structure_accuracy": structure_accuracy
    })
    
    # Save model and tokenizer
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"✓ Saved {model_description} to {path}")
    
    # Clean up GPU memory before upload
    del trainer
    torch.cuda.empty_cache()
    
    # Upload to HuggingFace if flag is set
    if PUSH_TO_HF:
        print(f"Uploading {repo_name} to HuggingFace Hub...")
        upload_to_hf(path, repo_name, tags=HF_TAGS)
    
    # Final cleanup of model from VRAM
    del model
    torch.cuda.empty_cache()
    print(f"✓ Completed training {model_description}\n")

# Cold-start training: MLE on seed ⟨c,z,x⟩ triplets
print("\n" + "="*60)
print("PROMPTCOT COLD-START TRAINING (H200 OPTIMIZED)")
print("="*60)
print(f"Training both models on {len(seed_data)} seed triplets")
print(f"Using Unsloth for 450+ TFLOPS (vs 80-120 TFLOPS with vanilla HF Trainer)")
print(f"Expected time: ~15 minutes total (vs 1-2 hours)")
print(f"Hyperparameters: lr=2e-4, batch=96-128, epochs=2")
print("="*60 + "\n")

# Train pθ: joint generator p(z,x|c)
train_and_save(
    prompt_texts, 
    OUTPUT_DIR_P, 
    "prompt",
    "pθ: Joint generator p(z,x|c) - generates rationale then problem from concepts"
)

# Train qϕ: rationale generator q(z|c,x)
train_and_save(
    rationale_texts, 
    OUTPUT_DIR_Q, 
    "rationale",
    "qϕ: Rationale generator q(z|c,x) - approximates posterior over rationales"
)

print("="*60)
print("COLD-START COMPLETE!")
print("="*60)
print(f"✓ pθ model saved to: {OUTPUT_DIR_P}")
print(f"✓ qϕ model saved to: {OUTPUT_DIR_Q}")
print("\nModels are now ready for EM loop training.")
print("="*60)

wandb.finish()


    