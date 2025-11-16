import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi
from dotenv import load_dotenv
from hf_config import HF_USERNAME, HF_VERSION, HF_REPO_ID, HF_P_BASE_PATH
import logging

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Model config (same as em.py)
MODEL_NAME = "Qwen/Qwen2.5-7B"
PROMPT_INIT_SUBPATH = f"{HF_P_BASE_PATH}cold-start"

# Find latest iteration or use cold-start
def find_latest_iteration_from_hf():
    """Find the latest iteration number from HuggingFace repository."""
    if not HF_TOKEN:
        log.warning("HF_TOKEN missing — cannot check HuggingFace for latest iteration")
        return None
    
    try:
        api = HfApi(token=HF_TOKEN)
        files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type="model", token=HF_TOKEN)
        
        iterations = []
        has_cold_start = False
        for file_path in files:
            if PROMPT_INIT_SUBPATH in file_path:
                has_cold_start = True
            
            if f"{HF_P_BASE_PATH}iter-" in file_path:
                try:
                    parts = file_path.split(f"{HF_P_BASE_PATH}iter-")
                    if len(parts) > 1:
                        iter_str = parts[1].split("/")[0]
                        iter_num = int(iter_str)
                        iterations.append(iter_num)
                except (ValueError, IndexError):
                    continue
        
        if iterations:
            latest = max(iterations)
            log.info(f"Found latest iteration {latest} on HuggingFace")
            return latest
        else:
            if has_cold_start:
                log.info("No iterations found, will load from cold-start")
            return None
    except Exception as e:
        log.warning(f"Failed to check HuggingFace: {e}")
        return None

# Load prompt model (no quantization for generation)
log.info("Loading base model for pθ (full precision)...")
base_p = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side='left',
    truncation_side='left'
)
tokenizer.pad_token = tokenizer.eos_token

latest_iter_hf = find_latest_iteration_from_hf()
if latest_iter_hf is not None:
    p_latest_path = f"{HF_P_BASE_PATH}latest/"
    log.info(f"Loading pθ adapters from {HF_REPO_ID}/{p_latest_path}...")
    pθ = PeftModel.from_pretrained(
        base_p,
        HF_REPO_ID,
        is_trainable=False,
        subfolder=p_latest_path.rstrip("/"),
        token=HF_TOKEN,
    )
else:
    log.info(f"Loading pθ adapters from cold-start: {HF_REPO_ID}/{PROMPT_INIT_SUBPATH}...")
    pθ = PeftModel.from_pretrained(
        base_p,
        HF_REPO_ID,
        is_trainable=False,
        subfolder=PROMPT_INIT_SUBPATH,
        token=HF_TOKEN,
    )

log.info("Model loaded. Generating...")

# User-provided data
foundational_concepts = [
    'Knowledge of place value and its application in multi-digit numbers',
    'Proficiency in basic addition facts, including sums of single-digit numbers',
    'Ability to manipulate digits within a number to form new numbers through addition',
    'Understanding of multi-step problem-solving strategies and the ability to apply them',
    'Development of number sense and reasoning skills, including recognizing single-digit numbers and their properties'
]
level = 'codeforces'
prompt_template = 'Given the foundational programming concepts and specified difficulty level, identify connections among these concepts and develop an olympiad-level coding problem that integrates them with appropriate complexity.\n\nFoundational Programming Concepts:\n1. Knowledge of place value and its application in multi-digit numbers\n2. Proficiency in basic addition facts, including sums of single-digit numbers\n3. Ability to manipulate digits within a number to form new numbers through addition\n4. Understanding of multi-step problem-solving strategies and the ability to apply them\n5. Development of number sense and reasoning skills, including recognizing single-digit numbers and their properties\n\nDifficulty Level: codeforces'

# Format input for prompt model (pθ format: Concepts: ... -> Rationale: ... Problem: ...)
concepts_str = ' | '.join(foundational_concepts)
input_text = f"Concepts: {concepts_str}\n"

# Generate
pθ.eval()
with torch.no_grad():
    inputs = tokenizer(input_text, return_tensors="pt").to(pθ.device)
    outputs = pθ.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n" + "="*60)
print("GENERATED OUTPUT:")
print("="*60)
print(generated_text)
print("="*60)