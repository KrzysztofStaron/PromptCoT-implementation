# generation.py — PROMPTCoT: Generate x (problem) from c (concepts)
import json
import torch
import random
import os
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import snapshot_download

# Load base model
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", padding_side='left')
tokenizer.pad_token = tokenizer.eos_token

# Download the "latest" subfolder from Hugging Face repo
cache_dir = "./models/prompt_model_latest"
if not os.path.exists(cache_dir):
    print("Downloading promptcot-p model from Hugging Face (latest subfolder)...")
    snapshot_download(
        repo_id="PanzerBread/promptcot-p",
        local_dir=cache_dir,
        allow_patterns=["latest/**"],
        local_dir_use_symlinks=False
    )
    # Move files from latest subfolder to the cache_dir root
    latest_path = os.path.join(cache_dir, "latest")
    if os.path.exists(latest_path):
        for item in os.listdir(latest_path):
            src = os.path.join(latest_path, item)
            dst = os.path.join(cache_dir, item)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
            else:
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)

# Load the prompt generation model prompt_model
# Used to generate problems from concepts
prompt_model = PeftModel.from_pretrained(base, cache_dir, is_trainable=False)
prompt_model.eval()

# Load concept pool from mathematics_concepts dataset (more diverse)
concepts = set()
with open("./data/base/mathematics_concepts.jsonl") as f:
    for line in f:
        concepts.update(json.loads(line)["concepts"])
        
concepts = list(concepts)

data = []
N = 100
BATCH_SIZE = 10

# Helper function to clean and extract problem from output
def extract_and_clean_problem(full_output, prompt, item_idx, attempt=0):
    """Extract and clean problem from model output."""
    # Extract problem (x) - everything after "Problem:"
    if "Problem:" in full_output:
        x = full_output.split("Problem:")[-1].strip()
    else:
        # If model didn't generate "Problem:", take everything after the prompt
        x = full_output[len(prompt):].strip() if len(full_output) > len(prompt) else ""
        if attempt == 0:  # Only warn on first attempt
            print(f"Warning: Generated {item_idx+1} did not produce Problem section")
    
    # Clean up the problem text - remove unwanted content
    # Remove chat-style markers first (they often appear at the start)
    chat_markers = ["Human:", "Assistant:", "User:", "System:", "User\n", "Assistant\n"]
    for marker in chat_markers:
        # Check if marker appears anywhere (not just at start)
        if marker in x:
            # If at start, remove it and everything before it
            if x.startswith(marker):
                x = x[len(marker):].strip()
            else:
                # Otherwise, take everything before it
                x = x.split(marker)[0].strip()
    
    # Stop at solution markers (we only want the problem, not the solution)
    # Order matters - check longer patterns first
    solution_markers = [
        "\n\nSolution:",
        "\n\nSolution\n",
        "\n\nSolution ",
        "\nSolution:",
        "\nSolution\n",
        "\nSolution ",
        "\n\nAnswer:",
        "\n\nAnswer\n",
        "\nAnswer:",
        "\nAnswer\n",
        "\n\nRationale:",
        "\nRationale:",
        "\n\nJustification:",
        "\nJustification:",
        "\n\nNote:",
        "\nNote:",
        "Solution:",
        "Answer:",
        "Rationale:",
        "Justification:",
        "The answer is:",
        "The answer is",
    ]
    for marker in solution_markers:
        if marker in x:
            # Find the position and check if it's not part of the problem itself
            idx = x.find(marker)
            # If it appears after a reasonable problem length (at least 50 chars), likely a solution
            if idx > 50 or "\n" in x[:idx]:
                x = x[:idx].strip()
                break  # Found a solution marker, stop processing
    
    # Remove trailing markdown tables or formatting artifacts
    if "##" in x:
        x = x.split("##")[0].strip()
    if "|" in x and x.count("|") > 3:  # Likely a markdown table
        # Find the first occurrence of a table-like pattern
        lines = x.split("\n")
        clean_lines = []
        for line in lines:
            if "|" in line and line.count("|") >= 2:
                break  # Stop at markdown table
            clean_lines.append(line)
        x = "\n".join(clean_lines).strip()
    
    # Remove common chat artifacts
    if "Please let me know" in x:
        x = x.split("Please let me know")[0].strip()
    if "Let me know if" in x:
        x = x.split("Let me know if")[0].strip()
    
    # Clean up: stop at any chat formatting tokens
    if "<" in x:
        x = x.split("<")[0].strip()
    
    # Only remove trailing incomplete sentences if we're very sure it's incomplete
    # Don't truncate if it ends with LaTeX delimiters (might be mid-expression)
    # Check if it ends with incomplete LaTeX (like "$z_" or "\\mathbb{")
    ends_with_incomplete_latex = (
        x.endswith("$") or 
        x.endswith("{") or 
        x.endswith("\\") or
        (x.count("$") % 2 != 0) or  # Unmatched dollar signs
        (x.count("{") > x.count("}"))  # Unmatched braces
    )
    
    if not ends_with_incomplete_latex:
        # Only clean up if we're sure it's not LaTeX
        if not x.endswith("}") and not x.endswith(".") and not x.endswith("?") and not x.endswith("!"):
            # Try to find the last complete sentence, but be conservative
            for punct in [".", "?", "!", "}"]:
                last_punct = x.rfind(punct)
                # Only truncate if punctuation is very close to the end (last 10%)
                if last_punct > len(x) * 0.9:
                    x = x[:last_punct + 1].strip()
                    break
    
    # Final cleanup - remove any trailing whitespace
    x = x.strip()
    
    # Fix common formatting issues
    # Fix unmatched parentheses at the end (common error)
    open_parens = x.count("(")
    close_parens = x.count(")")
    if open_parens > close_parens:
        # Check if the problem ends with an opening parenthesis or similar
        # Common pattern: "Round your answer to two decimal places."
        if x.endswith(".") and "(" in x:
            last_open = x.rfind("(")
            if last_open > len(x) * 0.7 and ")" not in x[last_open:]:
                # Add closing parenthesis before the period
                x = x[:-1] + ")."
        elif not x.endswith(")") and "(" in x:
            last_open = x.rfind("(")
            # If the last opening paren is near the end and no closing paren after it
            if last_open > len(x) * 0.8 and ")" not in x[last_open:]:
                # Add closing paren at the end if it makes sense
                if x.endswith(".") or x.endswith("?"):
                    # Insert before punctuation
                    punct = x[-1]
                    x = x[:-1] + ")" + punct
                else:
                    x = x + ")"
    
    return x

# Generate in batches
for batch_start in range(0, N, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, N)
    batch_size = batch_end - batch_start
    print(f"\nProcessing batch {batch_start//BATCH_SIZE + 1}/{(N + BATCH_SIZE - 1)//BATCH_SIZE} (items {batch_start+1}-{batch_end})")
    
    # Prepare batch prompts
    batch_concepts = []
    batch_prompts = []
    for i in range(batch_start, batch_end):
        # Sample 3 concepts
        c = random.sample(concepts, 3)
        batch_concepts.append(c)
        c_str = " | ".join(c)
        
        prompt = f"""Generate a math problem based on these concepts: {c_str}

Please create a hard mathematical problem that incorporates these concepts
don't generate rationale, nor the solution, only the problem

include instructions to put the final answer within \\boxed{{}}


Problem:"""
        batch_prompts.append(prompt)
    
    # Tokenize batch
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(prompt_model.device)
    
    # Generate in batch
    with torch.no_grad():
        output_ids = prompt_model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and process each output
    for batch_idx in range(batch_size):
        i = batch_start + batch_idx
        full_output = tokenizer.decode(output_ids[batch_idx], skip_special_tokens=True)
        x = extract_and_clean_problem(full_output, batch_prompts[batch_idx], i)
        
        # Validate - ensure we have a reasonable problem
        if len(x) < 20:
            print(f"Warning: Generated {i+1} problem is too short ({len(x)} chars)")
        
        # Check for balanced LaTeX
        dollar_count = x.count("$")
        if dollar_count % 2 != 0:
            print(f"Warning: Generated {i+1} has unmatched dollar signs (LaTeX)")
        
        # Check for balanced braces in LaTeX
        brace_diff = x.count("{") - x.count("}")
        if brace_diff > 2:  # Allow some difference for nested structures
            print(f"Warning: Generated {i+1} may have unbalanced braces")
        
        # Only add non-empty problems
        if x and len(x.strip()) > 0:
            data.append({
                "concepts": batch_concepts[batch_idx],
                "problem": x
            })
            print(f"Generated {i+1}/{N}: problem length={len(x)}")
        else:
            print(f"Skipping {i+1}/{N}: empty problem generated")

# Final save
output_file = "synthetic_generated.txt"
try:
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # Verify file was written
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"\n✅ Saved {len(data)} generated (c, x) pairs to {output_file}")
        print(f"   File size: {file_size:,} bytes")
        if N > len(data):
            print(f"   Note: {N - len(data)} problems were skipped due to being empty or invalid")
    else:
        print(f"\n❌ ERROR: File {output_file} was not created!")
except Exception as e:
    print(f"\n❌ ERROR: Failed to save file {output_file}: {e}")
    import traceback
    traceback.print_exc()