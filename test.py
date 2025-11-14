#!/usr/bin/env python3
"""
Display HuggingFace paths and verify environment variables before running em.py and cold_start.py
"""
import os
from dotenv import load_dotenv
from hf_config import HF_USERNAME, HF_VERSION, HF_REPO_ID, HF_P_BASE_PATH, HF_Q_BASE_PATH, HF_TAGS

load_dotenv()

print("=" * 80)
print("HUGGINGFACE CONFIGURATION CHECK")
print("=" * 80)
print()

# Check environment variables
print("ENVIRONMENT VARIABLES:")
print("-" * 80)
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    token_preview = HF_TOKEN[:10] + "..." if len(HF_TOKEN) > 10 else HF_TOKEN
    print(f"✓ HF_TOKEN: SET ({token_preview})")
else:
    print("✗ HF_TOKEN: NOT SET (required for uploading models)")
print()

# Display HF configuration from hf_config.py
print("HUGGINGFACE CONFIGURATION (from hf_config.py):")
print("-" * 80)
print(f"HF_USERNAME: {HF_USERNAME}")
print(f"HF_VERSION: {HF_VERSION}")
print(f"HF_REPO_ID: {HF_REPO_ID}")
print(f"HF_P_BASE_PATH: {HF_P_BASE_PATH}")
print(f"HF_Q_BASE_PATH: {HF_Q_BASE_PATH}")
print(f"HF_TAGS: {HF_TAGS}")
print()

# Display paths for cold_start.py
print("COLD-START TRAINING (cold_start.py) - Model Save Paths:")
print("-" * 80)
HF_PROMPT_PATH = f"{HF_P_BASE_PATH}cold-start/"
HF_RATIONALE_PATH = f"{HF_Q_BASE_PATH}cold-start/"
print(f"Prompt Model (pθ):")
print(f"  Repository: {HF_REPO_ID}")
print(f"  Path in repo: {HF_PROMPT_PATH}")
print(f"  Full path: {HF_REPO_ID}/{HF_PROMPT_PATH}")
print()
print(f"Rationale Model (qφ):")
print(f"  Repository: {HF_REPO_ID}")
print(f"  Path in repo: {HF_RATIONALE_PATH}")
print(f"  Full path: {HF_REPO_ID}/{HF_RATIONALE_PATH}")
print()

# Display paths for em.py
print("EM LOOP TRAINING (em.py) - Model Save Paths:")
print("-" * 80)
print("For each iteration (iter_num = 1, 2, 3, ...):")
print()
print(f"Prompt Model (pθ) - Per Iteration:")
print(f"  Repository: {HF_REPO_ID}")
print(f"  Path in repo: {HF_P_BASE_PATH}iter-<iter_num>/")
print(f"  Example (iter 1): {HF_REPO_ID}/{HF_P_BASE_PATH}iter-1/")
print()
print(f"Prompt Model (pθ) - Latest:")
print(f"  Repository: {HF_REPO_ID}")
print(f"  Path in repo: {HF_P_BASE_PATH}latest/")
print(f"  Full path: {HF_REPO_ID}/{HF_P_BASE_PATH}latest/")
print()
print(f"Rationale Model (qφ) - Per Iteration:")
print(f"  Repository: {HF_REPO_ID}")
print(f"  Path in repo: {HF_Q_BASE_PATH}iter-<iter_num>/")
print(f"  Example (iter 1): {HF_REPO_ID}/{HF_Q_BASE_PATH}iter-1/")
print()
print(f"Rationale Model (qφ) - Latest:")
print(f"  Repository: {HF_REPO_ID}")
print(f"  Path in repo: {HF_Q_BASE_PATH}latest/")
print(f"  Full path: {HF_REPO_ID}/{HF_Q_BASE_PATH}latest/")
print()

# Summary
print("=" * 80)
print("SUMMARY:")
print("=" * 80)
print(f"✓ Repository: {HF_REPO_ID}")
print(f"✓ Version: {HF_VERSION}")
print(f"✓ HF_TOKEN: {'SET' if HF_TOKEN else 'NOT SET'}")
if not HF_TOKEN:
    print("  ⚠ WARNING: Models will NOT be uploaded to HuggingFace without HF_TOKEN")
print("=" * 80)

