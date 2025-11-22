# hf_config.py
# Shared HuggingFace configuration for PromptCoT models

# HuggingFace Hub configuration
HF_USERNAME = "PanzerBread/PromptCoT"
HF_VERSION = "coding-0.2"  # Set this constant to version your models
HF_TAGS = ["math", "coding"]  # Tags to add to uploaded models

# Computed HuggingFace repo paths (base paths)
HF_REPO_ID = HF_USERNAME
HF_P_BASE_PATH = f"{HF_VERSION}/p/"
HF_Q_BASE_PATH = f"{HF_VERSION}/q/"

