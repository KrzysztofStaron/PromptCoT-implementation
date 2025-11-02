
# Collect 100+ real problems
# "Annotate (c, z, x)"
# Train qφ and pθ on seed
# "Implement reward R(c,x,z)"
# Run EM loop (E → M)
# Generate 100k+ problems
# Add verification (SymPy / pytest)
# Run self-play or SFT
# """"
# Components:
# 1. Seed triplets (c, z, x)
# 2. Rationale model qφ(z|c,x)
# 3. Prompt model pθ(x|z,c)
# 4. EM loop with reward
# 5. Self-play or SFT with verification
# """"


# This is the main coordinator file for the PromptCoT project
# Individual training scripts have been split into separate files:
# - finetune_rationale.py: Fine-tunes the rationale model qφ(z|c,x)
# - finetune_prompt.py: Fine-tunes the prompt model pθ(x|z,c)

