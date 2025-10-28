
# Collect 100+ real problems
# "Annotate (c, z, x)"
# Train qφ and pθ on seed
# "Implement reward R(c,x,z)"
# Run EM loop (E → M)
# Generate 100k+ problems
# Add verification (SymPy / pytest)
# Run self-play or SFT
# """"
# TL;DR – What You Must Build
#
# 5 things:
#
# Seed triples (c, z, x)
# Rationale model qφ(z|c,x)
# Prompt model pθ(x|z,c)
# EM loop with reward
# Self-play or SFT with verification
# """"


# 1. Seed triplets