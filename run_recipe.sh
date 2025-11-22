#!/bin/bash
set -e

# PromptCoT Grok-4.1 Recipe Orchestration

echo "==================================================="
echo "   PromptCoT Training Pipeline (H100 Optimized)    "
echo "==================================================="

# 1. Setup Environment
echo "[1/4] Setting up environment..."
bash setup_cloud.sh

# 2. Phase 0: Joint Pre-training
echo "[2/4] Starting Phase 0: Joint Supervised Pre-training..."
python3 train_phase0_joint.py

# 3. Phase 1: Split Warm-Start
echo "[3/4] Starting Phase 1: Split Warm-Start..."
python3 train_phase1_split.py

# 4. Phase 2: EM Loop
echo "[4/4] Starting Phase 2: EM Loop with Structure Enforcement..."
python3 train_phase2_em.py

echo "==================================================="
echo "          Training Pipeline Completed!             "
echo "==================================================="

