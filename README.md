# PromptCoT: Synthetic Dataset Generation for Reasoning Models

A comprehensive approach to generating high-quality synthetic datasets for mathematical and coding reasoning models.

## Overview

This project implements a systematic pipeline for creating Olympiad-level mathematical problems and training reasoning models through a multi-stage process involving concept-guided problem synthesis, rationale generation, and iterative refinement.

Based on the [PromptCoT paper](https://arxiv.org/pdf/2509.19894) - a method for generating synthetic datasets that improve mathematical reasoning through concept-guided problem synthesis.

## Nomenclature

- **c** - Concepts: Mathematical techniques, theorems, or problem-solving strategies
- **z** - Rationale: Step-by-step thinking plan
- **x** - Problem: The actual mathematical problem statement

## Architecture

### 1. Seed Data: (c, z, x) Triples

**Purpose**: Create a foundational dataset for generating Olympiad-level math questions

**Generation**: Uses the [PromptCoT-Problem-Generation-Dataset](https://huggingface.co/datasets/xl-zhao/PromptCoT-Problem-Generation-Dataset) which contains high-quality seed triples from AIME problems with extracted concepts and generated rationales. This provides the initial training data to kickstart the EM loop.

**Structure**:

```json
{
  "concepts": ["exponents", "modular arithmetic"],
  "rationale": "Use lifting-the-exponent lemma on x^n + 1...",
  "problem": "Find the smallest odd prime factor of 2019^8 + 1."
}
```

Note: 100–1,000 high-quality triples are sufficient to kickstart the EM loop. These seed triples compound in quality, so they must be excellent.

### 2. Rationale Model: qφ(z|c,x)

**Training**: Fine-tune a model on seed triples (c, z, x)

**Function**: Predict optimal thinking plan

- **Input**: (concepts, problem)
- **Output**: rationale

**Model**: LoRA fine-tuned `unsloth/DeepSeek-R1-Distill-Qwen-7B` with r=64, alpha=64, targeting attention projection matrices (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj).

**Format**: `[CONCEPTS]\n{concepts}\n[/CONCEPTS]\n\n[PROBLEM]\n{problem}\n[/PROBLEM]\n\n[RATIONALE]\n{rationale}\n[/RATIONALE]`

**Weights**: [PromptCoT-Rationale Model](https://huggingface.co/PanzerBread/promptCoT-rationale)

### 3. Prompt Generator: pθ(x|z,c)

**Training**: Fine-tune a model on rationale and problem pairs

**Function**: Generate challenging problems

- **Input**: (concepts, rationale)
- **Output**: problem

**Model**: LoRA fine-tuned `unsloth/DeepSeek-R1-Distill-Qwen-7B` with r=64, alpha=64, targeting attention projection matrices (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj).

**Format**: `[CONCEPTS]\n{concepts}\n[/CONCEPTS]\n\n[RATIONALE]\n{rationale}\n[/RATIONALE]\n\n[PROBLEM]\n{problem}\n[/PROBLEM]`

**Weights**: [PromptCoT-Prompt Model](https://huggingface.co/PanzerBread/promptCoT-prompt)

### 4. EM Loop with Reward-Based Selection

**E-step**:

- Generate k rationale candidates per problem using qφ (via vLLM for fast inference)
- Calculate rewards using pθ, select best rationale

**M-step**:

- Generate deterministic rationales using updated qφ
- Train pθ(x|z,c) on new (c, z_det, x) triples

**Reward Function**:

```
loss_x = NLL(pθ(x | z, c))  # Problem given concepts and rationale
loss_z = NLL(pθ(z | c))     # Rationale given concepts
reward = -(loss_x + loss_z)  # Higher reward = better quality
```

**Implementation**: Uses vLLM for fast batched inference during E-step generation, Unsloth for efficient training and reward computation.

### 5. Post-Training: Self-Play or SFT

#### A. Self-Play Approach

**Goal**: Push state-of-the-art performance for strong models

**Method**: Run a strong model through PPO/GRPO loop

- Solved synthetic problem → +1 reward
- Unsolved problem → 0 reward

#### B. Supervised Fine-Tuning (SFT)

**Goal**: Improve weaker models through knowledge distillation

**Method**: Weaker models learn from teacher-distilled traces

- Smaller models learn rationales from stronger models
- Focus on understanding problem-solving strategies

## Implementation Status

✅ **Completed:**

- Seed data generation
- LoRA fine-tuning infrastructure
- Rationale model (qφ) deployed to HuggingFace
- Prompt model (pθ) deployed to HuggingFace
- EM loop with reward-based selection

⏳ **Planned:**

- Self-play / SFT implementation

## Training Pipeline

The training process is divided into three phases:

### Phase 0: Cold-Start Training

Train both models separately on the seed dataset to establish initial capabilities.

#### Train Prompt Generator (pθ)

```bash
python train_phase0_p.py
```

This script:

- Loads `unsloth/DeepSeek-R1-Distill-Qwen-7B` base model (4-bit QLoRA)
- Trains pθ model on format: `[CONCEPTS]\n[RATIONALE]\n[PROBLEM]`
- Fine-tunes with LoRA (r=64, alpha=64) on attention projection matrices
- Saves to `./models/{HF_VERSION}/p/cold-start`
- Uploads to HuggingFace Hub (if `HF_TOKEN` is set)

#### Train Rationale Model (qφ)

```bash
python train_phase0_q.py
```

This script:

- Loads `unsloth/DeepSeek-R1-Distill-Qwen-7B` base model (4-bit QLoRA)
- Trains qφ model on format: `[CONCEPTS]\n[PROBLEM]\n[RATIONALE]`
- Fine-tunes with LoRA (r=64, alpha=64) on attention projection matrices
- Saves to `./models/{HF_VERSION}/q/cold-start`
- Uploads to HuggingFace Hub (if `HF_TOKEN` is set)

### Phase 2: EM Loop Training

Run the Expectation-Maximization loop to iteratively improve synthetic dataset generation:

```bash
python train_phase2_em.py [--k K] [--no-upload]
```

**Arguments:**

- `--k`: Number of rationale candidates to generate per prompt (default: 5)
- `--no-upload`: Disable uploading to HuggingFace

**Process:**

1. **E-step**:

   - Generate k rationale candidates per problem using qφ (via vLLM for fast inference)
   - Compute rewards using pθ: `reward = -(loss_x + loss_z)` where:
     - `loss_x` = NLL of problem given concepts and rationale
     - `loss_z` = NLL of rationale given concepts
   - Select best rationale based on reward
   - Update qφ on selected triples

2. **M-step**:

   - Generate deterministic rationales using updated qφ
   - Update pθ on new (concepts, rationale, problem) triples

3. **Iteration Management**:
   - Automatically resumes from latest checkpoint
   - Uploads checkpoints to HuggingFace after each iteration
   - Cleans up old iterations (keeps last 3)

**Configuration:**

- Set `HF_VERSION` in `hf_config.py` to version your models
- Number of triples scales inversely with k to keep computation constant
- Default: 2 EM iterations, configurable via `EM_ITERS` in script

## Configuration

**HuggingFace Setup:**

- Edit `hf_config.py` to set `HF_VERSION` and `HF_USERNAME`
- Set `HF_TOKEN` in `.env` file for automatic model uploads

**Requirements**: See `requirements.txt`

**Hardware**: Recommended H200 GPU (141GB VRAM) or similar high-memory GPU for Phase 2

OK NOTES FOR ME:

fixando.md consists of a fixer prompt for a highly-capable model to improve the model, by ensuring x,c, z are deeply connected, making EM loop more effective

what we doing now?

- Reinforcement learning of a small Quen model to improve it's coding capabilities
