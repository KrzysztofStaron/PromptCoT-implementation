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

**Generation**: We collected 253 high-quality seed triples from AIME 2024/2025 problems, using GPT-5 to extract concepts and generate rationales. This provides the initial training data to kickstart the EM loop.

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

**Model**: LoRA fine-tuned Qwen2.5-7B-Instruct with r=64, targeting attention projection matrices.

**Weights**: [PromptCoT-Rationale Model](https://huggingface.co/PanzerBread/promptCoT-rationale)

### 3. Prompt Generator: pθ(x|z,c)

**Training**: Fine-tune a model on rationale and problem pairs

**Function**: Generate challenging problems

- **Input**: (concepts, rationale)
- **Output**: problem

**Model**: LoRA fine-tuned Qwen2.5-7B-Instruct with r=64, targeting attention projection matrices.

**Weights**: [PromptCoT-Prompt Model](https://huggingface.co/PanzerBread/promptCoT-prompt)

### 4. EM Loop with Reward-Based Selection

**E-step**: Generate 8 rationales, calculate rewards, select the best one

**M-step**: Train pθ(x|z,c) on new (c, z_best, x) triples

**Reward Function**:

```
log_p_z = -loss_rationale_model(c, z)
log_p_x = -loss_prompt_model(c + z, x)
reward = log_p_z + log_p_x
```

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

- Seed data generation (253 triples from AIME 2024/2025)
- LoRA fine-tuning infrastructure
- Rationale model (qφ) deployed to HuggingFace
- Prompt model (pθ) deployed to HuggingFace

🔄 **In Progress:**

- EM loop with reward-based selection

⏳ **Planned:**

- Self-play / SFT implementation

## Running the Cold-Start Training

The cold-start script fine-tunes both models on the seed dataset:

```bash
python cold_start.py
```

This will:

1. Load Qwen2.5-7B-Instruct base model
2. Apply LoRA adapters (r=64, lora_alpha=16)
3. Train rationale model on (concepts, problem) → rationale pairs
4. Train prompt model on (concepts, rationale) → problem pairs
5. Upload models to HuggingFace Hub (if `HF_TOKEN` is set)

**Requirements**: See `requirements.txt`

**Environment**: Set `HF_TOKEN` in `.env` file for automatic model uploads.

## Paper Citation

If you use this implementation, please cite the PromptCoT paper:

```
@article{promptcot2025,
  title={PromptCoT: Concept-Guided Synthetic Data for Mathematical Reasoning},
  author={[Authors]},
  journal={arXiv preprint arXiv:2509.19894},
  year={2025}
}
```
