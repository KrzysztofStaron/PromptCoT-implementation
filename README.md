# PromptCoT: Synthetic Dataset Generation for Reasoning Models

A comprehensive approach to generating high-quality synthetic datasets for mathematical and coding reasoning models.

## Overview

This project implements a systematic pipeline for creating Olympiad-level mathematical problems and training reasoning models through a multi-stage process involving concept-guided problem synthesis, rationale generation, and iterative refinement.

## Architecture

### 1. Seed Data: (c, z, x) Triples

**Purpose**: Create a foundational dataset for generating Olympiad-level math questions

**Structure**:

```json
{
  "concepts": ["exponents", "modular arithmetic"],
  "rationale": "Use lifting-the-exponent lemma on x^n + 1...",
  "problem": "Find the smallest odd prime factor of 2019^8 + 1."
}
```

### 2. Rationale Model: qφ(z|c,x)

**Training**: Fine-tune a model on seed triples (c, z, x)

**Function**: Predict optimal thinking plan

- **Input**: (concepts, problem)
- **Output**: rationale

### 3. Prompt Generator: pθ(x|z,c)

**Training**: Fine-tune a model on rationale and problem pairs

**Function**: Generate challenging problems

- **Input**: (concepts, rationale)
- **Output**: problem

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
