<img src="https://github.com/inclusionAI/PromptCoT/assets/12345678/abc123.png" width="100" align="right" />

# PromptCoT 2.0: Scaling Prompt Synthesis for Large Language Model Reasoning

**Official Implementation**  
`arXiv:2509.19894` | [Paper PDF](https://arxiv.org/pdf/2509.19894) | [GitHub](https://github.com/inclusionAI/PromptCoT)

---

## TL;DR

> **PromptCoT 2.0 is a fully learnable, scalable framework that generates Olympiad-level math and competitive programming problems using an EM-optimized rationale-driven synthesis pipeline — producing harder, more diverse synthetic datasets than human-curated ones.**

It enables **two training regimes**:

- **Self-Play**: Strong models (30B+) improve autonomously via RL (PPO/GRPO) on verifiable problems.
- **SFT**: Weaker models (7B) learn from teacher-distilled reasoning traces.

**Results**:

- **+4.4 on AIME 24**, **+5.3 on HMMT**, **+35 Elo on Codeforces** (30B scale)
- **7B model reaches 73.1 on AIME 24** using **only synthetic data**

---

## Why PromptCoT 2.0?

| Problem                                        | Solution                                                            |
| ---------------------------------------------- | ------------------------------------------------------------------- |
| Human data is **expensive & limited**          | **Infinite**, **hard**, **verifiable** synthetic problems           |
| Existing synthetic data is **too easy/narrow** | **Rationale-guided**, **EM-refined**, **distributionally distinct** |
| Hand-crafted prompts don’t scale               | **Fully learnable**, **domain-agnostic**                            |

> _“Prompt synthesis is a new axis for scaling reasoning.”_ — **PromptCoT 2.0**

---

## Architecture Overview

```mermaid
graph TD
    A[Seed Triples (c, z, x)] --> B[Train qφ(z|c,x) & pθ(x|z,c)]
    B --> C[EM Loop: E-step → M-step]
    C --> D[Generate 100k+ (c, z, x)]
    D --> E{Post-Training}
    E -->|Self-Play| F[Strong LLM + PPO + SymPy/pytest]
    E -->|SFT| G[Teacher → Full Trace → Student]
    F --> H[SOTA 30B Model]
    G --> I[73.1 AIME 7B Model]
```
