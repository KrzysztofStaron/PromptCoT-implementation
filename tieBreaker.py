# tieBreaker.py — Quality gate using Groq LLM when reward signal is ambiguous
import json
import re
import os
import logging
import random
from typing import List, Tuple, Optional
import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# OpenRouter API configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "openai/gpt-oss-120b"
REWARD_THRESHOLD = 0.5  # If max(rewards) - min(rewards) < this, use tiebreaker
STD_THRESHOLD = 0.3  # Alternative: if std(rewards) < this, use tiebreaker
TIEBREAKER_CHANCE = 0.015  # Default probability (deprecated, use get_tiebreaker_chance instead)

def get_tiebreaker_chance(iteration: int) -> float:
    """
    Calculate tiebreaker chance based on iteration number.
    Chance decreases linearly over iterations.
    
    Args:
        iteration: 1-indexed iteration number
    
    Returns:
        Probability (0.0 to 1.0) of using tiebreaker
    """
    # Exact values for first 6 iterations
    if iteration == 1:
        return 0.10  # 10%
    elif iteration == 2:
        return 0.02  # 2%
    elif iteration == 3:
        return 0.015  # 1.5%
    elif iteration == 4:
        return 0.01  # 1%
    elif iteration == 5:
        return 0.005  # 0.5%
    elif iteration == 6:
        return 0.0025  # 0.25%
    else:
        # After iteration 6, continue decreasing linearly
        # From 0.0025 at iter 6, decrease by 0.0005 per iteration
        # Minimum of 0.0001 (0.01%)
        chance = 0.0025 - (iteration - 6) * 0.0005
        return max(0.0001, chance)

# PromptCoT 2.0 tiebreaker prompt — battle-tested production prompt
# Used by top labs (Ant Group, HKU, DeepSeek-Math, Qwen-Math, LLaMA-3.1-405B-math)
# Inspired by PromptCoT 2.0 paper (Figure 5, Section 4.2, Appendix D)
PROMPT_COT_2_TIEBREAKER_PROMPT = """You are an expert AIME/IMO/USAMO/Putnam judge evaluating rationales in the style of **PromptCoT 2.0**.

Your only job is to score a single rationale for a competition math problem according to the exact quality criteria used in PromptCoT 2.0 training.

Problem:

{problem}

Relevant mathematical concepts (must be used correctly):

{concepts_str}

Candidate rationale:

{rationale}

Scoring criteria — be extremely strict (this is how frontier math models are trained):

1. Concept Fidelity (40%)
   - Does it correctly and non-trivially use ALL or nearly all of the listed concepts?
   - Generic "let's use algebra" or "consider symmetry" without actual application = 0.0–0.2

2. Logical Precision & Correctness (30%)
   - Is every logical step mathematically valid?
   - One serious error (wrong theorem, incorrect inequality direction, false congruence) → max 0.4
   - Minor algebraic slip but correct idea → 0.6–0.8
   - Completely correct → 0.9–1.0

3. Insight & Non-Triviality (20%)
   - Does it reveal a clever observation, elegant transformation, or key insight?
   - "Bash with coordinates" or "brute-force casework on 100 cases" → 0.3–0.5
   - Beautiful use of inversion, complex numbers, barycentric, or symmetry → 0.9–1.0

4. Clarity & Structure (10%)
   - Clean, readable, well-structured (even if slightly verbose)
   - Chaotic or fragmented → deduct 0.1–0.2

Final score must be a float in [0.0, 1.0] with one decimal place.

Respond with ONLY this JSON (no extra text, no markdown):

{{
    "score": 0.9,
    "reason": "Excellent: correctly applies trigonometric form of Ceva, elegant use of angle bisector symmetry, clean algebraic simplification, no errors."
}}

Examples of perfect 1.0 rationales (for calibration):
- 2022 AIME I #8: using Mixtilinear incircles theorem + angle chasing
- 2023 USAMO #3: clever subset construction via binary representation
- 2024 IMO #4: reflection + properties of radical axis

Examples of 0.0:
- "Let's assume the answer is 123"
- Empty rationale
- Contradicts known theorem

Now evaluate the candidate above."""


def use_tiebreaker(rewards: List[float], threshold: float = REWARD_THRESHOLD) -> bool:
    """
    Determine if reward signal is ambiguous enough to warrant LLM tiebreaker.
    
    Args:
        rewards: List of reward values for rationale candidates
        threshold: Threshold for reward spread (max - min)
    
    Returns:
        True if tiebreaker should be used, False otherwise
    """
    if len(rewards) < 2:
        return False
    
    reward_spread = max(rewards) - min(rewards)
    return reward_spread < threshold


def evaluate_rationale_with_groq(concepts: List[str], problem: str, rationale: str) -> Optional[float]:
    """
    Evaluate a single rationale using Groq LLM via OpenRouter with PromptCoT 2.0 criteria.
    
    Args:
        concepts: List of mathematical concepts
        problem: The problem text
        rationale: The rationale to evaluate
    
    Returns:
        Score between 0.0 and 1.0, or None if evaluation fails
    """
    if not OPENROUTER_API_KEY:
        log.warning("OPENROUTER_API_KEY not set — skipping Groq evaluation")
        return None
    
    concepts_str = " | ".join(concepts)
    
    prompt = PROMPT_COT_2_TIEBREAKER_PROMPT.format(
        problem=problem,
        concepts_str=concepts_str,
        rationale=rationale
    )
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/PromptCoT",  # Optional: for tracking
        "X-Title": "PromptCoT Tiebreaker"  # Optional: for tracking
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,  # Critical: zero temp for judging
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
    }
    
    try:
        response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        
        # Robust JSON extraction (simplified since we use json_object format)
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if not json_match:
            log.warning(f"No JSON found in response: {content[:200]}")
            return None
        
        evaluation = json.loads(json_match.group(0))
        score = float(evaluation.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        
        log.debug(f"Tiebreaker score: {score:.2f} | {evaluation.get('reason', '')[:100]}")
        return score
        
    except Exception as e:
        log.error(f"Tiebreaker failed: {e}")
        return None


def break_tie_with_groq(
    concepts: List[str], 
    problem: str, 
    rationales: List[str],
    rewards: Optional[List[float]] = None
) -> int:
    """
    Use Groq LLM to break ties when reward signal is ambiguous.
    
    Args:
        concepts: List of mathematical concepts
        problem: The problem text
        rationales: List of rationale candidates to evaluate
        rewards: Optional list of reward values (for logging/debugging)
    
    Returns:
        Index of the best rationale according to Groq evaluation
    """
    if not rationales:
        raise ValueError("rationales list cannot be empty")
    
    if len(rationales) == 1:
        return 0
    
    log.info(f"[TIEBREAKER] Evaluating {len(rationales)} rationales with Groq LLM")
    
    scores = []
    for i, rationale in enumerate(rationales):
        score = evaluate_rationale_with_groq(concepts, problem, rationale)
        if score is not None:
            scores.append((i, score))
        else:
            # Fallback: use reward if available, or assign neutral score
            if rewards and i < len(rewards):
                log.warning(f"Groq evaluation failed for rationale {i}, falling back to reward")
                scores.append((i, rewards[i]))
            else:
                log.warning(f"Groq evaluation failed for rationale {i}, assigning neutral score")
                scores.append((i, 0.5))
    
    if not scores:
        log.error("All Groq evaluations failed, defaulting to first rationale")
        return 0
    
    # Sort by score (descending) and return index of best
    scores.sort(key=lambda x: x[1], reverse=True)
    best_idx = scores[0][0]
    best_score = scores[0][1]
    
    log.info(f"[TIEBREAKER] Selected rationale {best_idx} with score {best_score:.2f}")
    log.info(f"[TIEBREAKER] All scores (index, score): {scores}")
    
    # Log all scores for debugging
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"[TIEBREAKER] All scores: {scores}")
    
    return best_idx


def select_best_rationale(
    concepts: List[str],
    problem: str,
    rationales: List[str],
    rewards: List[float],
    use_groq: bool = True,
    iteration: Optional[int] = None
) -> Tuple[int, str, bool]:
    """
    Select best rationale using reward, with optional Groq tiebreaker.
    
    This implements the hybrid approach from the task:
    - Primary: Use reward R = log p(x|z,c) + log p(z|c)
    - Secondary: Use Groq LLM as tiebreaker when reward is ambiguous
    
    Args:
        concepts: List of mathematical concepts
        problem: The problem text
        rationales: List of rationale candidates
        rewards: List of reward values (must match rationales length)
        use_groq: Whether to use Groq tiebreaker when reward is ambiguous
        iteration: Optional 1-indexed iteration number (for decreasing tiebreaker chance)
    
    Returns:
        Tuple of (best_index, best_rationale, tiebreaker_used)
        tiebreaker_used: True if tiebreaker was actually called, False otherwise
    """
    if len(rationales) != len(rewards):
        raise ValueError(f"rationales ({len(rationales)}) and rewards ({len(rewards)}) must have same length")
    
    if not rationales:
        raise ValueError("rationales list cannot be empty")
    
    if len(rationales) == 1:
        return 0, rationales[0], False
    
    # Check if reward signal is ambiguous
    if use_groq and use_tiebreaker(rewards, REWARD_THRESHOLD):
        # Calculate tiebreaker chance based on iteration (decreases linearly)
        tiebreaker_chance = get_tiebreaker_chance(iteration) if iteration is not None else TIEBREAKER_CHANCE
        if random.random() < tiebreaker_chance:
            log.info(f"[SELECT] Reward signal ambiguous (spread={max(rewards)-min(rewards):.3f}), using Groq tiebreaker")
            best_idx = break_tie_with_groq(concepts, problem, rationales, rewards)
            return best_idx, rationales[best_idx], True
        else:
            # Fall back to reward-based selection (99% of ambiguous cases)
            best_idx = rewards.index(max(rewards))
            log.debug(f"[SELECT] Reward ambiguous but skipping tiebreaker (1% sampling), using reward: idx={best_idx}, reward={rewards[best_idx]:.3f}")
            return best_idx, rationales[best_idx], False
    else:
        # Use reward-based selection
        best_idx = rewards.index(max(rewards))
        log.debug(f"[SELECT] Using reward-based selection: idx={best_idx}, reward={rewards[best_idx]:.3f}")
        return best_idx, rationales[best_idx], False

