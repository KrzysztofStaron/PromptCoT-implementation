# tieBreaker.py — Quality gate using Groq LLM when reward signal is ambiguous
import json
import os
import logging
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
    Evaluate a single rationale using Groq LLM via OpenRouter.
    
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
    
    prompt = f"""Evaluate this mathematical rationale for quality. Consider:
1. Does it correctly use the given concepts: {concepts_str}?
2. Is it logically consistent with the problem?
3. Is it non-trivial (not empty or generic)?
4. Does it demonstrate proper mathematical reasoning?

Problem: {problem}

Rationale: {rationale}

Respond with ONLY a JSON object with this exact format:
{{
    "score": <float between 0.0 and 1.0>,
    "reason": "<brief explanation>"
}}

The score should be:
- 1.0: Excellent — uses concepts correctly, logical, non-trivial
- 0.7-0.9: Good — mostly correct with minor issues
- 0.4-0.6: Acceptable — some issues but usable
- 0.1-0.3: Poor — significant problems
- 0.0: Invalid — empty, contradictory, or completely wrong
"""
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/PromptCoT",  # Optional: for tracking
        "X-Title": "PromptCoT Tiebreaker"  # Optional: for tracking
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,  # Low temperature for consistent evaluation
        "max_tokens": 200
    }
    
    try:
        response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        
        # Extract JSON from response
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()
        else:
            # Find JSON object
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
            else:
                log.warning(f"Could not extract JSON from Groq response: {content[:100]}")
                return None
        
        evaluation = json.loads(json_str)
        score = float(evaluation.get("score", 0.0))
        
        # Clamp score to [0, 1]
        score = max(0.0, min(1.0, score))
        
        log.debug(f"Groq evaluation: score={score:.2f}, reason={evaluation.get('reason', 'N/A')}")
        return score
        
    except requests.exceptions.RequestException as e:
        log.error(f"OpenRouter API error: {e}")
        return None
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse Groq response as JSON: {e}")
        return None
    except Exception as e:
        log.error(f"Unexpected error in Groq evaluation: {e}")
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
    
    # Log all scores for debugging
    if log.isEnabledFor(logging.DEBUG):
        log.debug(f"[TIEBREAKER] All scores: {scores}")
    
    return best_idx


def select_best_rationale(
    concepts: List[str],
    problem: str,
    rationales: List[str],
    rewards: List[float],
    use_groq: bool = True
) -> Tuple[int, str]:
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
    
    Returns:
        Tuple of (best_index, best_rationale)
    """
    if len(rationales) != len(rewards):
        raise ValueError(f"rationales ({len(rationales)}) and rewards ({len(rewards)}) must have same length")
    
    if not rationales:
        raise ValueError("rationales list cannot be empty")
    
    if len(rationales) == 1:
        return 0, rationales[0]
    
    # Check if reward signal is ambiguous
    if use_groq and use_tiebreaker(rewards, REWARD_THRESHOLD):
        log.info(f"[SELECT] Reward signal ambiguous (spread={max(rewards)-min(rewards):.3f}), using Groq tiebreaker")
        best_idx = break_tie_with_groq(concepts, problem, rationales, rewards)
        return best_idx, rationales[best_idx]
    else:
        # Use reward-based selection
        best_idx = rewards.index(max(rewards))
        log.debug(f"[SELECT] Using reward-based selection: idx={best_idx}, reward={rewards[best_idx]:.3f}")
        return best_idx, rationales[best_idx]

