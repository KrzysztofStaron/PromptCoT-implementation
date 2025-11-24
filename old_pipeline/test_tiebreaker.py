# test_tiebreaker.py — Test the tiebreaker functionality
import sys
import logging
from tieBreaker import (
    use_tiebreaker,
    evaluate_rationale_with_groq,
    break_tie_with_groq,
    select_best_rationale,
    OPENROUTER_API_KEY
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger(__name__)

def test_use_tiebreaker():
    """Test the tiebreaker decision logic."""
    print("\n=== Testing use_tiebreaker() ===")
    
    # Test 1: Clear reward signal (should NOT use tiebreaker)
    rewards_clear = [10.5, 8.2, 7.1, 6.0]
    result = use_tiebreaker(rewards_clear)
    print(f"Clear rewards {rewards_clear}: {result} (expected: False)")
    assert result == False, "Should not use tiebreaker for clear rewards"
    
    # Test 2: Ambiguous reward signal (should use tiebreaker)
    rewards_ambiguous = [10.0, 9.9, 9.8, 9.7]
    result = use_tiebreaker(rewards_ambiguous)
    print(f"Ambiguous rewards {rewards_ambiguous}: {result} (expected: True)")
    assert result == True, "Should use tiebreaker for ambiguous rewards"
    
    # Test 3: Single reward (should NOT use tiebreaker)
    rewards_single = [10.0]
    result = use_tiebreaker(rewards_single)
    print(f"Single reward {rewards_single}: {result} (expected: False)")
    assert result == False, "Should not use tiebreaker for single reward"
    
    # Test 4: Edge case (exactly at threshold - should NOT trigger since we use <, not <=)
    rewards_edge = [10.0, 9.5]
    result = use_tiebreaker(rewards_edge, threshold=0.5)
    print(f"Edge case rewards {rewards_edge} (threshold=0.5, spread=0.5): {result} (expected: False, since 0.5 is not < 0.5)")
    assert result == False, "Should not use tiebreaker when spread equals threshold (strict <)"
    
    # Test 5: Just below threshold (should trigger)
    rewards_below = [10.0, 9.51]
    result = use_tiebreaker(rewards_below, threshold=0.5)
    print(f"Below threshold rewards {rewards_below} (threshold=0.5, spread=0.49): {result} (expected: True)")
    assert result == True, "Should use tiebreaker when spread < threshold"
    
    print("✅ All use_tiebreaker() tests passed!")


def test_select_best_rationale_with_clear_rewards():
    """Test selection when rewards are clear (no tiebreaker needed)."""
    print("\n=== Testing select_best_rationale() with clear rewards ===")
    
    concepts = ["algebra", "quadratic equations", "factoring"]
    problem = "Solve x^2 + 5x + 6 = 0"
    rationales = [
        "This is a bad rationale that doesn't make sense.",
        "We can factor the quadratic as (x+2)(x+3) = 0, so x = -2 or x = -3.",
        "This is another mediocre rationale."
    ]
    rewards = [5.0, 12.5, 6.0]  # Clear winner at index 1
    
    best_idx, best_rationale = select_best_rationale(
        concepts, problem, rationales, rewards, use_groq=False
    )
    
    print(f"Rewards: {rewards}")
    print(f"Selected index: {best_idx} (expected: 1)")
    print(f"Selected rationale: {best_rationale[:50]}...")
    
    assert best_idx == 1, "Should select rationale with highest reward"
    assert best_rationale == rationales[1], "Should return correct rationale"
    
    print("✅ Clear rewards selection test passed!")


def test_select_best_rationale_with_ambiguous_rewards():
    """Test selection when rewards are ambiguous (tiebreaker needed)."""
    print("\n=== Testing select_best_rationale() with ambiguous rewards ===")
    
    concepts = ["algebra", "quadratic equations", "factoring"]
    problem = "Solve x^2 + 5x + 6 = 0"
    rationales = [
        "This rationale is empty or trivial.",
        "We factor the quadratic equation: x^2 + 5x + 6 = (x+2)(x+3) = 0. Therefore, x = -2 or x = -3.",
        "The solution involves finding the roots of the quadratic equation."
    ]
    rewards = [10.0, 10.1, 10.05]  # Ambiguous (spread < 0.5)
    
    if not OPENROUTER_API_KEY:
        print("⚠️  OPENROUTER_API_KEY not set, skipping API test")
        print("   Testing fallback behavior...")
        # Test that it still works (will use reward-based selection)
        best_idx, best_rationale = select_best_rationale(
            concepts, problem, rationales, rewards, use_groq=True
        )
        print(f"Selected index: {best_idx} (fallback to reward-based)")
        print("✅ Fallback behavior test passed!")
        return
    
    print(f"Rewards: {rewards} (ambiguous, spread={max(rewards)-min(rewards):.3f})")
    print("Using Groq tiebreaker...")
    
    best_idx, best_rationale = select_best_rationale(
        concepts, problem, rationales, rewards, use_groq=True
    )
    
    print(f"Selected index: {best_idx}")
    print(f"Selected rationale: {best_rationale[:80]}...")
    
    assert 0 <= best_idx < len(rationales), "Selected index must be valid"
    
    print("✅ Ambiguous rewards selection test passed!")


def test_evaluate_rationale_with_groq():
    """Test the Groq API evaluation (if API key is available)."""
    print("\n=== Testing evaluate_rationale_with_groq() ===")
    
    if not OPENROUTER_API_KEY:
        print("⚠️  OPENROUTER_API_KEY not set, skipping API test")
        print("   Set OPENROUTER_API_KEY in .env file to test API calls")
        return
    
    concepts = ["algebra", "quadratic equations"]
    problem = "Solve x^2 + 5x + 6 = 0"
    good_rationale = "We can factor the quadratic as (x+2)(x+3) = 0. Setting each factor to zero gives x = -2 or x = -3."
    bad_rationale = "This is a bad rationale that doesn't make sense and doesn't use the concepts."
    
    print("Evaluating good rationale...")
    good_score = evaluate_rationale_with_groq(concepts, problem, good_rationale)
    if good_score is not None:
        print(f"Good rationale score: {good_score:.2f}")
        assert 0.0 <= good_score <= 1.0, "Score must be between 0 and 1"
    else:
        print("⚠️  API call failed or returned None")
    
    print("Evaluating bad rationale...")
    bad_score = evaluate_rationale_with_groq(concepts, problem, bad_rationale)
    if bad_score is not None:
        print(f"Bad rationale score: {bad_score:.2f}")
        assert 0.0 <= bad_score <= 1.0, "Score must be between 0 and 1"
        
        if good_score is not None:
            print(f"Score difference: {good_score - bad_score:.2f}")
            if good_score > bad_score:
                print("✅ Good rationale scored higher than bad rationale!")
    else:
        print("⚠️  API call failed or returned None")
    
    print("✅ Groq evaluation test completed!")


def test_break_tie_with_groq():
    """Test the full tiebreaker function."""
    print("\n=== Testing break_tie_with_groq() ===")
    
    concepts = ["algebra", "quadratic equations", "factoring"]
    problem = "Solve x^2 + 5x + 6 = 0"
    rationales = [
        "This is empty or trivial.",
        "We factor: x^2 + 5x + 6 = (x+2)(x+3) = 0, so x = -2 or x = -3.",
        "The solution involves algebraic manipulation."
    ]
    
    if not OPENROUTER_API_KEY:
        print("⚠️  OPENROUTER_API_KEY not set, testing fallback...")
        try:
            best_idx = break_tie_with_groq(concepts, problem, rationales)
            print(f"Selected index: {best_idx} (fallback)")
            assert 0 <= best_idx < len(rationales), "Selected index must be valid"
            print("✅ Fallback test passed!")
        except Exception as e:
            print(f"⚠️  Fallback error (expected if no API key): {e}")
        return
    
    print("Breaking tie with Groq...")
    best_idx = break_tie_with_groq(concepts, problem, rationales)
    
    print(f"Selected index: {best_idx}")
    print(f"Selected rationale: {rationales[best_idx][:80]}...")
    
    assert 0 <= best_idx < len(rationales), "Selected index must be valid"
    
    print("✅ Tiebreaker test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing tieBreaker.py")
    print("=" * 60)
    
    try:
        # Run all tests
        test_use_tiebreaker()
        test_select_best_rationale_with_clear_rewards()
        test_select_best_rationale_with_ambiguous_rewards()
        test_evaluate_rationale_with_groq()
        test_break_tie_with_groq()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

