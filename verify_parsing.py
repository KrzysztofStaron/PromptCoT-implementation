import re
import json

# Mock dataset row based on user example
example = {
    "prompt": "Given foundational concepts and difficulty level, identify connections and develop a question that integrates these concepts with appropriate complexity.\n\nFoundational Concepts:\n1. Ability to apply quantitative reasoning and estimation techniques to solve problems, including making approximations and using logical deductions to arrive at a solution.\n2. Ability to solve equations involving complex numbers, including finding conditions under which two complex numbers are equal, particularly in the context of their magnitudes and arguments.\n3. Fractional Arithmetic: Performing calculations with fractions to determine the final probability.\n4. Interpreting and solving problems involving nested operations or functions.\n5. Using logical reasoning to connect given data points and derive conclusions.\n\nDifficulty Level: HMMT-Feb",
    "completion": "<!-- BEGIN RATIONALE -->**Step 1: Selecting Foundational Concepts and Difficulty Level**\nThe problem designer starts by selecting the foundational concepts and the desired difficulty level. In this case, the concepts are quantitative reasoning, complex numbers, fractional arithmetic, nested operations, and logical reasoning. The difficulty level is HMMT-Feb, which is a challenging level requiring a combination of mathematical knowledge, problem-solving skills, and critical thinking.\n\n**Step 2: Identifying Key Skills and Knowledge**\nThe designer identifies key skills that students should demonstrate to solve the problem, including:\n- Quantitative reasoning and estimation techniques.\n- Understanding and solving equations involving complex numbers.\n- Fractional arithmetic for probability calculations.\n- Logical reasoning to connect given data points.\n\n**Step 3: Developing the Problem Statement**\nThe designer refines the problem by introducing a scenario where students estimate complex numbers based on sum and product conditions and use logical reasoning to determine the probability that they are distinct.\n\n**Step 4: Finalizing the Problem**\nThe final problem is:\n\n<!-- BEGIN PROBLEM -->\nLet $a$ and $b$ be complex numbers such that $|a+b|=|a-b|=1$. What is the probability that $a$ and $b$ are distinct complex numbers?\n<!-- END PROBLEM -->"
}

def parse_test(examples):
    # Simulating the logic in train_phase0_joint.py
    prompts = [examples['prompt']]
    completions = [examples['completion']]
    
    texts = []
    for p, c in zip(prompts, completions):
        # Concepts parsing
        concepts_match = re.search(r"Foundational Concepts:(.*?)Difficulty Level:", p, re.DOTALL)
        if concepts_match:
            concepts_text = concepts_match.group(1).strip()
            concepts_cleaned = re.sub(r"\d+\.\s*", "", concepts_text) 
            concepts_cleaned = " | ".join([line.strip() for line in concepts_cleaned.split('\n') if line.strip()])
        else:
            concepts_cleaned = "FAILED"

        # Rationale & Problem parsing - Robust Regex
        # Try finding END RATIONALE, if not, look for BEGIN PROBLEM
        rationale_pattern = r"<!-- BEGIN RATIONALE -->(.*?)(?:<!-- END RATIONALE -->|(?=<!-- BEGIN PROBLEM -->))"
        rationale_match = re.search(rationale_pattern, c, re.DOTALL)
        
        problem_match = re.search(r"<!-- BEGIN PROBLEM -->(.*?)<!-- END PROBLEM -->", c, re.DOTALL)
        
        rationale = rationale_match.group(1).strip() if rationale_match else "FAILED"
        problem = problem_match.group(1).strip() if problem_match else "FAILED"
        
        text = f"Concepts: {concepts_cleaned}\nRationale: {rationale}\nProblem: {problem}"
        texts.append(text)
        
        # Debug prints
        print("--- Parsed Concepts ---")
        print(concepts_cleaned[:100] + "...")
        print("\n--- Parsed Rationale (Snippet) ---")
        print(rationale[:100] + "...")
        print("\n--- Parsed Problem ---")
        print(problem)
        
    return texts

print("Running verification...")
parse_test(example)
print("\nVerification Complete.")

