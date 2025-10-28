import json
import openai
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup OpenAI
openai.api_key = "I'm not leaking my API key"  # Replace with your OpenAI API key

def extract_problem(prompt):
    # Extract problem text from prompt
    if "<\\uff5cUser\\uff5c>" in prompt:
        return prompt.split("<\\uff5cUser\\uff5c>")[1].split("<\\uff5cAssistant\\uff5c>")[0].strip()
    return prompt

def get_concepts_and_rationale(problem, solution):
    # Get concepts and rationale from GPT-5
    
    # Create concept examples for context
    concept_examples = "\n".join([f"- {concept}" for concept in unique_concepts[:20]])
    
f"""
Analyze this math problem and solution. Give me exactly 5 concepts and a rationale.

Problem: {problem}
Solution: {solution}

Here are examples of mathematical concepts, but feel free to use other concepts:
{concept_examples}

Return JSON format:
{{
    "concepts": ["concept1", "concept2", "concept3", "concept4", "concept5"],
    "rationale": "step by step reasoning"
}}

Guidelines:
- Choose concepts that are specific and actionable (like the examples above)
- Concepts should be mathematical techniques, theorems, or problem-solving strategies
- Make concepts concise but descriptive
- The rationale should explain the logical flow from problem to solution
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt}],
        )
        
        content = response.choices[0].message.content.strip()
        
        # Extract JSON
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        else:
            json_str = content[content.find("{"):content.rfind("}")+1]
        
        return json.loads(json_str)
        
    except Exception as e:
        print(f"Error: {e}")
        return None

# Load problems
problems = []
with open("math500.jsonl", 'r') as f:
    for line in f:
        problems.append(json.loads(line))

# Load sample concepts for context
sample_concepts = []
with open("data/mathematics_concepts.jsonl", 'r') as f:
    for i, line in enumerate(f):
        if i >= 50:  # Load first 50 concept sets
            break
        data = json.loads(line)
        sample_concepts.extend(data['concepts'])

# Get unique concepts and take a sample
unique_concepts = list(set(sample_concepts))[:100]  # Take 100 unique concepts
print(f"Loaded {len(unique_concepts)} sample concepts for context")

# Check existing results and resume from where we left off
results = []
start_index = 0

try:
    with open("annotated.jsonl", 'r') as f:
        for line in f:
            results.append(json.loads(line))
    start_index = len(results)
    print(f"Found {len(results)} existing annotations, resuming from problem {start_index + 1}")
except FileNotFoundError:
    print("No existing annotations found, starting from the beginning")

# Process remaining problems with parallelization (up to 3 concurrent)
problems_to_process = problems[start_index:500]
print(f"Processing {len(problems_to_process)} remaining problems...")

def process_single_problem(global_index, problem_data):
    problem_text = extract_problem(problem_data['prompt'])
    solution_text = problem_data['reference_solution']
    
    annotation = get_concepts_and_rationale(problem_text, solution_text)
    
    if annotation:
        triplet = {
            "concepts": annotation["concepts"],
            "rationale": annotation["rationale"],
            "problem": problem_text,
            "solution": solution_text
        }
        return global_index, triplet, True
    else:
        return global_index, None, False

# Use ThreadPoolExecutor for parallel processing
with ThreadPoolExecutor(max_workers=3) as executor:
    # Submit all tasks
    futures = []
    for i, p in enumerate(problems_to_process):
        global_index = start_index + i  # Maintain global problem numbering
        future = executor.submit(process_single_problem, global_index, p)
        futures.append(future)
    
    # Process completed tasks as they finish
    for future in as_completed(futures):
        global_index, triplet, success = future.result()
        
        if success:
            results.append(triplet)
            
            # Save after each successful request
            with open("annotated.jsonl", 'w') as f:
                for item in results:
                    f.write(json.dumps(item) + '\n')
            
            print(f"✓ Problem {global_index+1} done - Saved {len(results)} problems")
        else:
            print(f"✗ Problem {global_index+1} failed")

print(f"\n✅ Final result: {len(results)} problems processed")
print("📁 All saved to: annotated.jsonl")
