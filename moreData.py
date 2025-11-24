from datasets import load_dataset
import json
from openrouter import generateWithLMM


def structureprompt() -> str:
    return open("fixerPrompt.md", "r", encoding="utf-8").read()

def thinking_process_generation_prompt(problem, concepts, difficulty_level):
    concept_text = "\n".join(f"{i+1}. {concept}" for i, concept in enumerate(concepts))
    prompt = (
        "Imagine you are an expert in educational problem design.\n"
        f"You will be shown these components:\n\n"
        f"Problem: {problem}\n\n"
        f"Foundamental Concepts:\n{concept_text}\n\n"
        f"Difficulty Level: {difficulty_level}\n\n"
        "Your task is to reverse-engineer a clear thinking process that shows how a teacher might design this problem. This thinking process should:\n"
        "- Show how combining the given foundational concepts naturally leads to a problem at the specified difficulty level\n"
        "- Include all key decisions and reasoning that shaped the problem design\n"
        "- (IMPORTANT) The thinking process must be so precise and detailed that another teacher following these exact steps would recreate the identical problem\n"
        "- (IMPORTANT) The thinking process must be so natural and logical that another teacher could derive the same thinking process using only the foundational concepts and difficulty level\n\n"
        "Present your answer after 'Thinking Process: ' with the complete step-by-step thinking process described above."
    )

    return prompt

def download_data(num_examples: int = 20000) -> list[dict]:
    dataset = load_dataset("xl-zhao/PromptCoT-Problem-Generation-Dataset", split=f"train[:{num_examples}]")

    def parse_entry(entry) -> dict:
        text = entry["completion"]
        prompt = entry["prompt"]

        concepts_start = prompt.find("Foundational Concepts:")
        concepts_end = prompt.find("Difficulty Level:")

        if concepts_start != -1 and concepts_end != -1:
            concepts = prompt[concepts_start + len("Foundational Concepts:"):concepts_end].strip()
        else:
            concepts = ""

        # Extract rationale
        rationale_start = text.find("<!-- BEGIN RATIONALE -->")
        rationale_end = text.find("<!-- END RATIONALE -->")
        if rationale_start != -1 and rationale_end != -1:
            rationale = text[rationale_start + len("<!-- BEGIN RATIONALE -->"):rationale_end].strip()
        else:
            rationale = ""

        # Extract problem
        problem_start = text.find("<!-- BEGIN PROBLEM -->")
        problem_end = text.find("<!-- END PROBLEM -->")
        if problem_start != -1 and problem_end != -1:
            problem = text[problem_start + len("<!-- BEGIN PROBLEM -->"):problem_end].strip()
        else:
            problem = ""

        return {"concepts": concepts, "rationale": rationale, "problem": problem}

    parsed_data = []
    for entry in dataset:
        parsed_data.append(parse_entry(entry))
    return parsed_data

def fix_data(triplet: dict) -> dict:
    base_prompt = """

Given those instructions, fix the rationale and problem to be a valid PromptCoT 2.0 triple.
<-- BEGIN CONCEPTS -->: {} <-- END CONCEPTS -->
<-- BEGIN RATIONALE -->: {} <-- END RATIONALE -->
<-- BEGIN PROBLEM -->: {} <-- END PROBLEM -->

Response format
```json
{{
    "rationale": "The rationale for the problem",
    "problem": "The problem"
}}
```
"""
    prompt = structureprompt() + base_prompt.format(triplet["concepts"], triplet["rationale"], triplet["problem"])


    fixed_rationale = generateWithLMM(prompt, "openai/gpt-5", json_output=True)

    return {"concepts": triplet["concepts"], "rationale": fixed_rationale["rationale"], "problem": fixed_rationale["problem"]}

def main():
    dataset = download_data()

    testFixed = fix_data(dataset[0])

    json.dump(testFixed, open("moreData.json", "w"))

if __name__ == "__main__":
    main()
