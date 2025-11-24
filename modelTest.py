from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "xl-zhao/PromptCoT-Problem-Generation-Model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

foundational_concepts = [
    "Ability to apply quantitative reasoning and estimation techniques to solve problems, including making approximations and using logical deductions to arrive at a solution.",
    "Ability to solve equations involving complex numbers, including finding conditions under which two complex numbers are equal, particularly in the context of their magnitudes and arguments.",
    "Fractional arithmetic: Performing calculations with fractions to determine the final probability.",
    "Interpreting and solving problems involving nested operations or functions.",
    "Using logical reasoning to connect given data points and derive conclusions."
]

difficulty_level = "HMMT-Feb"

prompt = (
    "Given foundational concepts and difficulty level, identify connections and develop a question "
    "that integrates these concepts with appropriate complexity.\n\n"
    "Foundational Concepts:\n"
    + "\n".join(f"{i+1}. {concept}" for i, concept in enumerate(foundational_concepts))
    + f"\n\nDifficulty Level: {difficulty_level}"
)

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    output = model.generate(**inputs, max_length=4096, temperature=0.6)

generated_problem = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_problem)
