from vllm import LLM, SamplingParams

model_name = "xl-zhao/PromptCoT-Problem-Generation-Model"
llm = LLM(model=model_name, tensor_parallel_size=1)

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

sampling_params = SamplingParams(temperature=0.6, max_tokens=4096)
outputs = llm.generate([prompt], sampling_params)

print(outputs[0].outputs[0].text)
