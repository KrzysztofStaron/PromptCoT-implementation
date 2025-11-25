from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "xl-zhao/PromptCoT-2.0-Prompt-Generation-Model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

concept_text = "graph traversal, recursion, dynamic programming"
level = "codeforces"

prompt = f"""Given the foundational programming concepts and specified difficulty level, identify connections among these concepts and develop an olympiad-level coding problem that integrates them with appropriate complexity.

Foundational Programming Concepts:
{concept_text}

Difficulty Level: {level}"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=4096, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
