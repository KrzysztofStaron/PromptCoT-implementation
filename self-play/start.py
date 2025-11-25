from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import threading

# 32B model
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

# Create streamer for streaming output
streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

# Generate in a separate thread
generation_kwargs = dict(
    **inputs,
    streamer=streamer,
    max_length=4096,
    temperature=0.7
)
thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

# Print tokens as they arrive
for token in streamer:
    print(token, end="", flush=True)
print()  # New line at the end
