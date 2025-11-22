# generate_phase1_q.py
# Prosty skrypt - wczytaj model i uruchom prompty

from unsloth import FastLanguageModel
import torch
import json
from hf_config import HF_REPO_ID, HF_VERSION

# Ścieżka do modelu Phase 0
MODEL_PATH = f"{HF_REPO_ID}/{HF_VERSION}/joint"
BASE_MODEL = "unsloth/DeepSeek-R1-Distill-Qwen-7B"

def main():
    # Wczytaj bazowy model
    print(f"Wczytywanie bazowego modelu {BASE_MODEL}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=8192,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Wczytaj adapter Phase 0
    print(f"Wczytywanie adaptera z {MODEL_PATH}...")
    model.load_adapter(MODEL_PATH, adapter_name="phase0")
    model.set_adapter("phase0")
    
    # Ustaw do generowania
    FastLanguageModel.for_inference(model)
    print("✅ Model wczytany!")

    # Prompty do testowania
    prompts = [
        "The capital of France is",
        "Write a Python function to reverse a string:",
        "Solve the equation 3x + 5 = 20 for x.",
        "Explain quantum computing like I'm 10 years old."
    ]

    # Generuj
    print(f"\nGenerowanie dla {len(prompts)} promptów...")
    
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Prompt {i}/{len(prompts)} ---")
        print(f"Input: {prompt}")
        
        # Tokenizuj
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Generuj
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Dekoduj
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Usuń prompt z outputu
        generated_text = generated[len(prompt):].strip()
        
        print(f"Output: {generated_text}")
        print("-"*60)
        
        results.append({
            "prompt": prompt,
            "output": generated_text
        })
    
    # Zapisz do pliku
    with open("generated_outputs.jsonl", "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"\n✅ Gotowe! Wyniki zapisane do generated_outputs.jsonl")


if __name__ == "__main__":
    main()

