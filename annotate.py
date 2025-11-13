import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import openai
import requests
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

openai.api_key = OPENAI_API_KEY

DATA_DIR = Path("data")
BASE_DIR = DATA_DIR / "base"
CONCEPT_FILE = BASE_DIR / "mathematics_concepts.jsonl"
OUTPUT_FILE = DATA_DIR / "annotated.jsonl"
HARDNESS_CACHE_FILE = DATA_DIR / "hardness_cache.json"

 #   {"name": "openai/gpt-5-mini", "provider": "openrouter", "label": "openrouter:openai/gpt-5-mini"},
 #   {"name": "openai/gpt-4o-mini", "provider": "openrouter", "label": "openrouter:openai/gpt-4o-mini"},
 #   {"name": "anthropic/claude-4.5-sonnet", "provider": "openrouter", "label": "openrouter:anthropic/claude-3.5-sonnet"},

MODEL_CONFIGS: List[Dict[str, str]] = [
    {"name": "openai/gpt-oss-120b", "provider": "openrouter", "label": "openrouter:openai/gpt-oss-120b"},
]

CHAT_TOKEN_PATTERN = re.compile(r"<[^>]+>")

INVALID_RATIONALE_PHRASES = [
    "the answer is",
    "final answer",
    "we get the answer",
    "\\boxed",
    "answer:",
    "####",
]

OLYMPIAD_SOURCE_FILES = {
    "aime2024.jsonl",
    "aime2025.jsonl",
    "hmmt_feb25.jsonl",
    "livecodebench_v5.jsonl",
    "livecodebench_v6.jsonl",
    "codeforces_div2.jsonl",
    "qwq_aime2024_test.jsonl",
    "qwq_aime2025_test.jsonl",
}

BASELINE_SOLVER_MODEL = "qwen/qwen2.5-72b-instruct"

HARDNESS_CACHE: Dict[str, bool] = {}


def load_hardness_cache() -> Dict[str, bool]:
    if not HARDNESS_CACHE_FILE.exists():
        return {}
    with open(HARDNESS_CACHE_FILE, "r", encoding="utf-8") as handle:
        data = json.load(handle)
        return {str(k): bool(v) for k, v in data.items()}


def save_hardness_cache() -> None:
    with open(HARDNESS_CACHE_FILE, "w", encoding="utf-8") as handle:
        json.dump(HARDNESS_CACHE, handle, ensure_ascii=False, indent=2)


def expand_atomic_concepts(concepts: List[str]) -> List[str]:
    if not concepts:
        return []
    expanded: List[str] = []
    for concept in concepts:
        if not concept:
            continue
        parts = [item.strip() for item in re.split(r"[,/]| and ", concept) if item.strip()]
        if len(parts) == 1:
            expanded.append(parts[0])
        else:
            expanded.extend(parts)
    return expanded


def load_concept_pool() -> List[str]:
    pool: List[str] = []
    if not CONCEPT_FILE.exists():
        return pool
    with open(CONCEPT_FILE, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            pool.extend(expand_atomic_concepts(data.get("concepts", [])))
    unique = list({concept.strip(): None for concept in pool}.keys())
    return unique


def collect_atomic_concepts() -> List[str]:
    pool: List[str] = []
    for source in sorted(OLYMPIAD_SOURCE_FILES):
        path_candidates = [
            BASE_DIR / source,
            BASE_DIR / "qwq" / source,
            DATA_DIR / source,
        ]
        for candidate in path_candidates:
            if not candidate.exists():
                continue
            with open(candidate, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    raw_prompt = payload.get("prompt") or payload.get("Problem") or ""
                    cleaned = clean_prompt(raw_prompt)
                    pool.extend(extract_concepts_from_problem(cleaned))
            break
    unique = list({concept: None for concept in pool if concept}.keys())
    return unique


def extract_concepts_from_problem(problem: str) -> List[str]:
    concepts: List[str] = []
    lowered = problem.lower()
    keywords = {
        "mod": "Modular arithmetic",
        "modulo": "Modular arithmetic",
        "remainder": "Modular arithmetic",
        "prime": "Prime factorization",
        "divisible": "Divisibility arguments",
        "geometry": "Euclidean geometry",
        "circle": "Circle geometry",
        "triangle": "Triangle geometry",
        "probability": "Combinatorial probability",
        "expected": "Expected value",
        "permutation": "Permutations and combinations",
        "combination": "Permutations and combinations",
        "binomial": "Binomial coefficients",
        "polynomial": "Polynomial factorization",
        "log": "Logarithmic transformations",
        "limit": "Series convergence",
        "sequence": "Sequence analysis",
        "sum": "Summation techniques",
        "product": "Product telescoping",
    }
    for needle, concept in keywords.items():
        if needle in lowered:
            concepts.append(concept)
    return concepts


def clean_prompt(raw_prompt: str) -> str:
    text = raw_prompt.replace("\uff5c", " ").replace("\u2581", " ")
    text = CHAT_TOKEN_PATTERN.sub("\n", text)
    lines = [segment.strip() for segment in text.splitlines() if segment.strip()]
    if lines and lines[0].lower().startswith("please reason step"):
        lines = lines[1:]
    cleaned = "\n".join(lines).strip()
    return cleaned


def resolve_solution(payload: Dict[str, str]) -> Optional[str]:
    for key in ("reference_solution", "solution", "output"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def attempt_baseline_solve(problem: str) -> str:
    if not OPENROUTER_API_KEY:
        return ""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/PromptCoT",
        "X-Title": "PromptCoT Hardness Filter",
    }
    payload = {
        "model": BASELINE_SOLVER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Attempt to solve this Olympiad math problem. Respond with the boxed final answer "
                    "if you are certain, otherwise say 'unsure'.\n\n"
                    f"Problem:\n{problem}"
                ),
            }
        ],
        "max_tokens": 64,
        "temperature": 0.1,
        "provider": {
            "sort": "throughput",
        },
    }
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"].get("content", "")
        if isinstance(content, list):
            text = "".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in content
            ).strip()
            return text
        if isinstance(content, str):
            return content.strip()
        return ""
    except Exception:
        return ""


def is_hard_problem(problem: str, save_immediately: bool = False) -> bool:
    if problem in HARDNESS_CACHE:
        return HARDNESS_CACHE[problem]
    attempt = attempt_baseline_solve(problem)
    lowered = attempt.lower()
    hard = True
    if lowered:
        if "\\boxed" in lowered or "answer" in lowered:
            hard = False
        if "unsure" in lowered or "cannot" in lowered or "fail" in lowered:
            hard = True
    HARDNESS_CACHE[problem] = hard
    if save_immediately:
        save_hardness_cache()
    return hard


def collect_problem_bank() -> List[Dict[str, str]]:
    dataset_paths: List[Path] = []
    if BASE_DIR.exists():
        for candidate in BASE_DIR.rglob("*.jsonl"):
            if candidate.name == CONCEPT_FILE.name:
                continue
            if candidate.name in OLYMPIAD_SOURCE_FILES:
                dataset_paths.append(candidate)

    bank: List[Dict[str, str]] = []
    new_cache_entries = 0
    for dataset_path in dataset_paths:
        with open(dataset_path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle):
                if not line.strip():
                    continue
                payload = json.loads(line)
                raw_prompt = payload.get("prompt") or payload.get("Problem") or payload.get("output")
                solution = resolve_solution(payload)
                if not raw_prompt:
                    continue
                problem = clean_prompt(raw_prompt)
                if not problem:
                    continue
                was_cached = problem in HARDNESS_CACHE
                if not is_hard_problem(problem):
                    continue
                if not was_cached:
                    new_cache_entries += 1
                idx = payload.get("idx")
                if idx is None:
                    idx = line_number
                source_id = f"{dataset_path.stem}:{idx}"
                bank.append(
                    {
                        "source": dataset_path.stem,
                        "source_id": source_id,
                        "problem": problem,
                        "solution": solution,
                    }
                )
    if new_cache_entries > 0:
        save_hardness_cache()
        print(f"💾 Saved {new_cache_entries} new hardness results to cache")
    return bank


def parse_model_content(content: object) -> Optional[Dict[str, object]]:
    if isinstance(content, dict):
        return content
    text = ""
    if isinstance(content, list):
        fragments: List[str] = []
        for part in content:
            if isinstance(part, dict):
                fragments.append(str(part.get("text", "")))
            else:
                fragments.append(str(part))
        text = "".join(fragments).strip()
    elif isinstance(content, str):
        text = content.strip()
    if not text:
        return None
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def is_design_rationale(text: str) -> bool:
    lowered = text.lower()
    if any(phrase in lowered for phrase in INVALID_RATIONALE_PHRASES):
        return False
    return True


def build_prompt(problem: str, concept_examples: List[str]) -> str:
    """
    PromptCoT 2.0 Cold-Start Prompt: (c) → (c, z)
    Input: problem (x), concept pool
    Output: JSON { "concepts": [...5...], "rationale": "design plan" }
    NO SOLUTION SHOWN. NO NUMBERS. PURE META-REASONING.
    """
    # Sample 15–25 atomic concepts for diversity
    sample_size = min(25, len(concept_examples))
    selected = random.sample(concept_examples, sample_size)
    bullets = "\n".join(f"- {c}" for c in selected)

    return f"""You are an **Olympiad problem architect**. Your task is to **reverse-engineer a design blueprint** (`z`) that explains **how to construct a problem like the one below**, using **exactly 5 concepts** from the list.

**DO NOT solve the problem.**  
**DO NOT compute any numbers.**  
**DO NOT mention the final answer.**  
**DO NOT restate the solution path.**

---

**Target Problem (already written):**
{problem.strip()}

---

**Concept Bank, feel free to use them, or invent your own:**
{bullets}

---

**Output JSON only** (no extra text):
```json
{{
  "concepts": ["Concept1", "Concept2", "Concept3", "Concept4", "Concept5"],
  "rationale": "Multi-step design plan, on how to construct the problem using the concepts. Example: Select..., Force..., Weave..., Ensure..., Stress-test..., Refine..."
}}

```"""


def annotate_with_model(
    problem: str,
    model_cfg: Dict[str, str],
    concept_pool: List[str],
) -> Optional[Dict[str, object]]:
    sample_size = min(20, len(concept_pool))
    concept_examples = random.sample(concept_pool, sample_size) if sample_size else []
    prompt = build_prompt(problem, concept_examples)

    try:
        if model_cfg["provider"] == "openai":
            response = openai.chat.completions.create(
                model=model_cfg["name"],
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            message = response.choices[0].message
            parsed = parse_model_content(message.content)
        else:
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/PromptCoT",
                "X-Title": "PromptCoT Annotation",
            }
            payload = {
                "model": model_cfg["name"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 2048,
                "response_format": {"type": "json_object"},
                "provider": {
                    "sort": "throughput",
                },
            }
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            message = data["choices"][0]["message"]
            parsed = parse_model_content(message.get("content"))
        if not parsed:
            return None
        concepts_raw = parsed.get("concepts")
        rationale = parsed.get("rationale")
        if not isinstance(concepts_raw, list) or not isinstance(rationale, str):
            return None
        concepts = [str(concept).strip() for concept in concepts_raw if str(concept).strip()]
        if len(concepts) < 5:
            return None
        concepts = concepts[:5]
        rationale_text = rationale.strip()
        if not is_design_rationale(rationale_text):
            return None
        return {"concepts": concepts, "rationale": rationale_text}
    except Exception as error:
        print(f"Error from {model_cfg['label']}: {error}")
        return None


def load_existing_results() -> Tuple[List[Dict[str, object]], set]:
    records: List[Dict[str, object]] = []
    completed = set()
    if not OUTPUT_FILE.exists():
        return records, completed
    with open(OUTPUT_FILE, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records.append(record)
            problem_key = record.get("source_id") or record.get("id") or record.get("problem")
            if problem_key:
                completed.add(problem_key)
    return records, completed


def persist_results(items: List[Dict[str, object]]) -> None:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def select_active_models() -> List[Dict[str, str]]:
    active: List[Dict[str, str]] = []
    for cfg in MODEL_CONFIGS:
        if cfg["provider"] == "openai" and OPENAI_API_KEY:
            active.append(cfg)
        elif cfg["provider"] == "openrouter" and OPENROUTER_API_KEY:
            active.append(cfg)
    return active


def main() -> None:
    global HARDNESS_CACHE
    HARDNESS_CACHE = load_hardness_cache()
    cache_size = len(HARDNESS_CACHE)
    if cache_size > 0:
        print(f"📦 Loaded {cache_size} cached hardness results")
    
    concept_pool = collect_atomic_concepts()
    if not concept_pool:
        print("⚠️  Concept pool is empty; continuing but prompts will omit examples.")

    problem_bank = collect_problem_bank()
    print(f"📚 Loaded {len(problem_bank)} problems from {DATA_DIR} (tokens removed)")

    active_models = select_active_models()
    if not active_models:
        print("❌ No models available. Set OPENAI_API_KEY and/or OPENROUTER_API_KEY.")
        return

    print("🧠 Active models:")
    for cfg in active_models:
        print(f"  - {cfg['label']}")

    results, completed_ids = load_existing_results()
    print(f"🗂️  Existing annotations: {len(results)}")

    random.shuffle(problem_bank)
    processed_count = len(completed_ids)
    model_index = processed_count % len(active_models)
    total_requests = 0

    for item in problem_bank:
        if item["source_id"] in completed_ids:
            continue

        model_cfg = active_models[model_index % len(active_models)]
        annotation = annotate_with_model(
            item["problem"],
            model_cfg,
            concept_pool,
        )
        total_requests += 1

        if annotation:
            record = {
                "source": item["source"],
                "source_id": item["source_id"],
                "problem": item["problem"],
                "solution": item["solution"],
                "concepts": annotation["concepts"],
                "rationale": annotation["rationale"],
                "model": model_cfg["name"],
                "provider": model_cfg["provider"],
            }
            results.append(record)
            completed_ids.add(item["source_id"])
            processed_count += 1
            model_index = processed_count % len(active_models)
            persist_results(results)
            print(
                f"✓ {item['source_id']} via {model_cfg['label']} — total {len(results)} entries"
            )
        else:
            print(f"✗ {item['source_id']} via {model_cfg['label']} (no annotation)")
            model_index = (model_index + 1) % len(active_models)

        time.sleep(0.5)

    print(
        f"\n✅ Finished. Generated {len(results)} triplets after {total_requests} model calls."
    )
    print(f"📁 Output saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main() 
