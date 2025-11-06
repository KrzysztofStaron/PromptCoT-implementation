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

MODEL_CONFIGS: List[Dict[str, str]] = [
    {"name": "gpt-4o", "provider": "openai", "label": "openai:gpt-4o"},
    {"name": "openai/gpt-4.1-mini", "provider": "openrouter", "label": "openrouter:openai/gpt-4.1-mini"},
    {"name": "anthropic/claude-3.5-sonnet", "provider": "openrouter", "label": "openrouter:anthropic/claude-3.5-sonnet"},
    {"name": "google/gemini-1.5-pro", "provider": "openrouter", "label": "openrouter:google/gemini-1.5-pro"},
]

CHAT_TOKEN_PATTERN = re.compile(r"<[^>]+>")


def load_concept_pool() -> List[str]:
    pool: List[str] = []
    if not CONCEPT_FILE.exists():
        return pool
    with open(CONCEPT_FILE, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            pool.extend(data.get("concepts", []))
    unique = list({concept.strip(): None for concept in pool}.keys())
    return unique


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


def collect_problem_bank() -> List[Dict[str, str]]:
    dataset_paths: List[Path] = []
    if BASE_DIR.exists():
        dataset_paths.extend(
            [p for p in BASE_DIR.glob("*.jsonl") if p.name != CONCEPT_FILE.name]
        )
        qwq_dir = BASE_DIR / "qwq"
        if qwq_dir.exists():
            dataset_paths.extend(list(qwq_dir.glob("*.jsonl")))

    bank: List[Dict[str, str]] = []
    for dataset_path in dataset_paths:
        with open(dataset_path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle):
                if not line.strip():
                    continue
                payload = json.loads(line)
                raw_prompt = payload.get("prompt") or payload.get("Problem") or payload.get("output")
                solution = resolve_solution(payload)
                if not raw_prompt or not solution:
                    continue
                problem = clean_prompt(raw_prompt)
                if not problem:
                    continue
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


def build_prompt(problem: str, solution: str, concept_examples: List[str]) -> str:
    bullets = "\n".join(f"- {concept}" for concept in concept_examples)
    return (
        "Analyze the following math problem and its worked solution. "
        "Return a JSON object with exactly five high-signal mathematical concepts "
        "(ordered by importance) and a concise rationale that follows the solution's logic.\n\n"
        f"Problem:\n{problem}\n\n"
        f"Solution:\n{solution}\n\n"
        "Concept examples (use as inspiration, but prefer precise matches):\n"
        f"{bullets}\n\n"
        "Respond in this JSON schema:\n"
        "{\n  \"concepts\": [\"concept1\", \"concept2\", \"concept3\", \"concept4\", \"concept5\"],\n"
        "  \"rationale\": \"step-by-step reasoning\"\n}"
    )


def annotate_with_model(
    problem: str,
    solution: str,
    model_cfg: Dict[str, str],
    concept_pool: List[str],
) -> Optional[Dict[str, object]]:
    sample_size = min(20, len(concept_pool))
    concept_examples = random.sample(concept_pool, sample_size) if sample_size else []
    prompt = build_prompt(problem, solution, concept_examples)

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
                "max_tokens": 512,
                "response_format": {"type": "json_object"},
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
        return {"concepts": concepts, "rationale": rationale.strip()}
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
            model_key = record.get("model", "unknown")
            completed.add((problem_key, model_key))
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
    concept_pool = load_concept_pool()
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

    results, completed_pairs = load_existing_results()
    print(f"🗂️  Existing annotations: {len(results)}")

    total_requests = 0
    for item in problem_bank:
        for model_cfg in active_models:
            pair_key = (item["source_id"], model_cfg["name"])
            if pair_key in completed_pairs:
                continue
            annotation = annotate_with_model(
                item["problem"],
                item["solution"],
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
                completed_pairs.add(pair_key)
                persist_results(results)
                print(
                    f"✓ {item['source_id']} via {model_cfg['label']} — total {len(results)} entries"
                )
            else:
                print(f"✗ {item['source_id']} via {model_cfg['label']} (no annotation)")
            time.sleep(0.5)

    print(
        f"\n✅ Finished. Generated {len(results)} triplets after {total_requests} model calls."
    )
    print(f"📁 Output saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main() 
