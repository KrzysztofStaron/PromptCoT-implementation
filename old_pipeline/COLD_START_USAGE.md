# Cold Start Data Generation for PromptCoT 2.0

This script generates olympiad-level coding problems by streaming the PromptCoT-2.0-Concepts dataset and using the PromptCoT-2.0-Prompt-Generation-Model.

## Prerequisites

- Python 3.8+
- Required packages (install via `pip install -r requirements.txt`)
- For model loading: Linux, macOS, or WSL (Windows has TensorFlow compatibility issues)

## Usage

### Basic Usage (Generate All Problems)
```bash
python coldStartData.py
```

### Generate Limited Number of Problems
```bash
python coldStartData.py --max_problems 100
```

### Custom Output File
```bash
python coldStartData.py --max_problems 50 --output_file my_problems.jsonl
```

## Output Format

The script generates a JSONL file with entries like:
```json
{
  "concepts": ["Graph Traversal", "Dynamic Programming", "Recursion"],
  "level": "codeforces",
  "generated_problem": "Full generated problem text..."
}
```

## Dataset Information

- **Source**: `xl-zhao/PromptCoT-2.0-Concepts`
- **Size**: 416,009 entries
- **Format**: Each entry contains `foundational_concepts` (list), `level` (string), and `prompt` (string)
- **Usage**: The script uses the pre-formatted `prompt` field directly for generation

## Known Issues

- **Windows TensorFlow Error**: If you get TensorFlow import errors on Windows, use:
  - Linux/macOS
  - Windows Subsystem for Linux (WSL)
  - Google Colab
  - Other cloud environments

## Example Concepts from Dataset

- "Graph traversal, recursion, dynamic programming"
- "Modular Arithmetic, Dynamic Programming, Matrix Multiplication"
- "Depth-First Search, Breadth-First Search, Tree algorithms"

All at various difficulty levels (primarily "codeforces" level).
