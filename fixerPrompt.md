# ✅ **1. What a PromptCoT 2.0 Triple _is_**

A PromptCoT triple is:

> **(Concepts → Rationale → Problem)**
> A dataset row that aligns a set of key mathematical concepts (𝑐), a latent structured reasoning blueprint (𝑧), and a final well-formed problem (𝑥).

This triple is used in two ways:

### **Cold-start (supervised)**

The rationale generator is trained on:

- **p(z | c, x)** → infer rationale given concepts + problem
  The problem generator is trained on:
- **p(x | c, z)** → produce problem from concepts + rationale

### **EM (iterative refinement)**

The rationale becomes a **latent variable** that guides problem generation.

**For this to work, each component must be consistent, meaningful, and mathematically grounded.**

---

# ✅ **2. Structure and Requirements for Each Component**

Below is the exact structure you want.
This is the "gold standard" expected by PromptCoT 2.0.

---

# 🧩 **2.1. Concepts (c)**

### **Purpose**

Define _what mathematical ideas or skills the final problem is built from._

### **Format**

- Bullet list or comma-separated list
- Each entry is a _single mathematical concept_, not a skill like "logical reasoning"
- Should be recognizable and discrete, e.g.:

**Examples of valid concepts**

- “Properties of logarithms”
- “Modular arithmetic with primes”
- “Symmetry in complex numbers”
- “Cauchy-Schwarz inequality”
- “Expected value and combinatorial probability”
- “Unit circle geometry in ℂ”

### **Constraints**

- Concepts must **all appear in the final problem**
- Do **not** include “soft skills” like:

  - “logical reasoning”
  - “quantitative reasoning”
  - “making estimations”
  - “nested functions”
    These do not help the generative model and cause drift.

---

# 🧩 **2.2. Rationale (z)**

### **Purpose**

Latent structure guiding the final problem creation.

Not a solution.
Not a chain-of-thought.
Not a meta-explanation of how a human designs problems.

### **Format**

A short–medium paragraph that describes:

1. **How the concepts will interact**
2. **What mathematical relationships or constraints will be used**
3. **What structure the final problem should have**
4. **How to enforce difficulty, ambiguity removal, or edge cases**

### **Example patterns of a proper PromptCoT rationale**

- “Combine modular arithmetic with multiplicative orders to enforce a hidden periodicity.”
- “Use the symmetry of |z−a| = |z−b| to force the locus to be a perpendicular bisector.”
- “Introduce a combinatorial structure where the final quantity depends on two different binomial coefficients.”

### **Critical rule**

Rationale must describe **construction logic**, not **problem-solving** steps.
It is a “blueprint”, not a “solution”.

### **Constraints**

A rationale must **lead naturally** into the problem.
It must include all the concepts, and only those.

---

# 🧩 **2.3. Problem (x)**

### **Purpose**

A well-formed math competition problem embodying the concepts.

### **Requirements**

- Single, self-contained, unambiguous question
- Uses **exactly the concepts** from the concepts list
- Matches the structural guidance of the rationale
- Has a definite numeric or symbolic final answer
- Has AIME/HMMT-level structure if you're training for those tasks

### **Forbidden**

- Probability without a defined sample space
- “What is the probability…” with no randomness
- Riddles
- Open-ended proofs unless you explicitly train for them
- Problems that completely ignore the concepts or rationale

---

# ❗ **3. What the triple MUST NOT look like**

### ❌ Concepts:

- “logical reasoning”, “estimation”, “nested operations”, “solving problems” → too generic
- List of major skills instead of math concepts
- Concepts not present in final problem

### ❌ Rationale:

- A step-by-step solution → NO
- A meta explanation of “how a problem designer thinks” → NO
- Discussion of difficulty level selection → NO
- Describing the problem after the fact → NO
- Describing probability/fraction steps that never appear → NO

### ❌ Problem:

- Ill-posed probability questions
- Missing constraints
- Does not use the concepts
- Simple exercises (e.g., “compute 2+2i”)
- Problems answerable without all concepts
- Ambiguous wording

---

# ✅ **4. A Perfect PromptCoT 2.0 Triple (Good Example)**

### **Concepts**

- Geometry of complex numbers
- Perpendicular bisector property of equal distances
- Real inner product in ℂ as ℝ²

### **Rationale**

“To integrate these concepts, enforce two equal-distance constraints in the complex plane that force geometric structure. Use |z−a| = |z−b|–type symmetries so the solution must reason about perpendicular bisectors and inner products. Introduce an additional algebraic constraint so the problem cannot be solved by geometry alone, ensuring interaction between geometric and algebraic reasoning.”

### **Problem**

Let (a) and (b) be distinct complex numbers with (|z-a| = |z-b|).
Suppose further that
[
\operatorname{Re}(az) = 2.
]
Find all possible values of (|a-b|).

---

# ❌ A Bad Triple (Example of What You Submitted)

Your original triple failed because:

- Concepts included irrelevant skills
- Rationale described a meta-process, not construction logic
- Problem used none of the concepts
- “Probability” question was ill-defined
- No mathematical structure linking the three parts

---

# 🔥 Summary — What You Want for PromptCoT 2.0

### **Concepts:**

→ A small set of concrete math concepts used in the final problem.

### **Rationale:**

→ A construction blueprint describing how those concepts combine to produce a challenging problem.

### **Problem:**

→ A valid, well-posed, competition-style math problem that exactly matches the concepts and rationale.

If you'd like, I can now:

✅ **Generate a correct triple for your specific domain**
or
✅ **Rewrite your broken triple into a valid PromptCoT 2.0 triple**

Just say:
**“Create a proper PromptCoT 2.0 triple for X concepts.”**
