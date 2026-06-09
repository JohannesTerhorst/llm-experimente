# Project: Think Twice — Repeating Reasoning Summaries

**Paper:** "Think Twice: Does Repeating Reasoning Summaries Increase Chain-of-Thought Efficiency?"
**Team:** Fabian Karl, Kerui Ren, Jonas Wachter, Johannes Terhorst (TU Munich)

---

## Chapter 3 — Method (Draft)

### 3.1 Task Formulation

We study how repeating a pre-existing, distilled reasoning summary affects (a) final-answer accuracy and (b) the number of tokens generated during inference, without modifying model weights. Formally, let $x$ be a math problem, $y^*$ its ground-truth answer, and $s$ a pre-computed skill text that encodes an abstract, task-relevant reasoning strategy. We define the following prompt variants, each indexed by a repetition count $k \in \{1, 2, 3, 4, 5\}$:

| Condition | Prompt structure |
|---|---|
| **Direct (D)** | `[x]` — zero-shot baseline, no injected context |
| **Prompt Repetition (PR_k)** | `[x] × k` — the problem statement repeated k times |
| **Summary Repetition (SR_k)** | `[s × k] [x]` — the skill text repeated k times, then the problem |
| **Raw Reasoning Transfer (RT_k)** | `[raw_trace × k] [x]` — a full reasoning trace repeated k times |

The key design constraint is that all repetition-based conditions are matched on copy count $k$, isolating the *content* of the repeated object (problem vs. summary vs. raw trace) as the variable of interest.

The primary goal of this design is to **separate the repetition effect from the insight effect**: since the skill texts are taken directly from an existing dataset, there is no retrieval or extraction step that could introduce confounds. The only variable we manipulate is how many times the same skill text is prepended.

### 3.2 Dataset

We use the **TRS Reasoning-Skill dataset** (`stallone0000/Reasoning-Skill` on HuggingFace), which provides math problems together with pre-computed skill texts (abstract reasoning strategies). This dataset was chosen because:

1. It already contains (question, answer, skill_text) triples, so no auxiliary retriever or skill extractor is needed.
2. Eliminating retrieval and extraction removes two potential confounds — model choice for extraction and retrieval similarity score — which would otherwise entangle the insight effect with the repetition effect.
3. The math domain is well-suited to reasoning models and provides objective, binary accuracy evaluation.

For each problem we use its associated `skill_text` field directly as $s$ in the SR_k condition. For the RT_k condition we use a full model-generated reasoning trace for that same problem (generated once, then frozen).

### 3.3 Models

The primary comparison is **Qwen3-4B** run in two modes:
- **Thinking enabled** (`/think`) — reasoning model setting
- **Thinking disabled** (`/no-think`) — non-reasoning model setting

This controlled pair tests whether repetition effects differ between reasoning and non-reasoning inference on the same underlying model. A second model family is included for robustness to distinguish Qwen-specific behavior from general effects. Model choice for the second family is TBD.

All models are accessed via the Ollama local inference API (consistent with the existing experiment infrastructure). No fine-tuning is performed; all interventions are purely prompt-level.

### 3.4 Procedure

1. **Skill text source:** For each problem in the evaluation split, retrieve its `skill_text` from the TRS dataset. This is $s$.
2. **Raw trace generation:** For RT_k, run the reasoning model once per problem with thinking enabled and no injected context. Store the generated `<think>...</think>` block as the raw trace. This is done once and frozen.
3. **Prompt construction:** Construct all four conditions (D, PR_k, SR_k, RT_k) for each problem and each k.
4. **Inference:** For each (problem, condition, k) triple, run the model and record:
   - The final answer (extracted from model output)
   - Total generated tokens (thinking tokens + answer tokens, where applicable)
   - Wall-clock generation time
5. **Evaluation:** Compare the predicted answer against the ground-truth answer for binary accuracy.

### 3.5 Metrics

- **Accuracy** — fraction of problems answered correctly, computed per condition and per $k$.
- **Generated token count** — total tokens produced by the model (including hidden `<think>` tokens where thinking is enabled). This is the primary efficiency metric.
- **Accuracy vs. token count trade-off** — plotted as a curve over $k$ to identify whether any repetition count simultaneously improves accuracy and reduces tokens.

### 3.6 Baselines and Comparisons

The experimental design supports the following pairwise contrasts:

| Contrast | What it isolates |
|---|---|
| SR_k vs. D | Net effect of injecting a repeated summary |
| SR_k vs. PR_k | Semantic value of the summary over pure mechanical repetition |
| SR_k vs. RT_k | Value of distillation: abstract summary vs. raw reasoning trace |
| SR_k(k=1) vs. SR_k(k>1) | Pure repetition effect *within* the summary condition |
| Thinking ON vs. OFF | Whether effects are specific to reasoning vs. non-reasoning inference |

### 3.7 Training & Evaluation Objective

No training is performed. The evaluation objective is:

$$\mathcal{L} = \mathbb{E}_{x \sim \mathcal{D}} [\ell(f_\theta(x), y^*)]$$

where $\ell$ is exact-match accuracy, $f_\theta$ is the frozen model, and $x$ is the full constructed prompt for a given condition and $k$.

---

## Experiment Implementation Notes

- **Dataset:** Load `stallone0000/Reasoning-Skill` from HuggingFace using the `datasets` library.
- **Model API:** Use Ollama at `http://localhost:11434` with `qwen3:4b` (consistent with prior notebooks).
- **Thinking toggle:** Pass `thinking: true/false` in the model options, or use `/think` / `/no-think` system tokens per Qwen3 documentation.
- **Logging:** Append each result to a JSON log file (consistent with the existing `logs/` structure).
- **Plotting:** Generate 4-way plots over k for each metric (consistent with `plots/` structure).
- **Seed:** Fix a random seed and use a fixed subset of the dataset for reproducibility.
