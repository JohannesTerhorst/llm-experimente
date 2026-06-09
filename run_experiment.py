"""
Experiment runner for "Think Twice: Does Repeating Reasoning Summaries
Increase Chain-of-Thought Efficiency?"

Usage
-----
    python run_experiment.py [--run-id <name>] [--n-samples <n>] [--thinking]

The run is fully resumable: re-running with the same --run-id picks up where
inference left off.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from experiment_engine import (
    ExperimentConfig,
    DatasetLoader,
    PromptBuilder,
    ModelRunner,
    Evaluator,
    ResultStore,
    AnalysisEngine,
)


# ---------------------------------------------------------------------------
# Result row schema
# ---------------------------------------------------------------------------

def _make_row(
    sample_id: str,
    condition: str,
    k: int,
    seed: int,
    correct: bool,
    model_answer: str,
    gold_answer: str,
    total_tokens: int,
    prompt_tokens: int,
    latency_ms: float,
    think_content: str,
) -> dict:
    return {
        "sample_id": sample_id,
        "condition": condition,
        "k": k,
        "seed": seed,
        "correct": correct,
        "model_answer": model_answer,
        "gold_answer": gold_answer,
        "total_tokens": total_tokens,
        "prompt_tokens": prompt_tokens,
        "latency_ms": latency_ms,
        # Think content stored for qualitative analysis; omit from summary stats.
        "think_chars": len(think_content),
    }


# ---------------------------------------------------------------------------
# Core sweep
# ---------------------------------------------------------------------------

def run_sweep(config: ExperimentConfig) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(config.output_dir) / f"{config.run_id}_{timestamp}.jsonl"
    print(f"Run ID  : {config.run_id}")
    print(f"Log     : {log_path}")
    print(f"Model   : {config.model}  (thinking={config.thinking})")
    print(f"Samples : {config.n_samples}  seeds={config.seeds}")
    print()

    loader = DatasetLoader()
    builder = PromptBuilder()
    runner = ModelRunner(config.model, config.thinking, config.temperature, config.ollama_host)
    evaluator = Evaluator()
    store = ResultStore(str(log_path))

    total_conditions = len(config.conditions)
    total_ks = len(config.repetitions)

    for seed in config.seeds:
        print(f"=== Seed {seed} ===")
        samples = loader.load(config.dataset, config.n_samples, seed)

        for idx, sample in enumerate(samples):
            prefix = f"  [{idx+1}/{len(samples)}] {sample.id}"

            # ----------------------------------------------------------------
            # Step 1: Always run Direct (k=0) first to get the baseline
            # result and — when thinking is enabled — the raw reasoning trace.
            # ----------------------------------------------------------------
            if not store.is_done(sample.id, "D", 0, seed):
                prompt = builder.build("D", 0, sample)
                resp = runner.run(prompt)
                correct = evaluator.score(resp.answer, sample.gold_answer)
                store.append(_make_row(
                    sample.id, "D", 0, seed,
                    correct, resp.answer, sample.gold_answer,
                    resp.total_tokens, resp.prompt_tokens,
                    resp.latency_ms, resp.think_content,
                ))
                # Store the trace on the sample object for RT conditions.
                sample.raw_trace = resp.think_content
                mark = "✓" if correct else "✗"
                print(f"{prefix}  D k=0 [{mark}]  tokens={resp.total_tokens}")
            else:
                # Reconstruct trace from stored result so RT can still run.
                for row in store.iter_rows():
                    if (row["sample_id"] == sample.id
                            and row["condition"] == "D"
                            and row["seed"] == seed):
                        # think_content itself is not stored to save space;
                        # RT requires a re-run of D if not cached in memory.
                        break

            # ----------------------------------------------------------------
            # Step 2: Sweep all (condition, k) pairs.
            # ----------------------------------------------------------------
            for condition in config.conditions:
                for k in config.repetitions:
                    if store.is_done(sample.id, condition, k, seed):
                        continue

                    # RT requires the raw trace from the D run.
                    if condition == "RT" and not sample.raw_trace:
                        print(f"{prefix}  RT k={k} SKIP (no trace cached — re-run from scratch)")
                        continue

                    try:
                        prompt = builder.build(condition, k, sample)
                        resp = runner.run(prompt)
                        correct = evaluator.score(resp.answer, sample.gold_answer)
                        store.append(_make_row(
                            sample.id, condition, k, seed,
                            correct, resp.answer, sample.gold_answer,
                            resp.total_tokens, resp.prompt_tokens,
                            resp.latency_ms, resp.think_content,
                        ))
                        mark = "✓" if correct else "✗"
                        print(
                            f"{prefix}  {condition} k={k} [{mark}]"
                            f"  tokens={resp.total_tokens}"
                        )
                    except Exception as exc:
                        print(f"{prefix}  {condition} k={k} ERROR: {exc}")

    print()
    print("Sweep complete.")
    return str(log_path)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def run_analysis(log_path: str, config: ExperimentConfig) -> None:
    print("Running analysis...")
    engine = AnalysisEngine(log_path, config.output_dir, config.run_id)
    engine.save_csv()
    engine.plot_accuracy_vs_k()
    engine.plot_tokens_vs_k()
    latex = engine.to_latex_table()
    print()
    print("=== LaTeX Table ===")
    print(latex)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> ExperimentConfig:
    p = argparse.ArgumentParser(description="Experiment runner")
    p.add_argument("--run-id",    default="experiment", help="Unique name for this run")
    p.add_argument("--model",     default="qwen3.5:4b")
    p.add_argument("--thinking",  action="store_true",  help="Enable Qwen3 thinking mode")
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--seeds",     type=int, nargs="+", default=[42, 7, 13])
    p.add_argument("--ks",        type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--conditions",nargs="+",
                   default=["SR", "PR", "RT", "Fill"])
    p.add_argument("--output-dir",default="logs")
    p.add_argument("--no-analysis", action="store_true",
                   help="Skip analysis step after sweep")
    args = p.parse_args()

    return ExperimentConfig(
        model=args.model,
        thinking=args.thinking,
        dataset="trs",
        n_samples=args.n_samples,
        conditions=args.conditions,
        repetitions=args.ks,
        seeds=args.seeds,
        output_dir=args.output_dir,
        run_id=args.run_id,
    ), args.no_analysis


if __name__ == "__main__":
    config, skip_analysis = parse_args()
    log_path = run_sweep(config)
    if not skip_analysis:
        run_analysis(log_path, config)
