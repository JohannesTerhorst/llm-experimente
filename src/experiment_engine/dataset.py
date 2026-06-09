from __future__ import annotations

import gzip
import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from huggingface_hub import snapshot_download


@dataclass
class Sample:
    id: str
    question: str
    gold_solution: str       # full solution text from the dataset
    gold_answer: str         # extracted final answer (letter or number)
    skill_text: str
    raw_trace: Optional[str] = field(default=None, repr=False)  # filled by the D run


# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------

_MC_PATTERN = re.compile(
    r"\b(?:answer\s+is\s*[:\(]?\s*|choice\s+|option\s+|=\s*)"
    r"[\$\\]*\(?([A-E])\)?[\$\\]*",
    re.IGNORECASE,
)
_BOXED_PATTERN = re.compile(r"\\boxed\{([^}]+)\}")
_LAST_NUMBER = re.compile(r"[-+]?\d+(?:[,\./]\d+)*")


def extract_final_answer(text: str) -> str:
    """
    Best-effort extraction of the final answer from a solution text.
    Priority: \\boxed{} letter > \\boxed{} value > explicit MC letter > last number.
    Returns a normalised string.
    """
    # 1. LaTeX boxed answer — extract letter first, then fallback to full content
    boxed = _BOXED_PATTERN.findall(text)
    if boxed:
        inner = boxed[-1].strip()
        # e.g. \textbf{(B)}\ 58\frac{1}{2}  →  B
        mc_in_box = re.search(r"\(([A-E])\)", inner, re.IGNORECASE)
        if mc_in_box:
            return mc_in_box.group(1).upper()
        # plain number or expression
        return re.sub(r"[\\{}\s]", "", inner).lower()

    # 2. Explicit multiple-choice pattern in the last 300 chars
    tail = text[-300:]
    mc = _MC_PATTERN.findall(tail)
    if mc:
        return mc[-1].strip().upper()

    # 3. Standalone letter at end of text
    letters = re.findall(r"\b([A-E])\b", tail)
    if letters:
        return letters[-1].upper()

    # 4. Last number in text
    nums = _LAST_NUMBER.findall(text)
    if nums:
        return nums[-1].replace(",", "").strip()

    return text.strip()[-50:]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class DatasetLoader:
    def load(self, name: str, n_samples: int, seed: int) -> List[Sample]:
        if name == "trs":
            return self._load_trs(n_samples, seed)
        raise ValueError(f"Unknown dataset: {name}")

    # aops alone has ~7735 samples — sufficient for n_samples ≤ 200.
    # deepmath_103k adds ~103k more but is 59MB compressed; include only if needed.
    _MATH_SHARDS = [
        "data/aops_skill_corpus.jsonl.gz",
    ]
    _MATH_SHARDS_EXTENDED = [
        "data/aops_skill_corpus.jsonl.gz",
        "data/deepmath_103k_oss_skill_corpus.jsonl.gz",
    ]

    def _load_trs(self, n_samples: int, seed: int) -> List[Sample]:
        # The HF `datasets` loader fails on this repo because some shards have
        # extra columns that don't match the registered schema.  We bypass it
        # entirely and read the gzipped JSONL files from the hub cache directly.
        repo_dir = Path(
            snapshot_download(
                repo_id="stallone0000/Reasoning-Skill",
                repo_type="dataset",
            )
        )

        shards = self._MATH_SHARDS_EXTENDED if n_samples > 5000 else self._MATH_SHARDS
        all_rows: List[dict] = []
        for shard in shards:
            path = repo_dir / shard
            if not path.exists():
                continue
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        all_rows.append(json.loads(line))

        rng = random.Random(seed)
        rng.shuffle(all_rows)

        samples: List[Sample] = []
        for row in all_rows[:n_samples]:
            gold_answer = extract_final_answer(row["answer"])
            samples.append(Sample(
                id=row["question_id"],
                question=row["question"],
                gold_solution=row["answer"],
                gold_answer=gold_answer,
                skill_text=row["skill_text"],
            ))
        return samples
