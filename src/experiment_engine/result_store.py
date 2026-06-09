from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterator, Tuple


# Each row written to the JSONL log.
ResultRow = Dict

# The composite key that uniquely identifies one inference call.
RunKey = Tuple[str, str, int, int]   # (sample_id, condition, k, seed)


class ResultStore:
    """
    Append-only JSON-Lines store.  Supports resumption: completed run-keys
    are skipped on restart without re-running inference.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._completed: set[RunKey] = set()
        self._load_existing()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_done(self, sample_id: str, condition: str, k: int, seed: int) -> bool:
        return (sample_id, condition, k, seed) in self._completed

    def append(self, row: ResultRow) -> None:
        key: RunKey = (row["sample_id"], row["condition"], row["k"], row["seed"])
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._completed.add(key)

    def iter_rows(self) -> Iterator[ResultRow]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_existing(self) -> None:
        for row in self.iter_rows():
            key: RunKey = (row["sample_id"], row["condition"], row["k"], row["seed"])
            self._completed.add(key)
