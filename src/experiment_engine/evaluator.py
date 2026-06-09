from __future__ import annotations

import re

from .dataset import extract_final_answer


def _normalise(text: str) -> str:
    """Lowercase, strip whitespace and common LaTeX wrappers."""
    text = text.strip().lower()
    text = re.sub(r"[$\\{}]", "", text)
    text = re.sub(r"\s+", " ", text)
    # normalise fractions: 5 1/3 → 5.333, leave simple numbers alone
    return text


class Evaluator:
    def score(self, model_output: str, gold_answer: str) -> bool:
        """Return True if the model's output matches the gold answer."""
        predicted = _normalise(extract_final_answer(model_output))
        gold = _normalise(gold_answer)
        return predicted == gold or gold in predicted
