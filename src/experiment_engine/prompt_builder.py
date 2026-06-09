from __future__ import annotations

from .dataset import Sample

# Instruction appended to every prompt regardless of condition.
_ANSWER_INSTRUCTION = (
    "Provide only the final answer. "
    "For multiple choice, write the letter only (e.g. A). "
    "For numerical answers, write the number only."
)


class PromptBuilder:
    """
    Assembles the full prompt string for a given condition and repetition count.

    Conditions
    ----------
    D     (k=0) : question only — zero-shot baseline.
    SR    (k≥1) : skill_text × k, then question.
    PR    (k≥1) : question repeated k+1 times total.
    RT    (k≥1) : raw_trace × k, then question.  Requires sample.raw_trace to be set.
    Fill  (k≥1) : period-filler of the same character length as skill_text × k,
                  then question.  Length-matched noise control.
    """

    def build(self, condition: str, k: int, sample: Sample) -> str:
        if condition == "D" or k == 0:
            return self._wrap(sample.question)

        if condition == "SR":
            prefix = (sample.skill_text + "\n\n") * k
            return self._wrap(prefix + sample.question)

        if condition == "PR":
            # k extra copies of the question prepended → k+1 copies total
            prefix = (sample.question + "\n\n") * k
            return self._wrap(prefix + sample.question)

        if condition == "RT":
            if not sample.raw_trace:
                raise ValueError(
                    f"raw_trace not set for sample {sample.id}. "
                    "Run the D condition first."
                )
            prefix = (sample.raw_trace + "\n\n") * k
            return self._wrap(prefix + sample.question)

        if condition == "Fill":
            # Period filler matched to the character length of one skill_text copy.
            filler_unit = "." * len(sample.skill_text)
            prefix = (filler_unit + "\n\n") * k
            return self._wrap(prefix + sample.question)

        raise ValueError(f"Unknown condition: {condition}")

    @staticmethod
    def _wrap(body: str) -> str:
        return f"{body}\n\n{_ANSWER_INSTRUCTION}"
