from __future__ import annotations

import re
import time
from dataclasses import dataclass

import httpx
import ollama


@dataclass
class ModelResponse:
    answer: str          # final answer text (thinking block stripped)
    think_content: str   # raw content of the <think> block (empty if thinking off)
    total_tokens: int    # total generated tokens reported by Ollama (think + answer)
    prompt_tokens: int   # tokens in the prompt
    latency_ms: float


_THINK_TAG = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# Patterns searched in thinking content (in priority order) when content is empty.
_ANSWER_PATTERNS = [
    # "The answer is X" / "answer: X" / "= X" at a sentence boundary
    re.compile(r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*\**\(?([A-E0-9][^\n,;.]{0,40}?)\)?\.?\s*$", re.IGNORECASE | re.MULTILINE),
    # Boxed expression \boxed{X}
    re.compile(r"\\boxed\{([^}]+)\}"),
    # Standalone letter on its own line near the end
    re.compile(r"^\s*([A-E])\s*$", re.MULTILINE),
]


def _extract_from_thinking(thinking: str) -> str:
    """
    Search the thinking block for an explicit final-answer signal.
    Tries answer patterns against the last 1000 chars first (highest density),
    then the full block.  Falls back to last non-empty line only if nothing matches.
    """
    for window in (thinking[-1000:], thinking):
        for pattern in _ANSWER_PATTERNS:
            matches = pattern.findall(window)
            if matches:
                return matches[-1].strip()

    # Last resort: last non-empty line
    return next(
        (l.strip() for l in reversed(thinking.splitlines()) if l.strip()),
        "",
    )


class ModelRunner:
    # Hard wall-clock limit per call.  With num_predict=2048 worst-case is
    # ~30 s; 120 s gives 4× headroom before the timeout fires.
    _TIMEOUT_S = 120

    def __init__(
        self,
        model: str,
        thinking: bool,
        temperature: float = 0.0,
        host: str = "http://172.19.64.1:11434",
    ):
        self.model = model
        self.thinking = thinking
        self.temperature = temperature
        self._client = ollama.Client(
            host=host,
            # httpx timeout: (connect, read).  The read timeout covers the full
            # streaming response, so it must be >= the worst-case generation time.
            timeout=httpx.Timeout(float(self._TIMEOUT_S), connect=10.0),
        )

    def run(self, prompt: str) -> ModelResponse:
        messages = [{"role": "user", "content": prompt}]

        # num_predict caps generated tokens: 2048 ≈ 25 s worst case for thinking,
        # 256 is generous for a direct answer in non-thinking mode.
        # num_ctx must be large enough for prompt + generated tokens to fit.
        options: dict = {
            "temperature": self.temperature,
            "num_ctx": 4096,
            "num_predict": 2048 if self.thinking else 256,
        }

        t0 = time.time()
        response = self._client.chat(
            model=self.model,
            messages=messages,
            options=options,
            think=self.thinking,
        )
        latency_ms = (time.time() - t0) * 1000

        raw = response.message.content or ""
        think_content = getattr(response.message, "thinking", "") or ""

        # Older Ollama builds embed the think block inside content.
        if not think_content:
            m = _THINK_TAG.search(raw)
            if m:
                think_content = m.group(1).strip()
                raw = _THINK_TAG.sub("", raw).strip()

        # When the thinking budget is exhausted the model never emits content.
        # Extract the best available answer signal from the thinking block.
        if not raw.strip() and think_content:
            raw = _extract_from_thinking(think_content)

        return ModelResponse(
            answer=raw.strip(),
            think_content=think_content,
            total_tokens=response.eval_count or 0,
            prompt_tokens=response.prompt_eval_count or 0,
            latency_ms=latency_ms,
        )
