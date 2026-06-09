from dataclasses import dataclass, field
from typing import List


@dataclass
class ExperimentConfig:
    # Model
    model: str = "qwen3.5:4b"
    thinking: bool = True

    # Dataset
    dataset: str = "trs"
    n_samples: int = 200

    # Sweep parameters
    # conditions: D is always run first (k=0); remaining conditions are swept over repetitions
    conditions: List[str] = field(default_factory=lambda: ["SR", "PR", "RT", "Fill"])
    repetitions: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])

    # Statistical validity
    seeds: List[int] = field(default_factory=lambda: [42, 7, 13])

    # Output
    output_dir: str = "logs"
    run_id: str = "experiment"

    # Inference
    temperature: float = 0.0
    ollama_host: str = "http://172.19.64.1:11434"
