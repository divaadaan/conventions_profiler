"""Core data models for the pipeline"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Essay:
    """A single essay object for the extraction pipeline.

        essay_id: Unique identifier.
        text: Raw essay text.
        prompt_id: Prompt/task grouping identifier.
        score: Ground truth conventions score.
        pass_results: Dict of intermediate results by pass name.
        features: Final extracted feature dict.
    """
    essay_id: str
    text: str
    prompt_id: int
    score: float | None = None
    pass_results: dict[str, Any] = field(default_factory=dict)
    features: dict[str, float | int] = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        spacy_total = self.pass_results.get("spacy", {}).get("total_words")
        if spacy_total is not None:
            return spacy_total
        return len(self.text.split())
