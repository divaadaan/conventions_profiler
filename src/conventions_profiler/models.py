"""
Core data models for the pipeline.

The Essay object flows through all passes, accumulating intermediate results.
Feature modules then extract numeric features from these intermediate results.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Essay:
    """A single essay flowing through the extraction pipeline.

    Attributes:
        essay_id: Unique identifier.
        text: Raw essay text.
        prompt_id: ASAP prompt number.
        conventions_score: Ground truth conventions score (raw scale per prompt).
        conventions_z: Z-scored conventions score for cross-prompt analysis.
        pass_results: Dict of intermediate results keyed by pass name.
            Populated by each pass during processing.
        features: Final extracted feature dict. Populated after all passes complete.
    """
    essay_id: str
    text: str
    prompt_id: int
    conventions_score: float | None = None
    conventions_z: float | None = None
    pass_results: dict[str, Any] = field(default_factory=dict)
    features: dict[str, float | int] = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.text.split())
