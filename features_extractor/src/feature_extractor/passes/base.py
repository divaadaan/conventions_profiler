"""Base class for processing passes."""

from abc import ABC, abstractmethod

from feature_extractor.models import Essay


class BasePass(ABC):
    """Abstract base class for pipeline processing passes."""

    def __init__(self, config: dict):
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this pass, used as key in Essay.pass_results."""
        ...

    @abstractmethod
    def load_model(self) -> None:
        """Load the model/resource."""
        ...

    @abstractmethod
    def process_essay(self, essay: Essay) -> None:
        """Process a single essay and store results in essay.pass_results[self.name]"""
        ...

    def unload_model(self) -> None:
        """Optional: free resources after processing all essays."""
        pass

    def process_batch(self, essays: list[Essay]) -> None:
        """batch-optimized passes"""
        for essay in essays:
            self.process_essay(essay)
