"""
Base class for processing passes.

A pass loads a model, processes all essays to produce intermediate results,
and optionally unloads the model to free resources. Feature modules then
operate on the intermediate results stored in Essay.pass_results.
"""

from abc import ABC, abstractmethod

from conventions_profiler.models import Essay


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
        """Load the model/resource. Called once before processing begins."""
        ...

    @abstractmethod
    def process_essay(self, essay: Essay) -> None:
        """Process a single essay and store results in essay.pass_results[self.name].

        This method should NOT compute final features — only intermediate
        representations (parse trees, error match lists, etc.) that feature
        modules will consume.
        """
        ...

    def unload_model(self) -> None:
        """Optional: free resources after processing all essays."""
        pass

    def process_batch(self, essays: list[Essay]) -> None:
        """Process a list of essays. Override for batch-optimized passes."""
        for essay in essays:
            self.process_essay(essay)
