"""
Pipeline orchestrator.

Runs passes in sequence, manages checkpointing, and coordinates feature extraction.
Handles both single-GPU and CPU execution. Multi-GPU parallelism is deferred
to an extension module.
"""

import logging
import time
from pathlib import Path

from tqdm import tqdm

from conventions_profiler.models import Essay
from conventions_profiler.features.aggregation import compute_aggregation_features
from conventions_profiler.features.dependency import compute_dependency_features
from conventions_profiler.features.error_severity import compute_error_severity_features
from conventions_profiler.features.punctuation import compute_punctuation_features
from conventions_profiler.features.spelling import compute_spelling_features
from conventions_profiler.features.syntactic_complexity import compute_syntactic_complexity_features
from conventions_profiler.passes.base import BasePass
from conventions_profiler.passes.languagetool_pass import LanguageToolPass
from conventions_profiler.passes.spacy_pass import SpacyPass
from conventions_profiler.pipeline.checkpoint import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)

# Feature modules to run after all passes complete.
# Each takes an Essay (with populated pass_results) and returns a feature dict.
FEATURE_MODULES = [
    ("dependency", compute_dependency_features),
    ("syntactic_complexity", compute_syntactic_complexity_features),
    ("punctuation", compute_punctuation_features),
    ("error_severity", compute_error_severity_features),
    ("spelling", compute_spelling_features),
    ("aggregation", compute_aggregation_features),
]


class Orchestrator:
    """Runs the full extraction pipeline: passes → features."""

    def __init__(self, config: dict, checkpoint_dir: Path | None = None):
        self.config = config
        self.checkpoint_dir = checkpoint_dir

    def _get_passes(self) -> list[BasePass]:
        """Instantiate passes in execution order."""
        return [
            SpacyPass(self.config),
            LanguageToolPass(self.config),
        ]

    def run_passes(self, essays: list[Essay], force: bool = False) -> None:
        """Execute all passes over the essay corpus.

        Parameters
        ----------
        essays
            List of Essay objects to process.
        force
            If True, ignore checkpoints and re-run all passes.
        """
        for pass_instance in self._get_passes():
            pass_name = pass_instance.name

            # Try loading checkpoint
            if self.checkpoint_dir and not force:
                if load_checkpoint(essays, pass_name, self.checkpoint_dir):
                    logger.info(f"Skipping pass '{pass_name}' (checkpoint)")
                    continue

            # Run pass
            logger.info(f"Pass '{pass_name}': processing {len(essays)} essays...")
            pass_instance.load_model()
            start = time.time()

            # Use batch processing if available
            if hasattr(pass_instance, "process_batch") and pass_name == "spacy":
                # Process in batches for spaCy pipe efficiency
                batch_size = self.config.get("hardware", {}).get("batch_size", 16)
                for i in tqdm(
                    range(0, len(essays), batch_size),
                    desc=f"Pass: {pass_name}",
                    unit="batch",
                ):
                    batch = essays[i:i + batch_size]
                    pass_instance.process_batch(batch)
            else:
                for essay in tqdm(essays, desc=f"Pass: {pass_name}", unit="essay"):
                    pass_instance.process_essay(essay)

            elapsed = time.time() - start
            eps = len(essays) / elapsed if elapsed > 0 else 0
            logger.info(
                f"Pass '{pass_name}': completed in {elapsed:.1f}s "
                f"({eps:.1f} essays/sec)"
            )

            # Unload model to free resources before saving checkpoint
            pass_instance.unload_model()

            # Save checkpoint
            if self.checkpoint_dir:
                save_checkpoint(essays, pass_name, self.checkpoint_dir)

    def compute_features(self, essays: list[Essay]) -> None:
        """Run all feature modules over essays with populated pass_results.

        Features are stored in each essay.features dict.
        """
        logger.info(f"Computing features for {len(essays)} essays...")

        for essay in tqdm(essays, desc="Feature extraction", unit="essay"):
            for module_name, compute_fn in FEATURE_MODULES:
                try:
                    features = compute_fn(essay)
                    essay.features.update(features)
                except Exception as e:
                    logger.error(
                        f"Feature module '{module_name}' failed for "
                        f"essay {essay.essay_id}: {e}"
                    )

    def run(self, essays: list[Essay], force: bool = False) -> list[Essay]:
        """Execute the full pipeline: passes → features.

        Parameters
        ----------
        essays
            List of Essay objects with text populated.
        force
            If True, re-run all passes ignoring checkpoints.

        Returns
        -------
        list[Essay]
            Same essays with pass_results and features populated.
        """
        self.run_passes(essays, force=force)
        self.compute_features(essays)

        # Log feature count
        if essays:
            n_features = len(essays[0].features)
            logger.info(f"Extraction complete: {n_features} features per essay")

        return essays
