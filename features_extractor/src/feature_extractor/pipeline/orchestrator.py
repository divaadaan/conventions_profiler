"""Pipeline orchestrator."""

import logging
import time
from pathlib import Path

from tqdm import tqdm

from feature_extractor.models import Essay
from feature_extractor.features.aggregation import compute_aggregation_features
from feature_extractor.features.capitalization import compute_capitalization_features
from feature_extractor.features.dependency import compute_dependency_features
from feature_extractor.features.error_severity import compute_error_severity_features
from feature_extractor.features.lemma_vocab import compute_lemma_vocab_features
from feature_extractor.features.punctuation import compute_punctuation_features
from feature_extractor.features.readability import compute_readability_features
from feature_extractor.features.sentence_boundary import compute_sentence_boundary_features
from feature_extractor.features.spelling import compute_spelling_features
from feature_extractor.features.surprisal import compute_surprisal_features
from feature_extractor.features.syntactic_complexity import compute_syntactic_complexity_features
from feature_extractor.features.registry import get_enabled_features, get_required_modules
from feature_extractor.passes.base import BasePass
from feature_extractor.passes.languagetool_pass import LanguageToolPass
from feature_extractor.passes.spacy_pass import SpacyPass
from feature_extractor.passes.surprisal_pass import SurprisalPass
from feature_extractor.pipeline.checkpoint import (
    load_checkpoint,
    load_partial_checkpoint,
    save_checkpoint_incremental,
)

logger = logging.getLogger(__name__)

FEATURE_MODULES = [
    ("dependency", compute_dependency_features),
    ("syntactic_complexity", compute_syntactic_complexity_features),
    ("punctuation", compute_punctuation_features),
    ("error_severity", compute_error_severity_features),
    ("spelling", compute_spelling_features),
    ("capitalization", compute_capitalization_features),
    ("sentence_boundary", compute_sentence_boundary_features),
    ("readability", compute_readability_features),
    ("lemma_vocab", compute_lemma_vocab_features),
    ("surprisal", compute_surprisal_features),
    ("aggregation", compute_aggregation_features),
]


class Orchestrator:
    """Runs the full extraction pipeline: passes → features."""

    def __init__(self, config: dict, checkpoint_dir: Path | None = None):
        self.config = config
        self.checkpoint_dir = checkpoint_dir

        # get enabled features from registry + config overrides
        feature_overrides = config.get("features", {}).get("overrides", {})
        self.enabled_features = get_enabled_features(feature_overrides)
        self.required_modules = get_required_modules(self.enabled_features)
        logger.info(
            f"Enabled features: {len(self.enabled_features)} "
            f"(modules: {sorted(self.required_modules)})"
        )

    def _get_passes(self) -> list[BasePass]:
        """Instantiate passes in execution order."""
        passes: list[BasePass] = [
            SpacyPass(self.config),
            LanguageToolPass(self.config),
        ]
        surprisal_cfg = self.config.get("models", {}).get("surprisal", {})
        if surprisal_cfg.get("enabled", False):
            passes.append(SurprisalPass(self.config))
        return passes

    def _invalidate_stale_checkpoints(
        self, essays: list[Essay], pass_instance: BasePass
    ) -> list[Essay]:
        """Clear pass_results entries that fail per-pass validation (e.g., a
        model version change for the surprisal pass). Returns the essays
        whose entries were invalidated and now need re-processing.
        """
        pass_name = pass_instance.name
        stale = []
        for essay in essays:
            if pass_name not in essay.pass_results:
                continue
            if not pass_instance.validate_pass_results(essay):
                del essay.pass_results[pass_name]
                stale.append(essay)
        if stale:
            logger.warning(
                f"Pass '{pass_name}': invalidated {len(stale)} stale "
                f"checkpoint entries; re-running for those essays."
            )
        return stale

    def run_passes(self, essays: list[Essay], force: bool = False) -> None:
        """Execute all passes over the essay corpus.
            force flag -> ignore checkpoints and re-run all passes.
        """
        chunk_size = self.config.get("cache", {}).get("checkpoint_chunk_size", 500)

        for pass_instance in self._get_passes():
            pass_name = pass_instance.name

            if self.checkpoint_dir and not force:
                if load_checkpoint(essays, pass_name, self.checkpoint_dir):
                    stale = self._invalidate_stale_checkpoints(essays, pass_instance)
                    if not stale:
                        logger.info(f"Skipping pass '{pass_name}' (checkpoint)")
                        continue
                    remaining = stale
                else:
                    remaining = load_partial_checkpoint(
                        essays, pass_name, self.checkpoint_dir
                    )
                    stale = self._invalidate_stale_checkpoints(essays, pass_instance)
                    remaining_ids = {e.essay_id for e in remaining}
                    for e in stale:
                        if e.essay_id not in remaining_ids:
                            remaining.append(e)
            else:
                remaining = essays

            if not remaining:
                logger.info(f"Skipping pass '{pass_name}' (fully checkpointed)")
                continue

            # Run pass
            logger.info(
                f"Pass '{pass_name}': processing {len(remaining)} essays "
                f"({len(essays) - len(remaining)} restored from checkpoint)..."
            )
            pass_instance.load_model()
            start = time.time()
            batch_size = self.config.get("hardware", {}).get("batch_size", 16)

            for chunk_start in range(0, len(remaining), chunk_size):
                chunk = remaining[chunk_start:chunk_start + chunk_size]

                if hasattr(pass_instance, "process_batch") and pass_name == "spacy":
                    for i in tqdm(
                        range(0, len(chunk), batch_size),
                        desc=f"Pass: {pass_name} (chunk {chunk_start // chunk_size + 1})",
                        unit="batch",
                    ):
                        batch = chunk[i:i + batch_size]
                        pass_instance.process_batch(batch)
                else:
                    for essay in tqdm(
                        chunk,
                        desc=f"Pass: {pass_name} (chunk {chunk_start // chunk_size + 1})",
                        unit="essay",
                    ):
                        pass_instance.process_essay(essay)

                # Incremental save after each chunk
                if self.checkpoint_dir:
                    save_checkpoint_incremental(
                        chunk, pass_name, self.checkpoint_dir
                    )

            elapsed = time.time() - start
            eps = len(remaining) / elapsed if elapsed > 0 else 0
            logger.info(
                f"Pass '{pass_name}': completed in {elapsed:.1f}s "
                f"({eps:.1f} essays/sec)"
            )

            # Unload model to free resources
            pass_instance.unload_model()

    def compute_features(self, essays: list[Essay]) -> None:
        """Run enabled feature modules and filter to enabled features"""
        active_modules = [
            (name, fn) for name, fn in FEATURE_MODULES
            if name in self.required_modules
        ]
        skipped = [name for name, _ in FEATURE_MODULES if name not in self.required_modules]
        if skipped:
            logger.info(f"Skipping feature modules (no enabled features): {skipped}")

        logger.info(f"Computing features for {len(essays)} essays...")

        for essay in tqdm(essays, desc="Feature extraction", unit="essay"):
            all_features = {}
            for module_name, compute_fn in active_modules:
                try:
                    all_features.update(compute_fn(essay))
                except Exception as e:
                    logger.error(
                        f"Feature module '{module_name}' failed for "
                        f"essay {essay.essay_id}: {e}"
                    )
            # Filter to only enabled features
            essay.features = {
                k: v for k, v in all_features.items()
                if k in self.enabled_features
            }

    def run(self, essays: list[Essay], force: bool = False) -> list[Essay]:
        """Execute the full pipeline: passes → features."""
        self.run_passes(essays, force=force)
        self.compute_features(essays)

        if essays:
            n_features = len(essays[0].features)
            logger.info(f"Extraction complete: {n_features} features per essay")

        return essays
