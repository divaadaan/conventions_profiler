"""
Main entry point for the conventions profiler pipeline.

Usage:
    python -m conventions_profiler.main --config config/default.yaml
    python -m conventions_profiler.main --config config/default.yaml --force
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from conventions_profiler.config.schemas import load_config
from conventions_profiler.data.loaders.asap_pp import load_asap_pp
from conventions_profiler.pipeline.cache import PipelineCache
from conventions_profiler.pipeline.orchestrator import Orchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def essays_to_dataframe(essays) -> pd.DataFrame:
    """Convert processed essays to a feature matrix DataFrame."""
    rows = []
    for essay in essays:
        row = {
            "essay_id": essay.essay_id,
            "prompt_id": essay.prompt_id,
            "conventions_score": essay.conventions_score,
            "conventions_z": essay.conventions_z,
            "word_count": essay.word_count,
        }
        row.update(essay.features)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Conventions Profiler Pipeline")
    parser.add_argument(
        "--config", type=str, default="config/default.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-extraction, ignoring cache",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for feature matrix (default: cache directory)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config_dict = config.model_dump()
    device = config.hardware.resolve_device()
    logger.info(f"Device: {device}")
    logger.info(f"spaCy model: {config.models.spacy.model}")

    # Load data
    logger.info("Loading ASAP++ data...")
    essays = load_asap_pp(
        essays_path=config.datasets.asap_pp.essays_path,
        annotations_path=config.datasets.asap_pp.annotations_path,
        prompts=config.datasets.asap_pp.prompts,
    )
    logger.info(f"Loaded {len(essays)} essays")

    # Set up pipeline
    cache = PipelineCache(
        cache_dir=config.cache.cache_path,
        enabled=config.cache.enabled,
    )
    orchestrator = Orchestrator(config=config_dict, cache=cache)

    # Run pipeline
    essays = orchestrator.run(essays, force=args.force)

    # Convert to DataFrame
    df = essays_to_dataframe(essays)
    logger.info(f"Feature matrix shape: {df.shape}")
    logger.info(f"Features: {sorted(df.columns.tolist())}")

    # Save
    output_path = args.output or str(config.cache.cache_path / "feature_matrix.parquet")
    df.to_parquet(output_path, index=False)
    logger.info(f"Feature matrix saved to {output_path}")

    # Also save as CSV for easy inspection
    csv_path = Path(output_path).with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Feature matrix (CSV) saved to {csv_path}")

    # Quick summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Essays processed: {len(essays)}")
    print(f"Features extracted: {len(df.columns) - 5}")  # subtract metadata cols
    print(f"Output: {output_path}")
    print("\nFeature summary:")
    feature_cols = [c for c in df.columns if c not in
                    {"essay_id", "prompt_id", "conventions_score", "conventions_z", "word_count"}]
    print(df[feature_cols].describe().T[["mean", "std", "min", "max"]].to_string())


if __name__ == "__main__":
    main()
