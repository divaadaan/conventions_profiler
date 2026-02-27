"""
Main entry point for the conventions profiler pipeline.

Usage:
    python -m conventions_profiler.main --config config/default.yaml
    python -m conventions_profiler.main --config config/default.yaml --force
    python -m conventions_profiler.main --limit 50
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from conventions_profiler.config.schemas import load_config
from conventions_profiler.data.loaders.csv_loader import load_dataset
from conventions_profiler.pipeline.orchestrator import Orchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


METADATA_COLS = {"essay_id", "prompt_id", "score", "word_count"}


def essays_to_dataframe(essays) -> pd.DataFrame:
    """Convert processed essays to DataFrame"""
    rows = []
    for essay in essays:
        row = {
            "essay_id": essay.essay_id,
            "prompt_id": essay.prompt_id,
            "score": essay.score,
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
        help="Force re-extraction, ignoring checkpoints",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N essays",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for feature matrix, default tp cache directory",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config_dict = config.model_dump()
    device = config.hardware.resolve_device()
    logger.info(f"Device: {device}")
    logger.info(f"spaCy model: {config.models.spacy.model}")

    logger.info("Loading data...")
    col_map = config.dataset.columns.model_dump()
    essays = load_dataset(path=config.dataset.path, columns=col_map)

    if args.limit is not None:
        essays = essays[:args.limit]
    logger.info(f"Loaded {len(essays)} essays")

    if config.scoring.normalize:
        low_score, high_score = config.scoring.score_min, config.scoring.score_max
        for essay in essays:
            if essay.score is not None:
                essay.score = (essay.score - low_score) / (high_score - low_score)

    checkpoint_dir = config.cache.cache_path / "checkpoints"
    orchestrator = Orchestrator(config=config_dict, checkpoint_dir=checkpoint_dir)

    essays = orchestrator.run(essays, force=args.force)

    df = essays_to_dataframe(essays)
    logger.info(f"Feature matrix shape: {df.shape}")
    logger.info(f"Features: {sorted(df.columns.tolist())}")

    # Save
    output_path = args.output or str(config.cache.cache_path / "feature_matrix.parquet")
    df.to_parquet(output_path, index=False)
    logger.info(f"Feature matrix saved to {output_path}")
    csv_path = Path(output_path).with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Feature matrix (CSV) saved to {csv_path}")

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Essays processed: {len(essays)}")
    print(f"Features extracted: {len(df.columns) - len(METADATA_COLS)}")
    print(f"Output: {output_path}")
    print("\nFeature summary:")
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    print(df[feature_cols].describe().T[["mean", "std", "min", "max"]].to_string())


if __name__ == "__main__":
    main()
