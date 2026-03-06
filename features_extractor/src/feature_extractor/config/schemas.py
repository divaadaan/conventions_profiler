"""
Configuration schema and loader for the feature extractor pipeline.

Loads from YAML, validates with pydantic, and resolves hardware settings.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class HardwareConfig(BaseModel):
    device: str = "auto"
    batch_size: int = 16

    def resolve_device(self) -> str:
        if self.device == "auto":
            try:
                import torch
                return "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device


class SpacyModelConfig(BaseModel):
    model: str = "en_core_web_trf"


class LanguageToolConfig(BaseModel):
    server_port: int = 8081
    language: str = "en-US"


class ModelsConfig(BaseModel):
    spacy: SpacyModelConfig = Field(default_factory=SpacyModelConfig)
    languagetool: LanguageToolConfig = Field(default_factory=LanguageToolConfig)


class CacheConfig(BaseModel):
    enabled: bool = True
    directory: str = "./cache"
    format: str = "parquet"
    checkpoint_chunk_size: int = 500

    @property
    def cache_path(self) -> Path:
        path = Path(self.directory)
        path.mkdir(parents=True, exist_ok=True)
        return path

class ScoringConfig(BaseModel):
    score_min: float = 0.0
    score_max: float = 1.0
    normalize: bool = False

class FeaturesConfig(BaseModel):
    overrides: dict[str, bool] = Field(default_factory=dict)


class ColumnMapping(BaseModel):
    essay_id: str = "essay_id"
    text: str = "text"
    prompt_id: str = "prompt_id"
    score: str = "score"


class DatasetConfig(BaseModel):
    path: str = "./src/feature_extractor/data/YOUR-PROCESSED-DATA-PATH.csv"
    columns: ColumnMapping = Field(default_factory=ColumnMapping)


class PipelineConfig(BaseModel):
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)


def load_config(config_path: str | Path = "config/default.yaml") -> PipelineConfig:
    """Load pipeline configuration from YAML."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return PipelineConfig(**raw)
