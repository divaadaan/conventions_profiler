# feature-extractor

Linguistic feature extraction pipeline for conventions scoring in automated essay scoring. Extracts psycholinguistically motivated features from student essay texts — dependency complexity, syntactic structure, punctuation usage, error severity, and spelling. Outputs a feature matrix suitable for scoring model training or cut-point derivation.

## Installation

Install PyTorch with CUDA separately before installing this package — it cannot be included in `requirements.txt` because it requires a CUDA-specific index URL.

Go to [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally), select your platform and CUDA version, and run the generated command inside your virtual environment. Example for CUDA 12.4:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Then install this package from the repo root and download the spaCy model:

```bash
pip install -e "features_extractor/[dev]"
python -m spacy download en_core_web_trf
```

LanguageTool requires a Java runtime (JRE 8+). The `language-tool-python` dependency handles downloading the LanguageTool server itself, but Java must be available on your `PATH`.

## Setting Up a New Dataset

Start by copying the default configuration to the `features_extractor` directory:

```bash
cp features_extractor/src/feature_extractor/config/default.yaml features_extractor/config.yaml
```

This is your working config file. The values in `config.yaml` override the defaults baked into the Pydantic schema (`config/schemas.py`), so anything you set here is the ground truth for pipeline behaviour. The sections below walk through what to adjust.

### Dataset and column mapping

The pipeline expects a CSV file with the following required columns:

| Column | Description |
|---|---|
| `essay_id` | Unique identifier per essay |
| `text` | The raw essay text |
| `score` | Numeric score (ground truth) |

An optional `prompt_id` column distinguishes prompt groups. If absent, all essays are treated as a single prompt. It is best to include this for the statistical analysis, this supports cross prompt feature validation.

Point `dataset.path` at your CSV. If your CSV uses different column names, map them under `dataset.columns`:

```yaml
dataset:
  path: "./data/my_dataset.csv"
  columns:
    essay_id: "id"
    text: "essay_text"
    score: "conventions_score"
    prompt_id: "task_id"
```

Only columns that differ from the defaults (`essay_id`, `text`, `score`, `prompt_id`) need to be specified.

### Scoring

Scores can be discrete labels (1, 2, 3, ...) or continuous values. If you want the pipeline to min-max normalise scores to a 0-1 range, set the bounds and enable it:

```yaml
scoring:
  score_min: 1.0
  score_max: 6.0
  normalise: true
```

If `normalise` is `false` (the default), scores pass through as-is. Either way, the analysis module's `discrete_scores` parameter independently controls whether cut-point derivation uses quartile bands or discrete score levels.

### Hardware

```yaml
hardware:
  device: "auto"   # "auto" | "cuda:0" | "cpu"
  batch_size: 16
```

`auto` selects CUDA when available and falls back to CPU. The batch size controls how many essays are passed to `nlp.pipe()` in the spaCy pass. Reduce it if you run into GPU memory issues.

### Models

```yaml
models:
  spacy:
    model: "en_core_web_trf"   # use "en_core_web_lg" for CPU-only environments
  languagetool:
    server_port: 8081
    language: "en-CA"
```

The transformer model (`en_core_web_trf`) requires a GPU. If running on CPU, switch to `en_core_web_lg` and download it with `python -m spacy download en_core_web_lg`. This is American English, as spaCy does not have a Canadian English model. languageTool is configured to use Canadian English by default (models.languagetool.language).

### Caching and checkpoints

```yaml
cache:
  enabled: true
  directory: "./cache"
  format: "parquet"
  checkpoint_chunk_size: 500
```

The pipeline saves incremental checkpoints after every `checkpoint_chunk_size` essays. If a run is interrupted, it resumes from the last completed chunk automatically. Set `enabled: false` to disable caching entirely, or change the directory if you prefer a different location.

### Feature overrides

```yaml
features:
  overrides: {}
```

The feature registry (`features/registry.py`) defines the full set of available features and which are enabled by default (~28). The `overrides` dict in your config is where you toggle individual features on or off. These overrides take precedence over the registry defaults — if a feature appears in `overrides`, that value wins. See [Selecting Features](#selecting-features) for details.

## Running the Pipeline

Run extraction from the `features_extractor` directory:

```bash
cd features_extractor
feature-extractor --config config.yaml
```

Useful flags:

- `--force` — ignore checkpoints and re-run all passes from scratch.
- `--limit N` — process only the first N essays.
- `--output PATH` — write the feature matrix to a specific path instead of the default `cache/feature_matrix.parquet`.
- `--all-features` — extract every feature in the registry, ignoring default enabled/disabled settings.
- `--analyze` — run validation and cut-point analysis after extraction (see [Validation and Analysis](#validation-and-analysis)).
- `--consistency FLOAT` — fraction of prompts a feature must be significant in to pass cross-prompt filtering (default: `0.75`).

Output is written to the cache directory as both `feature_matrix.parquet` and `feature_matrix.csv`. Intermediate checkpoints are saved to `cache/checkpoints/` and are used to resume interrupted runs automatically.

## Selecting Features

The feature registry (`features/registry.py`) defines all available features. 28 features derived from running the pipeline on ASAP++ are enabled by default. To toggle specific features, use the `features.overrides` config key:

```yaml
features:
  overrides:
    dep_mdd_mean: true          # enable a non-default feature
    spell_misspelled_ratio: false  # disable a default feature
```

Features are grouped into six modules: `dependency`, `syntactic_complexity`, `punctuation`, `error_severity`, `spelling`, and `aggregation`. Only modules with at least one enabled feature are executed during the feature computation stage.

## Validation and Analysis

Analysis is a separate step that runs after extraction. Use the `--analyze` flag to include it:

```bash
feature-extractor --config config.yaml --analyze
```

The `--consistency` flag controls how strict cross-prompt filtering is. With 6 prompts (e.g. ASAP++), 0.75 requires significance in 5/6. With 44 prompts, it requires 33/44. Raise it for stricter filtering or lower it to retain more features:

```bash
feature-extractor --config config.yaml --analyze --consistency 0.80
```

Alternatively, run it from the notebook at `analysis/analysis.ipynb`, or call the runner directly:

```python
from feature_extractor.analysis.runner import run_validation_pipeline

results = run_validation_pipeline(
    feature_matrix_path="cache/feature_matrix.parquet",
    output_dir="cache/results",
    corr_threshold=0.85,
    consistency_threshold=0.75,
    discrete_scores=False,
)
```

The analysis runs four steps:

1. **Per-prompt validation** — Spearman rank correlation and Mann-Whitney U tests (with Benjamini-Hochberg FDR correction) for each feature against the target score.
2. **Cross-prompt ranking** — ranks features by significance consistency and effect direction across prompts. A feature must be significant in at least `consistency_threshold` fraction of prompts (default 75%) with a consistent sign to pass.
3. **Correlation pruning** — removes redundant features within each module using pairwise Pearson correlation above the threshold, keeping the feature with the stronger cross-prompt signal.
4. **Cut-point derivation** — computes feature thresholds from band medians (quartile-based or discrete scoring), producing the final cut-point table.

Results are exported as CSVs: per-prompt Spearman/MWU tables, cross-prompt ranking, correlation pruning log, and the final `feature_cuts.csv`.

## Architecture

The pipeline has two main stages coordinated by the `Orchestrator`:

### Processing passes

Two NLP passes run sequentially over the essay corpus, each storing structured results in `essay.pass_results`:

- **SpacyPass** — loads a spaCy transformer model (`en_core_web_trf` by default) and parses each essay. Pre-computes sentence-level dependency structures, tree depth, clause counts, mean dependency distance, and phrasal elaboration metrics. Supports batched inference via `nlp.pipe()`.
- **LanguageToolPass** — starts a local LanguageTool server and detects grammar, spelling, and style errors. Stores match details and error category counts.

Each pass loads its model into memory, processes all essays, then unloads. Passes are checkpointed incrementally to parquet — if the pipeline is interrupted, it resumes from the last completed chunk rather than reprocessing from the start.

### Feature computation

After both passes complete, enabled feature modules run as pure functions over the stored pass results. Each module reads from `essay.pass_results` and writes computed features to `essay.features`. The orchestrator filters the output to only the features enabled in the registry (after applying config overrides).

Six feature modules are available:

- **dependency** — mean dependency distance (overall and per-relation), tree depth, fragment detection.
- **syntactic_complexity** — sentence length variation, clause density, phrasal elaboration, phrasal-to-clausal ratio.
- **punctuation** — punctuation density, comma splice detection, punctuation-to-clause ratio.
- **error_severity** — LanguageTool error counts and density by severity tier (global, medium, local) and by category.
- **spelling** — misspelling rates, content vs. function word errors, edit distance classification.
- **aggregation** — document-level summaries: sentence count, words per sentence, errors per sentence, sliding window error concentration.

### Configuration

All settings are defined as Pydantic models in `config/schemas.py` and loaded from a YAML file. The config controls hardware (device, batch size), model selection, caching behavior, dataset path and column mapping, feature overrides, and score normalization.
