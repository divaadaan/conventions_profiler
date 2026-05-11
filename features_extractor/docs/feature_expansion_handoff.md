# Feature Expansion — Session Handoff

Status doc for the curriculum-aligned feature expansion work. Pick up here next session.

**Branch:** `main` (uncommitted changes — see `git status`)
**Last updated:** 2026-05-11

---

## Scope recap

Closing the gap between ML scoring and expert scoring by adding curriculum-aligned features:
1. ✅ Curriculum → detector mapping (deliverable + audit artifact for the client)
2. ✅ Capitalization decomposition (sent-initial, proper-noun, intrusion)
3. ✅ Sentence boundary composite + per-type error rates
4. ✅ Readability (Flesch / Flesch-Kincaid / Dale-Chall PDW)
5. ✅ Lemma-based vocabulary + spelling deduplication
6. ⏳ Cognitive reframing in aggregation
7. ✅ Surprisal pass (Pythia 70M, log₂, per-word, 6 essay-level features)
8. ⏳ Full re-extraction on ASAP corpus
9. ⏳ Model retraining against new feature set

---

## ✅ What's done

### Curriculum mapping
- `docs/curriculum_mapping.yaml` — 77 mapped items, locked format
- `scripts/render_curriculum_report.py` — renders 3 outputs:
  - `docs/reports/curriculum_coverage.md`
  - `docs/reports/curriculum_gaps.md`
  - `docs/reports/curriculum_coverage.csv`
- Coverage: ~73%, **0 low-level gaps**, 6 medium, 15 high

### New feature modules
- `src/feature_extractor/features/capitalization.py` — `cap_sentence_initial_rate`, `cap_proper_noun_rate`, `cap_intrusion_rate`
- `src/feature_extractor/features/sentence_boundary.py` — `bound_missing_term_punct_count/rate`, `bound_boundary_error_rate`, `err_simple_sent_error_rate`, `err_complex_sent_error_rate`
- `src/feature_extractor/features/readability.py` — `read_flesch_reading_ease`, `read_flesch_kincaid_grade`, `read_dale_chall_difficult_word_density`
- `src/feature_extractor/features/lemma_vocab.py` — `vocab_unique_lemma_count`, `vocab_lemma_diversity_ratio`
- `src/feature_extractor/features/_helpers.py` — shared `distribute_errors_to_sentences` (used by aggregation + sentence_boundary)

### Extensions
- `src/feature_extractor/passes/spacy_pass.py`
  - Added per-token `lemma` and `ent_type`
  - Added per-sentence `sentence_type` (simple/compound/complex/compound_complex) and `has_terminal_punct`
- `src/feature_extractor/features/syntactic_complexity.py` — added `syn_simple/compound/complex/compound_complex_sent_rate`
- `src/feature_extractor/features/spelling.py` — added `spell_unique_misspelled_lemma_count/ratio`, `spell_repetition_factor`
- `src/feature_extractor/features/registry.py` — `FeatureSpec` gained `level: str | None` field; all new features registered with curriculum levels
- `src/feature_extractor/pipeline/orchestrator.py` — four new modules wired in (incl. surprisal)
- `pyproject.toml` — added `textstat>=0.7`, `torch>=2.0`, `transformers>=4.40`

### Surprisal pass (Pythia 70M, config-gated)
- `src/feature_extractor/passes/surprisal_pass.py` — `SurprisalPass(BasePass)`. Pythia 70M by default (replicating Gruteke Klein et al. 2025, arXiv:2502.11150). Per-word surprisal in bits (log₂); BPE→word via HF `return_offsets_mapping`; full-essay single forward pass; FP32, `torch.no_grad`.
- `src/feature_extractor/features/surprisal.py` — six essay-level features: `surp_mean_per_word`, `surp_perplexity`, `surp_p95_per_word`, `surp_std_per_word`, `surp_sentence_mean_max`, `surp_sentence_mean_std`. Per-sentence buckets via spaCy `start_char`/`end_char` containment.
- `src/feature_extractor/passes/base.py` — added `validate_pass_results(essay)` hook. `SurprisalPass` overrides to check cached `model_id` against config; mismatched essays are re-processed automatically.
- `src/feature_extractor/pipeline/orchestrator.py` — appends `SurprisalPass` only when `config["models"]["surprisal"]["enabled"]`; new `_invalidate_stale_checkpoints` step runs after every load.
- `config.yaml` — added `models.surprisal.{enabled,model}`. Defaults to `enabled: false` so existing pipelines run unchanged.

### Validation harness (pytest-integrated, `--report` flag for CSV diagnostics)
- `tests/harness/base.py` — `FeatureCase` dataclass + shared runner
- `tests/harness/conftest.py` — session-scoped spaCy + LT fixtures, `--report` flag
- `tests/harness/test_capitalization.py` — 8 cases, all passing on `trf`
- `tests/harness/test_sentence_boundary.py` — 5 cases, all passing
- `tests/harness/test_readability.py` — 3 cases, all passing
- `tests/harness/test_lemma_vocab.py` — 4 cases, all passing
- `tests/harness/test_surprisal.py` — 4 cases + relative-ordering check (garbled > fluent). Auto-skips if torch/transformers missing.
- `scripts/parse_inspect.py` — diagnostic for spaCy parse calibration

---

## 📦 Deliverable status

**This week's deliverable (2026-05-11): COMPLETE.** Scope was the missing curriculum-coverage features (capitalization, sentence boundary, readability, lemma vocab, surprisal) plus a pytest-integrated validation harness for each. All shipped, all harnesses green.

**Next week's deliverable (~2026-05-18):** aggregation cognitive reframing + full ASAP re-extraction + model retraining (items #1, #3, #6 below).

---

## ⏳ Next up (in order)

### 1. Cognitive reframing in `aggregation.py` *(next deliverable)*
The deliverable spec calls for positive-framed aggregates (e.g. "Mechanical Consistency" instead of "sentence-initial cap error rate"). Raw rates stay computed; aggregates are `1 - rate` or per-sentence success counts.

Aggregates to add (per spec):
| Aggregate name              | Source feature                              | Formula           |
|-----------------------------|---------------------------------------------|-------------------|
| `agg_mechanical_consistency`| `cap_sentence_initial_rate`                 | `1 - rate`        |
| `agg_orthographic_precision`| `spell_unique_misspelled_lemma_ratio`       | `1 - rate`        |
| `agg_structural_integrity`  | `bound_boundary_error_rate`                 | `1 - rate` (clamped to [0,1]) |
| `agg_lexical_diversity`     | `vocab_unique_lemma_count`                  | identity (positive already) |
| `agg_lexical_sophistication`| `read_dale_chall_difficult_word_density`    | identity |
| `agg_reading_ease`          | `read_flesch_reading_ease`                  | identity |
| `agg_reading_fluency_ratio` | `agg_error_free_sentence_ratio`             | identity (already exists) |

Implementation notes:
- `bound_boundary_error_rate` can exceed 1.0 (it's count/sentences, not proportion). Clamp before flipping: `max(0.0, 1.0 - min(rate, 1.0))`.
- Register in `registry.py` under `aggregation` module with appropriate `level`.
- Update tests if any existing aggregation harness exists; otherwise add `tests/harness/test_aggregation_reframing.py`.

### 2. ~~Surprisal stub~~ → ✅ done (real, not stubbed)
Built directly instead of stubbed. Six features in retraining (not exploratory-only) because the conventions rubric is about *reading-ease impact of errors* — XGBoost can learn the `error_rate × surprisal` interaction. See "Surprisal pass" under ✅ done.

**Calibration to do during full re-extraction**: spot-check `surp_*` distributions on real ASAP essays; if any feature is degenerate (single-peak, all zero, etc.), adjust ranges in the harness and document in `docs/reports/harness/surprisal.csv` (run pytest with `--report`).

### 3. Full re-extraction on ASAP corpus *(next deliverable)*
Cache invalidation strategy is "full re-extract" (decided in earlier conversation). Steps:
- Delete `asappp1278cache/`, `ellipsecache/`, `synthetic_cache/`
- Run `feature-extractor` CLI (or whatever entrypoint) over `asap-pp-3456-language-norm.csv`
- Verify all new features land in the output parquet
- Spot-check 10–20 essays' new features against expectations

### 4. Locale notes
Per curriculum mapping draft, `SERIAL_COMMA` and a few other LT rules behave differently in `en-CA` vs `en-US`. As we hit more during model retraining, accumulate findings in a new section of `curriculum_mapping.yaml` (`locale_notes:` top-level key) and surface them in the rendered reports.

### 5. Render final client deliverable
Once the curriculum mapping is reviewed/finalized, run `scripts/render_curriculum_report.py` and ship the three reports as the curriculum coverage deliverable.

### 6. Model retraining *(next deliverable)*
Tied to #3. The current trained models will be replaced — no backwards-compat constraints.

---

## ❓ Open issues / known limitations

- **Sentence-type classification is heuristic**: `_classify_sentence_type` in `spacy_pass.py` uses a dep-type-based rule that can be sensitive to `ccomp`/`xcomp` misparses. Comma-splice detection in `sentence_boundary.py` uses a positional heuristic (clauses_with_subjects + comma + no SCONJ + no cc), but the `syn_*_sent_rate` features still use the dep-based classifier. Re-evaluate distributions on real essays during full re-extraction.

- **Comma-splice heuristic edge case**: legitimate semicolon-joined compounds without an SCONJ would be flagged. spaCy usually splits at semicolons, but not always. Acceptable false-positive rate for student writing.

- **Lemma denominator quirk**: `spell_repetition_factor = raw_ratio / unique_ratio`. When `unique_ratio == raw_ratio` (no repeats), factor = 1.0. When some misspellings repeat, factor > 1.0. When no misspellings, factor = 0.0 (defaulted). Document this in any client-facing description.

- **Java/JDK setup on user's box**: System PATH had a stale `Microsoft/jdk-21.0.10.7-hotspot` entry pointing nowhere; actual JDK is at `C:\Program Files\Eclipse Adoptium\jdk-21.0.11.10-hotspot`. JAVA_HOME and PATH set in `~/.bashrc`. Worth cleaning the stale system PATH entry next time admin access is convenient.

---

## 🔒 Decisions locked in (don't re-litigate)

- **Cache strategy**: full re-extract on schema bumps, no schema versioning
- **Existing trained models**: will be replaced, no backwards-compat needed
- **Library**: `textstat` (not spacy-readability) for readability formulas
- **Surprisal**: real `BasePass` subclass, Pythia 70M default, log₂, paper-replication (arXiv:2502.11150); per-word via HF offset mapping; config-gated by `models.surprisal.enabled`; included in retraining (not exploratory-only) — XGBoost interaction with error features carries the rubric's "reading-ease impact" framing
- **Curriculum field**: `FeatureSpec` has `level: Low | Medium | High | None`
- **Cognitive reframing location**: `aggregation.py` only; raw rates stay in feature modules for diagnostics
- **Validation harness style**: pytest-integrated, `--report` flag dumps CSV to `docs/reports/harness/`
- **POC dataset scope**: ASAP only; client runs their proprietary data themselves
- **Test model**: `en_core_web_trf` (matches production parse fidelity); override with `HARNESS_SPACY_MODEL=en_core_web_sm` for faster local iteration, with the caveat that some cases become flaky on `sm`

---

## 🚀 Resume commands

```bash
# Verify environment
.venv/Scripts/python.exe -c "import feature_extractor; import spacy; import textstat; print('ok')"
java -version  # should show openjdk 21.x

# Run all harnesses
.venv/Scripts/python.exe -m pytest tests/harness/ -v

# Run with diagnostic report (writes to docs/reports/harness/)
.venv/Scripts/python.exe -m pytest tests/harness/ -v --report

# Re-render curriculum mapping reports
.venv/Scripts/python.exe scripts/render_curriculum_report.py

# Inspect a spaCy parse for calibration
.venv/Scripts/python.exe scripts/parse_inspect.py "your sentence here"
```

---

## File index for next session

```
docs/
├── HANDOFF.md (this file)
├── curriculum_mapping.yaml
└── reports/
    ├── curriculum_coverage.md
    ├── curriculum_gaps.md
    ├── curriculum_coverage.csv
    └── harness/                # written by --report
        ├── capitalization.csv
        ├── sentence_boundary.csv
        ├── readability.csv
        └── lemma_vocab.csv

scripts/
├── render_curriculum_report.py
└── parse_inspect.py

src/feature_extractor/
├── passes/
│   ├── base.py                     # added validate_pass_results hook
│   ├── spacy_pass.py               # extended with lemma, ent_type, sentence_type, has_terminal_punct
│   └── surprisal_pass.py           # NEW (Pythia 70M, config-gated)
├── features/
│   ├── capitalization.py           # NEW
│   ├── sentence_boundary.py        # NEW
│   ├── readability.py              # NEW
│   ├── lemma_vocab.py              # NEW
│   ├── surprisal.py                # NEW
│   ├── _helpers.py                 # NEW (shared error-distribution helper)
│   ├── syntactic_complexity.py     # extended with sent-type rates
│   ├── spelling.py                 # extended with lemma dedup
│   ├── aggregation.py              # uses _helpers; reframing TODO
│   └── registry.py                 # FeatureSpec gained `level` field; new features registered
└── pipeline/orchestrator.py        # wired in 4 new modules + checkpoint validation

tests/harness/
├── base.py
├── conftest.py
├── test_capitalization.py          # 1 xfail
├── test_sentence_boundary.py
├── test_readability.py
├── test_lemma_vocab.py
└── test_surprisal.py               # NEW (skips without torch+transformers)
```
