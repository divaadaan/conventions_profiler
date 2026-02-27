"""Dependency parse features measuring structural interpretability, extracted from spacypass"""

import statistics

from conventions_profiler.models import Essay


def _safe_mean(values: list[float], default: float = 0.0) -> float:
    return statistics.mean(values) if values else default


def _safe_stdev(values: list[float], default: float = 0.0) -> float:
    return statistics.stdev(values) if len(values) >= 2 else default


def _safe_max(values: list[float], default: float = 0.0) -> float:
    return max(values) if values else default


def compute_dependency_features(essay: Essay) -> dict[str, float]:
    """Extract dependency-based features from spaCy pass results."""
    spacy_data = essay.pass_results.get("spacy", {})
    sentences = spacy_data.get("sentences", [])

    if not sentences:
        return _empty_features()

    features: dict[str, float] = {}

    # Mean Dependency Distance (MDD)
    sentence_mdds = [s["mdd"] for s in sentences]
    features["dep_mdd_mean"] = _safe_mean(sentence_mdds)
    features["dep_mdd_stdev"] = _safe_stdev(sentence_mdds)
    features["dep_mdd_max"] = _safe_max(sentence_mdds)

    # MDD by dependency type aggregated across sentences
    type_distances: dict[str, list[float]] = {}
    for sent in sentences:
        for dep_type, mdd_val in sent.get("mdd_by_type", {}).items():
            type_distances.setdefault(dep_type, []).append(mdd_val)

    # Key dependency types for conventions scoring
    for dep_type in ["advcl", "relcl", "ccomp", "xcomp", "conj", "amod", "prep"]:
        vals = type_distances.get(dep_type, [])
        features[f"dep_mdd_{dep_type}"] = _safe_mean(vals)

    # Adjacent dependency proportion
    adj_props = [s["adjacent_dep_proportion"] for s in sentences]
    features["dep_adjacent_prop_mean"] = _safe_mean(adj_props)

    # Max arc length
    max_arcs = [s["max_arc_length"] for s in sentences]
    features["dep_max_arc_mean"] = _safe_mean(max_arcs)
    features["dep_max_arc_max"] = _safe_max(max_arcs)

    # Tree depth
    depths = [s["tree_depth"] for s in sentences]
    features["dep_tree_depth_mean"] = _safe_mean(depths)
    features["dep_tree_depth_max"] = _safe_max(depths)

    # Tree depth normalized by sentence length
    depth_ratios = [
        s["tree_depth"] / s["word_count"] if s["word_count"] > 0 else 0.0
        for s in sentences
    ]
    features["dep_tree_depth_ratio_mean"] = _safe_mean(depth_ratios)

    # Fragment detection
    fragment_count = 0
    for sent in sentences:
        if sent["has_verb_root"] and sent["nsubj_count"] == 0:
            fragment_count += 1
    features["dep_fragment_count"] = float(fragment_count)
    features["dep_fragment_ratio"] = fragment_count / len(sentences) if sentences else 0.0

    # LanguageTool-based fragment and run-on counts
    lt_data = essay.pass_results.get("languagetool", {})
    lt_matches = lt_data.get("matches", [])

    fragment_rules = {
        "MISSING_TO_BEFORE_A_VERB", "PRP_PAST_PART", "MD_BASEFORM",
        "THE_CC", "DT_JJ_NO_NOUN", "A_INFINITIVE",
    }
    run_on_rules = {"COMMA_COMPOUND_SENTENCE", "COMMA_COMPOUND_SENTENCE_2"}

    features["dep_lt_fragment_count"] = float(
        sum(1 for m in lt_matches if m.get("rule_id") in fragment_rules)
    )
    features["dep_lt_run_on_count"] = float(
        sum(1 for m in lt_matches if m.get("rule_id") in run_on_rules)
    )

    return features


def _empty_features() -> dict[str, float]:
    """Return zeroed features when no sentences are available."""
    keys = [
        "dep_mdd_mean", "dep_mdd_stdev", "dep_mdd_max",
        "dep_mdd_advcl", "dep_mdd_relcl", "dep_mdd_ccomp", "dep_mdd_xcomp",
        "dep_mdd_conj", "dep_mdd_amod", "dep_mdd_prep",
        "dep_adjacent_prop_mean",
        "dep_max_arc_mean", "dep_max_arc_max",
        "dep_tree_depth_mean", "dep_tree_depth_max", "dep_tree_depth_ratio_mean",
        "dep_fragment_count", "dep_fragment_ratio",
        "dep_multi_root_count", "dep_multi_root_ratio",
        "dep_lt_fragment_count", "dep_lt_run_on_count",
    ]
    return {k: 0.0 for k in keys}
