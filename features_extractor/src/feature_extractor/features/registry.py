"""central feature registry"""

from dataclasses import dataclass

@dataclass
class FeatureSpec:
    name: str
    module: str
    description: str
    default_enabled: bool

_DEFAULT_ENABLED = {
    "agg_errors_per_sentence_mean",
    "agg_num_sentences",
    "agg_words_per_sentence",
    "dep_fragment_count",
    "dep_mdd_advcl",
    "dep_mdd_amod",
    "dep_mdd_max",
    "dep_mdd_prep",
    "dep_mdd_relcl",
    "dep_mdd_xcomp",
    "dep_tree_depth_max",
    "dep_tree_depth_ratio_mean",
    "err_cat_typos",
    "err_global_density",
    "err_local_density",
    "err_medium_density",
    "punct_comma_splice_ratio",
    "punct_to_clause_ratio",
    "spell_content_error_ratio",
    "spell_function_word_errors",
    "spell_misspelled_ratio",
    "syn_clauses_per_sentence_max",
    "syn_clauses_per_sentence_mean",
    "syn_long_high_elaboration_count",
    "syn_phrasal_clausal_ratio_mean",
    "syn_phrasal_elaboration_mean",
    "syn_phrasal_elaboration_stdev",
    "syn_sentence_length_stdev",
}

FEATURE_REGISTRY: dict[str, FeatureSpec] = {
      # --- Dependency ---
      "dep_mdd_mean": FeatureSpec("dep_mdd_mean", "dependency", "Mean dependency distance across all dependencies", False),
      "dep_mdd_stdev": FeatureSpec("dep_mdd_stdev", "dependency", "Standard deviation of dependency distances", False),
      "dep_mdd_max": FeatureSpec("dep_mdd_max", "dependency", "Maximum per-sentence mean dependency distance", True),
      "dep_mdd_advcl": FeatureSpec("dep_mdd_advcl", "dependency", "Mean dependency distance for adverbial clause relations", True),
      "dep_mdd_relcl": FeatureSpec("dep_mdd_relcl", "dependency", "Mean dependency distance for relative clause relations", True),
      "dep_mdd_ccomp": FeatureSpec("dep_mdd_ccomp", "dependency", "Mean dependency distance for clausal complement relations", False),
      "dep_mdd_xcomp": FeatureSpec("dep_mdd_xcomp", "dependency", "Mean dependency distance for open clausal complement relations", True),
      "dep_mdd_conj": FeatureSpec("dep_mdd_conj", "dependency", "Mean dependency distance for conjunct relations", False),
      "dep_mdd_amod": FeatureSpec("dep_mdd_amod", "dependency", "Mean dependency distance for adjectival modifier relations", True),
      "dep_mdd_prep": FeatureSpec("dep_mdd_prep", "dependency", "Mean dependency distance for prepositional relations", True),
      "dep_adjacent_prop_mean": FeatureSpec("dep_adjacent_prop_mean", "dependency", "Mean proportion of adjacent dependencies per sentence", False),
      "dep_max_arc_mean": FeatureSpec("dep_max_arc_mean", "dependency", "Mean maximum arc length per sentence", False),
      "dep_max_arc_max": FeatureSpec("dep_max_arc_max", "dependency", "Maximum arc length across all sentences", False),
      "dep_tree_depth_mean": FeatureSpec("dep_tree_depth_mean", "dependency", "Mean parse tree depth", False),
      "dep_tree_depth_max": FeatureSpec("dep_tree_depth_max", "dependency", "Maximum parse tree depth", True),
      "dep_tree_depth_ratio_mean": FeatureSpec("dep_tree_depth_ratio_mean", "dependency", "Mean tree depth to sentence length ratio", True),
      "dep_fragment_count": FeatureSpec("dep_fragment_count", "dependency", "Sentence fragments detected via dependency parse (no root verb)", True),
      "dep_fragment_ratio": FeatureSpec("dep_fragment_ratio", "dependency", "Proportion of sentences that are fragments", False),
      "dep_lt_fragment_count": FeatureSpec("dep_lt_fragment_count", "dependency", "LanguageTool-detected fragment count", False),
      "dep_lt_run_on_count": FeatureSpec("dep_lt_run_on_count", "dependency", "LanguageTool-detected run-on sentence count", False),
      # --- Syntactic Complexity ---
      "syn_sentence_length_mean": FeatureSpec("syn_sentence_length_mean", "syntactic_complexity", "Mean sentence length in words", False),
      "syn_sentence_length_stdev": FeatureSpec("syn_sentence_length_stdev", "syntactic_complexity", "Standard deviation of sentence length", True),
      "syn_sentence_length_max": FeatureSpec("syn_sentence_length_max", "syntactic_complexity", "Maximum sentence length in words", False),
      "syn_clauses_per_sentence_mean": FeatureSpec("syn_clauses_per_sentence_mean", "syntactic_complexity", "Mean number of clauses per sentence", True),
      "syn_clauses_per_sentence_max": FeatureSpec("syn_clauses_per_sentence_max", "syntactic_complexity", "Maximum clauses in a single sentence", True),
      "syn_phrasal_elaboration_mean": FeatureSpec("syn_phrasal_elaboration_mean", "syntactic_complexity", "Mean phrasal elaboration score per sentence", True),
      "syn_phrasal_elaboration_stdev": FeatureSpec("syn_phrasal_elaboration_stdev", "syntactic_complexity", "Standard deviation of phrasal elaboration", True),
      "syn_phrasal_clausal_ratio_mean": FeatureSpec("syn_phrasal_clausal_ratio_mean", "syntactic_complexity", "Mean ratio of phrasal to clausal complexity", True),
      "syn_long_low_elaboration_count": FeatureSpec("syn_long_low_elaboration_count", "syntactic_complexity", "Long sentences with low phrasal elaboration", False),
      "syn_long_high_elaboration_count": FeatureSpec("syn_long_high_elaboration_count", "syntactic_complexity", "Long sentences with high phrasal elaboration", True),
      "syn_sentence_length_cv": FeatureSpec("syn_sentence_length_cv", "syntactic_complexity", "Coefficient of variation of sentence length", False),
      # --- Punctuation ---
      "punct_density": FeatureSpec("punct_density", "punctuation", "Punctuation marks per word", False),
      "punct_to_clause_ratio": FeatureSpec("punct_to_clause_ratio", "punctuation", "Punctuation marks per clause", True),
      "punct_comma_splice_candidates": FeatureSpec("punct_comma_splice_candidates", "punctuation", "Candidate comma splice occurrences", False),
      "punct_comma_splice_ratio": FeatureSpec("punct_comma_splice_ratio", "punctuation", "Proportion of sentences with potential comma splices", True),
      "punct_length_outlier_count": FeatureSpec("punct_length_outlier_count", "punctuation", "Sentence length outlier count", False),
      "punct_length_outlier_ratio": FeatureSpec("punct_length_outlier_ratio", "punctuation", "Proportion of sentence length outliers", False),
      "punct_low_punct_long_count": FeatureSpec("punct_low_punct_long_count", "punctuation", "Long sentences with insufficient punctuation", False),
      "punct_low_punct_long_ratio": FeatureSpec("punct_low_punct_long_ratio", "punctuation", "Proportion of long sentences with low punctuation", False),
      # --- Error Severity ---
      "err_total_count": FeatureSpec("err_total_count", "error_severity", "Total LanguageTool error count", False),
      "err_total_density": FeatureSpec("err_total_density", "error_severity", "Total errors per word", False),
      "err_global_count": FeatureSpec("err_global_count", "error_severity", "Global (discourse-level) error count", False),
      "err_medium_count": FeatureSpec("err_medium_count", "error_severity", "Medium-severity error count", False),
      "err_local_count": FeatureSpec("err_local_count", "error_severity", "Local (word-level) error count", False),
      "err_global_density": FeatureSpec("err_global_density", "error_severity", "Global errors per word", True),
      "err_medium_density": FeatureSpec("err_medium_density", "error_severity", "Medium-severity errors per word", True),
      "err_local_density": FeatureSpec("err_local_density", "error_severity", "Local errors per word", True),
      "err_weighted_density": FeatureSpec("err_weighted_density", "error_severity", "Severity-weighted errors per word", False),
      "err_global_ratio": FeatureSpec("err_global_ratio", "error_severity", "Proportion of errors that are global", False),
      "err_cat_grammar": FeatureSpec("err_cat_grammar", "error_severity", "Grammar category error count", False),
      "err_cat_typos": FeatureSpec("err_cat_typos", "error_severity", "Typographical error count", True),
      "err_cat_punctuation": FeatureSpec("err_cat_punctuation", "error_severity", "Punctuation category error count", False),
      "err_cat_casing": FeatureSpec("err_cat_casing", "error_severity", "Capitalization error count", False),
      "err_cat_confused_words": FeatureSpec("err_cat_confused_words", "error_severity", "Confused/misused word error count", False),
      "err_cat_style": FeatureSpec("err_cat_style", "error_severity", "Style issue count", False),
      "err_cat_typography": FeatureSpec("err_cat_typography", "error_severity", "Typography formatting error count", False),
      "err_confused_words_count": FeatureSpec("err_confused_words_count", "error_severity", "Confused word count (LanguageTool)", False),
      "err_confused_words_density": FeatureSpec("err_confused_words_density", "error_severity", "Confused words per word", False),
      # --- Spelling ---
      "spell_misspelled_count": FeatureSpec("spell_misspelled_count", "spelling", "Total misspelled word count", False),
      "spell_misspelled_ratio": FeatureSpec("spell_misspelled_ratio", "spelling", "Proportion of misspelled words", True),
      "spell_content_word_errors": FeatureSpec("spell_content_word_errors", "spelling", "Misspelled content words", False),
      "spell_function_word_errors": FeatureSpec("spell_function_word_errors", "spelling", "Misspelled function words", True),
      "spell_content_error_ratio": FeatureSpec("spell_content_error_ratio", "spelling", "Content word misspelling rate", True),
      "spell_close_match_ratio": FeatureSpec("spell_close_match_ratio", "spelling", "Proportion of misspellings with close dictionary match", False),
      "spell_far_misspelling_count": FeatureSpec("spell_far_misspelling_count", "spelling", "Misspellings with no close dictionary match", False),
      "spell_far_misspelling_ratio": FeatureSpec("spell_far_misspelling_ratio", "spelling", "Proportion of misspellings far from dictionary", False),
      # --- Aggregation ---
      "agg_num_sentences": FeatureSpec("agg_num_sentences", "aggregation", "Total sentence count", True),
      "agg_total_words": FeatureSpec("agg_total_words", "aggregation", "Total word count", False),
      "agg_words_per_sentence": FeatureSpec("agg_words_per_sentence", "aggregation", "Mean words per sentence", True),
      "agg_errors_per_sentence_mean": FeatureSpec("agg_errors_per_sentence_mean", "aggregation", "Mean errors per sentence", True),
      "agg_errors_per_sentence_max": FeatureSpec("agg_errors_per_sentence_max", "aggregation", "Maximum errors in a single sentence", False),
      "agg_errors_per_sentence_stdev": FeatureSpec("agg_errors_per_sentence_stdev", "aggregation", "Standard deviation of per-sentence error count", False),
      "agg_error_free_sentence_ratio": FeatureSpec("agg_error_free_sentence_ratio", "aggregation", "Proportion of error-free sentences", False),
      "agg_worst_window_3_errors": FeatureSpec("agg_worst_window_3_errors", "aggregation", "Maximum total errors in any 3-sentence window", False),
      "agg_worst_window_3_mean": FeatureSpec("agg_worst_window_3_mean", "aggregation", "Maximum mean errors in any 3-sentence window", False),
      "agg_mdd_consistency": FeatureSpec("agg_mdd_consistency", "aggregation", "Consistency of dependency distance across sentences", False),
      "agg_error_concentration": FeatureSpec("agg_error_concentration", "aggregation", "Concentration of errors across sentences (Gini coefficient)", False),
  }

def get_enabled_features(config_overrides: dict[str, bool] | None = None) -> set[str]:
    """Return the enabled feature names after applying overrides"""
    enabled = {
        name for name, spec in FEATURE_REGISTRY.items() if spec.default_enabled
    }
    if config_overrides:
        for name, is_enabled in config_overrides.items():
            if name not in FEATURE_REGISTRY:
                raise ValueError(f"Unknown feature in config override: {name}")
            if is_enabled:
                enabled.add(name)
            else:
                enabled.discard(name)
    return enabled


def get_required_modules(enabled_features: set[str]) -> set[str]:
    """Return the module names that have at least one enabled feature"""
    return {
        FEATURE_REGISTRY[name].module
        for name in enabled_features
        if name in FEATURE_REGISTRY
    }
