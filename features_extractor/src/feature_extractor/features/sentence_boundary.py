"""Sentence boundary error features.

Captures the expert-scoring "structural integrity" signal: composite of
missing terminal punctuation, comma splices, fragments, and run-ons,
normalized by sentence count. Plus per-sentence-type error rates that
attribute errors to simple vs complex constructions.
"""

from feature_extractor.features._helpers import distribute_errors_to_sentences
from feature_extractor.models import Essay

# LanguageTool rule IDs that signal a sentence-boundary failure (comma splice
# / run-on style). Same rules feed err_severity's GLOBAL_RULES set.
_BOUNDARY_LT_RULES = {
    "COMMA_COMPOUND_SENTENCE",
    "COMMA_COMPOUND_SENTENCE_2",
}


def compute_sentence_boundary_features(essay: Essay) -> dict[str, float]:
    spacy_data = essay.pass_results.get("spacy", {})
    lt_data = essay.pass_results.get("languagetool", {})
    sentences = spacy_data.get("sentences", [])
    matches = lt_data.get("matches", [])

    if not sentences:
        return _empty_features()

    n = len(sentences)

    missing_term_punct = sum(1 for s in sentences if not s.get("has_terminal_punct", True))
    fragments = sum(1 for s in sentences if s.get("has_verb_root") and s.get("nsubj_count", 0) == 0)
    boundary_lt_count = sum(1 for m in matches if m.get("rule_id") in _BOUNDARY_LT_RULES)

    # Comma-splice heuristic. A sentence is flagged as a comma splice when:
    #   - it has 2+ verb-with-subject clauses (clauses_with_subjects >= 2)
    #   - it contains at least one comma
    #   - it has no `cc` dep (no coordinating conjunction)
    #   - it has no SCONJ token (no subordinator like "Although"/"Because"/"While")
    # This is positional/lexical rather than dep-type-based because
    # `en_core_web_sm` sometimes parses comma-spliced clauses with weird dep
    # types (ccomp, xcomp) instead of conj/parataxis. The SCONJ check guards
    # against false positives on legitimate "Although X, Y" complex sentences.
    comma_splice_heuristic = 0
    for sent in sentences:
        if sent.get("clauses_with_subjects", 0) < 2:
            continue
        tokens = sent.get("tokens", [])
        if not any(t["text"] == "," for t in tokens):
            continue
        if any(t.get("pos") == "SCONJ" for t in tokens):
            continue
        deps_in_sent = [d.get("dep") for d in sent.get("dep_triples", [])]
        if "cc" in deps_in_sent:
            continue
        comma_splice_heuristic += 1

    errors_per_sent = distribute_errors_to_sentences(sentences, matches)

    # Per-sentence-type counts and error counts
    type_totals: dict[str, int] = {"simple": 0, "compound": 0, "complex": 0, "compound_complex": 0}
    type_errors: dict[str, int] = {"simple": 0, "compound": 0, "complex": 0, "compound_complex": 0}
    for sent, err_count in zip(sentences, errors_per_sent):
        st = sent.get("sentence_type", "simple")
        if st in type_totals:
            type_totals[st] += 1
            if err_count > 0:
                type_errors[st] += 1

    return {
        # Boundary components
        "bound_missing_term_punct_count": float(missing_term_punct),
        "bound_missing_term_punct_rate": missing_term_punct / n,
        # Composite: structural integrity signal (negative — flipped to positive in aggregation)
        "bound_boundary_error_rate": (
            missing_term_punct + fragments + boundary_lt_count + comma_splice_heuristic
        ) / n,
        # Per-type error rates
        "err_simple_sent_error_rate": (
            type_errors["simple"] / type_totals["simple"] if type_totals["simple"] else 0.0
        ),
        "err_complex_sent_error_rate": (
            type_errors["complex"] / type_totals["complex"] if type_totals["complex"] else 0.0
        ),
    }


def _empty_features() -> dict[str, float]:
    return {
        "bound_missing_term_punct_count": 0.0,
        "bound_missing_term_punct_rate": 0.0,
        "bound_boundary_error_rate": 0.0,
        "err_simple_sent_error_rate": 0.0,
        "err_complex_sent_error_rate": 0.0,
    }
