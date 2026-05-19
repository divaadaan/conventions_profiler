"""Sentence boundary error features.

Composite "structural integrity" rate: missing terminal punctuation, comma
splices, fragments, and run-ons per sentence, plus per-sentence-type error
rates attributing errors to simple vs complex constructions.
"""

from feature_extractor.features._helpers import distribute_errors_to_sentences
from feature_extractor.models import Essay

# Also referenced in err_severity GLOBAL_RULES.
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

    # Positional/lexical heuristic — `en_core_web_sm` parses comma-spliced
    # clauses with inconsistent dep types (ccomp/xcomp instead of conj/parataxis).
    # SCONJ guard avoids flagging "Although X, Y".
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

    type_totals: dict[str, int] = {"simple": 0, "compound": 0, "complex": 0, "compound_complex": 0}
    type_errors: dict[str, int] = {"simple": 0, "compound": 0, "complex": 0, "compound_complex": 0}
    for sent, err_count in zip(sentences, errors_per_sent):
        st = sent.get("sentence_type", "simple")
        if st in type_totals:
            type_totals[st] += 1
            if err_count > 0:
                type_errors[st] += 1

    missing_term_punct_rate = missing_term_punct / n
    simple_error_rate = (
        type_errors["simple"] / type_totals["simple"] if type_totals["simple"] else 0.0
    )
    complex_error_rate = (
        type_errors["complex"] / type_totals["complex"] if type_totals["complex"] else 0.0
    )
    boundary_error_rate = (
        missing_term_punct + fragments + boundary_lt_count + comma_splice_heuristic
    ) / n

    return {
        "bound_missing_term_punct_count": float(missing_term_punct),
        "bound_missing_term_punct_rate": missing_term_punct_rate,
        "bound_boundary_error_rate": boundary_error_rate,
        "err_simple_sent_error_rate": simple_error_rate,
        "err_complex_sent_error_rate": complex_error_rate,
        "bound_terminal_punct_correct_rate": 1.0 - missing_term_punct_rate,
        "err_simple_sent_correct_rate": 1.0 - simple_error_rate,
        "err_complex_sent_correct_rate": 1.0 - complex_error_rate,
        # boundary_error_rate can exceed 1 (multiple errors per sentence), clamp before flipping.
        "bound_boundary_correct_rate": max(
            0.0, 1.0 - min(boundary_error_rate, 1.0)
        ),
    }


def _empty_features() -> dict[str, float]:
    return {
        "bound_missing_term_punct_count": 0.0,
        "bound_missing_term_punct_rate": 0.0,
        "bound_boundary_error_rate": 0.0,
        "err_simple_sent_error_rate": 0.0,
        "err_complex_sent_error_rate": 0.0,
        "bound_terminal_punct_correct_rate": 1.0,
        "err_simple_sent_correct_rate": 1.0,
        "err_complex_sent_correct_rate": 1.0,
        "bound_boundary_correct_rate": 1.0,
    }
