"""Lemma-based vocabulary features.

Keyed on lemmas so inflected variants ('run', 'runs', 'running', 'ran') count
as one type.
"""

from collections import Counter

from feature_extractor.models import Essay


def _alpha_lemmas(sentences: list[dict]) -> list[str]:
    out = []
    for sent in sentences:
        for tok in sent.get("tokens", []):
            if tok.get("is_alpha") and tok.get("lemma"):
                out.append(tok["lemma"])
    return out


def compute_lemma_vocab_features(essay: Essay) -> dict[str, float]:
    spacy_data = essay.pass_results.get("spacy", {})
    sentences = spacy_data.get("sentences", [])

    lemmas = _alpha_lemmas(sentences)
    if not lemmas:
        return _empty_features()

    counts = Counter(lemmas)
    hapax = sum(1 for c in counts.values() if c == 1)
    distinct = len(counts)
    total = len(lemmas)

    return {
        "vocab_unique_lemma_count": float(hapax),
        "vocab_lemma_diversity_ratio": distinct / total,
    }


def _empty_features() -> dict[str, float]:
    return {
        "vocab_unique_lemma_count": 0.0,
        "vocab_lemma_diversity_ratio": 0.0,
    }
