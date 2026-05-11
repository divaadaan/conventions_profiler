"""SpaCy processing pass.
runs a nlp() call per essay and stores the parsed Doc.
pre-computes intermediate structures that the feature modules use"""

import logging
from typing import Any

import spacy
from spacy.tokens import Doc

from feature_extractor.models import Essay
from feature_extractor.passes.base import BasePass

logger = logging.getLogger(__name__)

# Subordinate-clause dep relations used for sentence-type classification.
_SUBORD_DEPS = {"advcl", "relcl", "ccomp", "xcomp", "acl", "csubj", "csubjpass"}


def _classify_sentence_type(sent_tokens) -> str:
    """Classify a sentence span as simple / compound / complex / compound_complex.

    Heuristic:
      - subordinate clause = any dep in _SUBORD_DEPS
      - coordinated independent clause = a 'conj' dep where the conjoined
        token is a verb/aux with its own subject

    Note: this classifier is dep-type-based and may misclassify comma-spliced
    sentences (en_core_web_sm sometimes assigns weird dep types like ccomp).
    Comma splice detection lives in sentence_boundary.py and uses a more
    robust positional heuristic that does not depend on this classification.
    """
    has_subord = False
    has_coord_clause = False
    for token in sent_tokens:
        if token.dep_ in _SUBORD_DEPS:
            has_subord = True
        if token.dep_ == "conj" and token.pos_ in ("VERB", "AUX"):
            has_subj = any(
                child.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass")
                for child in token.children
            )
            if has_subj:
                has_coord_clause = True
    if has_subord and has_coord_clause:
        return "compound_complex"
    if has_coord_clause:
        return "compound"
    if has_subord:
        return "complex"
    return "simple"


def _has_terminal_punct(sent_text: str) -> bool:
    s = sent_text.rstrip()
    if not s:
        return False
    return s[-1] in ".!?"


def _compute_sentence_data(sent) -> dict[str, Any]:
    """Extract intermediate data from a single spaCy parse. """
    tokens = list(sent)
    deps = []
    nsubj_count = 0
    root_count = 0
    punct_count = 0
    max_arc_length = 0

    for token in tokens:
        if token.dep_ == "ROOT":
            root_count += 1

        if token.dep_ == "nsubj":
            nsubj_count += 1

        if token.pos_ == "PUNCT":
            punct_count += 1

        # Dependency arc: distance between token and its head
        if token.dep_ != "ROOT":
            arc_length = abs(token.i - token.head.i)
            deps.append({
                "token_i": token.i,
                "head_i": token.head.i,
                "dep": token.dep_,
                "arc_length": arc_length,
            })
            max_arc_length = max(max_arc_length, arc_length)

    # Mean dependency distance for this sentence
    arc_lengths = [d["arc_length"] for d in deps]
    mdd = sum(arc_lengths) / len(arc_lengths) if arc_lengths else 0.0

    # Proportion of adjacent dependencies (arc length == 1)
    adjacent_count = sum(1 for a in arc_lengths if a == 1)
    adjacent_prop = adjacent_count / len(arc_lengths) if arc_lengths else 0.0

    # Tree depth via recursive traversal
    def _tree_depth(token, depth=0):
        children_depths = [_tree_depth(child, depth + 1) for child in token.children]
        return max(children_depths) if children_depths else depth

    root_tokens = [t for t in tokens if t.dep_ == "ROOT"]
    tree_depth = max((_tree_depth(r) for r in root_tokens), default=0)

    # Verb groups with subjects (clause approximation)
    verbs_with_subjects = 0
    for token in tokens:
        if token.pos_ == "VERB" or token.dep_ == "ROOT":
            has_subject = any(
                child.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass")
                for child in token.children
            )
            if has_subject:
                verbs_with_subjects += 1

    # Phrasal elaboration: count dependents per noun phrase
    noun_dep_counts = []
    for token in tokens:
        if token.pos_ in ("NOUN", "PROPN"):
            phrasal_deps = [
                child for child in token.children
                if child.dep_ in ("amod", "prep", "relcl", "acl", "appos",
                                   "nmod", "compound", "det", "poss", "nummod")
            ]
            noun_dep_counts.append(len(phrasal_deps))

    # MDD by dependency type
    dep_type_groups = {}
    for d in deps:
        dep_type = d["dep"]
        dep_type_groups.setdefault(dep_type, []).append(d["arc_length"])

    mdd_by_type = {
        dep_type: sum(lengths) / len(lengths)
        for dep_type, lengths in dep_type_groups.items()
    }

    return {
        "text": sent.text,
        "start_char": sent.start_char,
        "end_char": sent.end_char,
        "word_count": len([t for t in tokens if not t.is_punct and not t.is_space]),
        "token_count": len(tokens),
        "tokens": [
            {
                "text": t.text,
                "lower": t.lower_,
                "lemma": t.lemma_.lower(),
                "pos": t.pos_,
                "is_alpha": t.is_alpha,
                "ent_type": t.ent_type_,
            }
            for t in tokens
        ],
        "sentence_type": _classify_sentence_type(tokens),
        "has_terminal_punct": _has_terminal_punct(sent.text),
        "mdd": mdd,
        "adjacent_dep_proportion": adjacent_prop,
        "max_arc_length": max_arc_length,
        "tree_depth": tree_depth,
        "root_count": root_count,
        "nsubj_count": nsubj_count,
        "punct_count": punct_count,
        "has_verb_root": any(t.dep_ == "ROOT" and t.pos_ == "VERB" for t in tokens),
        "clauses_with_subjects": verbs_with_subjects,
        "noun_dep_counts": noun_dep_counts,
        "mdd_by_type": mdd_by_type,
        "dep_triples": deps,
        "pos_sequence": [t.pos_ for t in tokens],
    }


class SpacyPass(BasePass):
    """Process essays with spaCy"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.nlp = None

    @property
    def name(self) -> str:
        return "spacy"

    def load_model(self) -> None:
        model_name = self.config.get("models", {}).get("spacy", {}).get(
            "model", "en_core_web_trf"
        )
        logger.info(f"Loading spaCy model: {model_name}")
        self.nlp = spacy.load(model_name)
        logger.info(f"spaCy pipeline: {self.nlp.pipe_names}")

    def process_essay(self, essay: Essay) -> None:
        if self.nlp is None:
            raise RuntimeError("spaCyPass.load_model() must be called before processing.")

        doc: Doc = self.nlp(essay.text)

        sentences = [_compute_sentence_data(sent) for sent in doc.sents]

        essay.pass_results[self.name] = {
            "sentences": sentences,
            "num_sentences": len(sentences),
            "total_words": sum(s["word_count"] for s in sentences),
            "total_tokens": sum(s["token_count"] for s in sentences),
        }

    def process_batch(self, essays: list[Essay]) -> None:
        """Use spaCy's nlp.pipe() for batch processing efficiency."""
        if self.nlp is None:
            raise RuntimeError("SpacyPass.load_model() must be called before processing.")

        texts = [e.text for e in essays]
        batch_size = self.config.get("hardware", {}).get("batch_size", 16)

        for essay, doc in zip(essays, self.nlp.pipe(texts, batch_size=batch_size)):
            sentences = [_compute_sentence_data(sent) for sent in doc.sents]

            essay.pass_results[self.name] = {
                "sentences": sentences,
                "num_sentences": len(sentences),
                "total_words": sum(s["word_count"] for s in sentences),
                "total_tokens": sum(s["token_count"] for s in sentences),
            }

    def unload_model(self) -> None:
        self.nlp = None
        logger.info("Unloaded spaCy model.")
