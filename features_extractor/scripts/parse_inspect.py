"""Print the spaCy parse for one or more sentences. Useful when calibrating
heuristics in the feature modules.

Usage:
    python scripts/parse_inspect.py "The students walked to the park, they played soccer."
    python scripts/parse_inspect.py  # uses a default set of comma-splice / boundary cases

Optional: --model en_core_web_sm  (defaults to en_core_web_trf)
"""

from __future__ import annotations

import argparse

import spacy

DEFAULT_TEXTS = [
    "The students walked to the park, they played soccer.",
    "The students walked to the park, and they played soccer.",
    "The students walked to the park; they played soccer.",
    "The dog ran fast the cat sat still.",
    "Although the experiment failed, the researchers published their findings.",
]


def inspect(nlp, text: str) -> None:
    doc = nlp(text)
    print("=" * 80)
    print(f"TEXT: {text}")
    print("=" * 80)
    for sent_i, sent in enumerate(doc.sents):
        print(f"\nSentence {sent_i}: {sent.text}")
        print(f"{'i':>3} {'text':<14s} {'POS':<6s} {'DEP':<12s} {'ENT':<8s} HEAD")
        print("-" * 60)
        for t in sent:
            print(
                f"{t.i:>3} {t.text:<14s} {t.pos_:<6s} {t.dep_:<12s} "
                f"{t.ent_type_ or '-':<8s} {t.head.text}({t.head.i})"
            )

        # Summary signals used by the feature modules
        deps = [t.dep_ for t in sent]
        has_cc = "cc" in deps
        has_parataxis = "parataxis" in deps
        has_conj_verb = any(
            t.dep_ == "conj" and t.pos_ in ("VERB", "AUX")
            and any(c.dep_ in ("nsubj", "nsubjpass") for c in t.children)
            for t in sent
        )
        has_para_verb = any(
            t.dep_ == "parataxis" and t.pos_ in ("VERB", "AUX")
            and any(c.dep_ in ("nsubj", "nsubjpass") for c in t.children)
            for t in sent
        )
        print(
            f"\nsignals: cc={has_cc} parataxis={has_parataxis} "
            f"conj_verb_with_subj={has_conj_verb} parataxis_verb_with_subj={has_para_verb}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("texts", nargs="*", help="Texts to parse (defaults to a fixed set)")
    parser.add_argument("--model", default="en_core_web_trf")
    args = parser.parse_args()

    nlp = spacy.load(args.model)
    texts = args.texts if args.texts else DEFAULT_TEXTS
    for t in texts:
        inspect(nlp, t)
        print()


if __name__ == "__main__":
    main()
