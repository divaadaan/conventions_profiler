"""
Dependency Tree Viewer
======================

Renders side-by-side ASCII dependency trees for MDD validation cases
where errors caused parse distortion (MDD relative change > 0).

Usage:
    python dep_tree_viewer.py
"""

import spacy
from mdd_validation_harness import build_test_cases, extract_doc_mdd_features


def render_tree(doc) -> str:
    """Render a spaCy doc as an indented dependency tree."""
    lines = []
    for sent in doc.sents:
        root_tokens = [t for t in sent if t.dep_ == "ROOT"]
        for root in root_tokens:
            lines.extend(_render_subtree(root, prefix="", is_last=True))
        lines.append("")
    return "\n".join(lines)


def _render_subtree(token, prefix: str, is_last: bool) -> list[str]:
    """Recursively render a token and its children as an indented tree."""
    connector = "└── " if is_last else "├── "
    label = f"{token.text} [{token.dep_}:{token.pos_}]"
    lines = [f"{prefix}{connector}{label}"]

    child_prefix = prefix + ("    " if is_last else "│   ")
    children = sorted(token.children, key=lambda c: c.i)
    for i, child in enumerate(children):
        lines.extend(_render_subtree(child, child_prefix, i == len(children) - 1))

    return lines


def main():
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_trf")

    cases = build_test_cases()

    # Filter to cases with actual errors (skip baselines)
    error_cases = [c for c in cases if c.error_type != "none"]

    print(f"\nRendering trees for {len(error_cases)} error cases\n")

    for case in error_cases:
        doc_clean = nlp(case.clean)
        doc_corrupt = nlp(case.corrupted)

        feat_clean = extract_doc_mdd_features(doc_clean)
        feat_corrupt = extract_doc_mdd_features(doc_corrupt)

        mdd_clean = feat_clean.get("dep_mdd_mean", 0.0)
        mdd_corrupt = feat_corrupt.get("dep_mdd_mean", 0.0)
        if mdd_clean > 0:
            rel_change = abs(mdd_corrupt - mdd_clean) / mdd_clean
        else:
            rel_change = 0.0

        marker = "***" if rel_change >= 0.10 else ""

        print("=" * 80)
        print(f"  {case.id} ({case.error_type}) {marker}")
        print(f"  {case.error_description}")
        print(f"  MDD: {mdd_clean:.4f} -> {mdd_corrupt:.4f}  (Δ {rel_change:.1%})")
        print("=" * 80)

        print(f"\n  CLEAN: \"{case.clean}\"")
        tree_clean = render_tree(doc_clean)
        for line in tree_clean.split("\n"):
            print(f"  {line}")

        print(f"\n  CORRUPTED: \"{case.corrupted}\"")
        tree_corrupt = render_tree(doc_corrupt)
        for line in tree_corrupt.split("\n"):
            print(f"  {line}")

        print()


if __name__ == "__main__":
    main()
