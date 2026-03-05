"""
MDD Feature Validation Harness
==============================

Validates that MDD features behave as expected when conventions errors
are introduced into sentences of varying syntactic complexity.

Key questions this harness answers:
1. Do MDD features correctly reflect known syntactic complexity tiers?
2. When errors are injected, do MDD values shift (indicating parse distortion)
   or stay stable (indicating clean separation from error features)?
3. Are MDD features and LanguageTool detections measuring orthogonal or
   confounded signal?

Usage:
    python mdd_validation_harness.py

Requirements:
    - spaCy with en_core_web_trf (or en_core_web_sm for quick tests)
    - language_tool_python
    - pandas (for reporting)
"""

import json
import statistics
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import spacy
import language_tool_python


# ---------------------------------------------------------------------------
# 1. SYNTHETIC TEST CASES
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    """A single validation case: clean sentence + error-injected variant."""
    id: str
    complexity_tier: str          # "low", "medium", "high"
    structure_type: str           # Which dep type is being exercised
    clean: str                    # Grammatically correct sentence
    corrupted: str                # Same sentence with a specific error injected
    error_type: str               # What kind of error was injected
    error_description: str        # Human-readable description
    expected_lt_rules: list[str] = field(default_factory=list)  # Expected LT rule IDs


def build_test_cases() -> list[TestCase]:
    """
    Build synthetic test cases across 3 complexity tiers x multiple error types.
    
    Design principles:
    - Each tier targets specific MDD-relevant dependency types
    - Errors are injected one at a time for clean attribution
    - Expected LanguageTool rules are noted for cross-checking
    """
    cases = []

    # ===================================================================
    # TIER 1: LOW COMPLEXITY (simple SVO, short dependencies)
    # Expected MDD: ~1.5-2.5
    # ===================================================================

    # -- Baseline low-complexity sentences --
    cases.append(TestCase(
        id="low_01_clean",
        complexity_tier="low",
        structure_type="simple_svo",
        clean="The student completed the assignment on time.",
        corrupted="The student completed the assignment on time.",
        error_type="none",
        error_description="Clean baseline — no error injected",
    ))

    cases.append(TestCase(
        id="low_02_spelling",
        complexity_tier="low",
        structure_type="simple_svo",
        clean="The student completed the assignment on time.",
        corrupted="The studnet complted the assignment on time.",
        error_type="spelling",
        error_description="Two misspellings in simple SVO (should NOT affect parse)",
        expected_lt_rules=["MORFOLOGIK_RULE_EN_US", "MORFOLOGIK_RULE_EN_US"],
    ))

    cases.append(TestCase(
        id="low_03_svagree",
        complexity_tier="low",
        structure_type="simple_svo",
        clean="The student completes the assignment on time.",
        corrupted="The student complete the assignment on time.",
        error_type="sv_agreement",
        error_description="Subject-verb agreement error in simple SVO",
        expected_lt_rules=["SHE_LIVE"],
    ))

    cases.append(TestCase(
        id="low_04_comma_splice",
        complexity_tier="low",
        structure_type="simple_svo",
        clean="The student finished early. She left the room.",
        corrupted="The student finished early, she left the room.",
        error_type="comma_splice",
        error_description="Comma splice fusing two independent clauses",
        expected_lt_rules=[],  # LT does not detect comma splices
    ))

    cases.append(TestCase(
        id="low_05_missing_period",
        complexity_tier="low",
        structure_type="simple_svo",
        clean="The dog ran fast. The cat sat still.",
        corrupted="The dog ran fast the cat sat still.",
        error_type="missing_boundary",
        error_description="Missing sentence boundary (run-on) — may cause parse distortion",
    ))

    # ===================================================================
    # TIER 2: MEDIUM COMPLEXITY (adverbial clauses, coordination)
    # Expected MDD: ~2.5-3.5
    # Key dep types: advcl, conj
    # ===================================================================

    cases.append(TestCase(
        id="med_01_clean",
        complexity_tier="medium",
        structure_type="advcl",
        clean="Although the experiment failed, the researchers published their findings.",
        corrupted="Although the experiment failed, the researchers published their findings.",
        error_type="none",
        error_description="Clean baseline — adverbial clause",
    ))

    cases.append(TestCase(
        id="med_02_spelling",
        complexity_tier="medium",
        structure_type="advcl",
        clean="Although the experiment failed, the researchers published their findings.",
        corrupted="Although the expiriment failed, the reserchers published their findings.",
        error_type="spelling",
        error_description="Two misspellings in advcl structure",
        expected_lt_rules=["MORFOLOGIK_RULE_EN_US", "MORFOLOGIK_RULE_EN_US"],
    ))

    cases.append(TestCase(
        id="med_03_missing_comma",
        complexity_tier="medium",
        structure_type="advcl",
        clean="Although the experiment failed, the researchers published their findings.",
        corrupted="Although the experiment failed the researchers published their findings.",
        error_type="missing_comma",
        error_description="Missing comma after adverbial clause — may shift dep parsing",
    ))

    cases.append(TestCase(
        id="med_04_conj_clean",
        complexity_tier="medium",
        structure_type="conj",
        clean="The team analyzed the data and wrote a detailed report.",
        corrupted="The team analyzed the data and wrote a detailed report.",
        error_type="none",
        error_description="Clean baseline — coordination with conj",
    ))

    cases.append(TestCase(
        id="med_05_conj_error",
        complexity_tier="medium",
        structure_type="conj",
        clean="The team analyzed the data and wrote a detailed report.",
        corrupted="The team analized the data and writed a detailed report.",
        error_type="spelling_morphological",
        error_description="Spelling + morphological error in coordinated structure",
        expected_lt_rules=["MORFOLOGIK_RULE_EN_US", "WROTE"],
    ))

    cases.append(TestCase(
        id="med_06_fragment",
        complexity_tier="medium",
        structure_type="advcl",
        clean="Because the weather was bad, they canceled the game.",
        corrupted="Because the weather was bad. They canceled the game.",
        error_type="fragment",
        error_description="Subordinate clause turned into fragment by period insertion",
    ))

    # ===================================================================
    # TIER 3: HIGH COMPLEXITY (relative clauses, xcomp, ccomp, stacked prep)
    # Expected MDD: ~3.5-5.0+
    # Key dep types: relcl, xcomp, ccomp, prep (stacked)
    # ===================================================================

    cases.append(TestCase(
        id="high_01_relcl_clean",
        complexity_tier="high",
        structure_type="relcl",
        clean="The professor who taught the advanced seminar assigned a paper that required extensive research.",
        corrupted="The professor who taught the advanced seminar assigned a paper that required extensive research.",
        error_type="none",
        error_description="Clean baseline — nested relative clauses",
    ))

    cases.append(TestCase(
        id="high_02_relcl_spelling",
        complexity_tier="high",
        structure_type="relcl",
        clean="The professor who taught the advanced seminar assigned a paper that required extensive research.",
        corrupted="The proffessor who tought the advanced seminar assigned a paper that requried extensive reserch.",
        error_type="spelling",
        error_description="Multiple misspellings in relcl structure",
        expected_lt_rules=["MORFOLOGIK_RULE_EN_US", "MORFOLOGIK_RULE_EN_US", "MORFOLOGIK_RULE_EN_US", "MORFOLOGIK_RULE_EN_US"],
    ))

    cases.append(TestCase(
        id="high_03_relcl_missing_pronoun",
        complexity_tier="high",
        structure_type="relcl",
        clean="The professor who taught the advanced seminar assigned a paper that required extensive research.",
        corrupted="The professor taught the advanced seminar assigned a paper required extensive research.",
        error_type="missing_rel_pronoun",
        error_description="Missing relative pronouns 'who' and 'that' — LIKELY to distort parse significantly",
    ))

    cases.append(TestCase(
        id="high_04_xcomp_clean",
        complexity_tier="high",
        structure_type="xcomp",
        clean="The committee decided to postpone the vote until they managed to review all the evidence.",
        corrupted="The committee decided to postpone the vote until they managed to review all the evidence.",
        error_type="none",
        error_description="Clean baseline — xcomp with temporal advcl",
    ))

    cases.append(TestCase(
        id="high_05_xcomp_error",
        complexity_tier="high",
        structure_type="xcomp",
        clean="The committee decided to postpone the vote until they managed to review all the evidence.",
        corrupted="The committee decided postpone the vote until they managed review all the evidence.",
        error_type="missing_infinitive_marker",
        error_description="Missing 'to' in xcomp — may collapse xcomp into direct object parse",
    ))

    cases.append(TestCase(
        id="high_06_ccomp_clean",
        complexity_tier="high",
        structure_type="ccomp",
        clean="The researchers argued that the results demonstrated a significant correlation between the variables.",
        corrupted="The researchers argued that the results demonstrated a significant correlation between the variables.",
        error_type="none",
        error_description="Clean baseline — ccomp with prepositional phrase",
    ))

    cases.append(TestCase(
        id="high_07_ccomp_error",
        complexity_tier="high",
        structure_type="ccomp",
        clean="The researchers argued that the results demonstrated a significant correlation between the variables.",
        corrupted="The researchers argued the results demonstrated a significant correlation between the variables.",
        error_type="missing_complementizer",
        error_description="Missing 'that' complementizer — may still parse correctly (common omission)",
    ))

    cases.append(TestCase(
        id="high_08_prep_stacked_clean",
        complexity_tier="high",
        structure_type="prep",
        clean="The report on the impact of climate change on agricultural productivity in developing nations was released.",
        corrupted="The report on the impact of climate change on agricultural productivity in developing nations was released.",
        error_type="none",
        error_description="Clean baseline — stacked prepositional phrases (high dep_mdd_prep expected)",
    ))

    cases.append(TestCase(
        id="high_09_prep_stacked_error",
        complexity_tier="high",
        structure_type="prep",
        clean="The report on the impact of climate change on agricultural productivity in developing nations was released.",
        corrupted="The report on the impact of climate change on agricultural productivity in developing nations released.",
        error_type="missing_aux",
        error_description="Missing auxiliary 'was' — may shift ROOT assignment and tree structure",
    ))

    cases.append(TestCase(
        id="high_10_amod_clean",
        complexity_tier="high",
        structure_type="amod",
        clean="The newly appointed chief financial officer presented the thoroughly revised quarterly earnings report.",
        corrupted="The newly appointed chief financial officer presented the thoroughly revised quarterly earnings report.",
        error_type="none",
        error_description="Clean baseline — heavy amod stacking on nouns",
    ))

    cases.append(TestCase(
        id="high_11_amod_error",
        complexity_tier="high",
        structure_type="amod",
        clean="The newly appointed chief financial officer presented the thoroughly revised quarterly earnings report.",
        corrupted="The new appointed chief financial officer presented the thorough revised quarterly earnings report.",
        error_type="adjective_form",
        error_description="Adverb→adjective substitution — may change amod distances",
    ))

    return cases


# ---------------------------------------------------------------------------
# 2. FEATURE EXTRACTION (mirrors your pipeline's logic)
# ---------------------------------------------------------------------------

def compute_sentence_mdd_data(sent) -> dict[str, Any]:
    """
    Extract MDD data from a spaCy Span, mirroring _compute_sentence_data
    from your spacy_pass.py.
    """
    tokens = list(sent)
    deps = []
    nsubj_count = 0
    root_count = 0

    for token in tokens:
        if token.dep_ == "ROOT":
            root_count += 1
        if token.dep_ == "nsubj":
            nsubj_count += 1
        if token.dep_ != "ROOT":
            arc_length = abs(token.i - token.head.i)
            deps.append({
                "token_i": token.i,
                "head_i": token.head.i,
                "dep": token.dep_,
                "arc_length": arc_length,
            })

    arc_lengths = [d["arc_length"] for d in deps]
    mdd = sum(arc_lengths) / len(arc_lengths) if arc_lengths else 0.0

    # MDD by dependency type
    dep_type_groups: dict[str, list[int]] = {}
    for d in deps:
        dep_type_groups.setdefault(d["dep"], []).append(d["arc_length"])

    mdd_by_type = {
        dep_type: sum(lengths) / len(lengths)
        for dep_type, lengths in dep_type_groups.items()
    }

    # Tree depth
    def _tree_depth(token, depth=0):
        children_depths = [_tree_depth(child, depth + 1) for child in token.children]
        return max(children_depths) if children_depths else depth

    root_tokens = [t for t in tokens if t.dep_ == "ROOT"]
    tree_depth = max((_tree_depth(r) for r in root_tokens), default=0)

    max_arc = max(arc_lengths) if arc_lengths else 0

    return {
        "text": sent.text,
        "word_count": len([t for t in tokens if not t.is_punct and not t.is_space]),
        "mdd": mdd,
        "max_arc_length": max_arc,
        "tree_depth": tree_depth,
        "root_count": root_count,
        "nsubj_count": nsubj_count,
        "has_verb_root": any(t.dep_ == "ROOT" and t.pos_ == "VERB" for t in tokens),
        "mdd_by_type": mdd_by_type,
        "dep_triples": deps,
    }


def extract_doc_mdd_features(doc) -> dict[str, float]:
    """
    Extract document-level MDD features, mirroring compute_dependency_features
    from your dependency.py.
    """
    sentences = [compute_sentence_mdd_data(sent) for sent in doc.sents]

    if not sentences:
        return {}

    sentence_mdds = [s["mdd"] for s in sentences]

    features = {
        "dep_mdd_mean": statistics.mean(sentence_mdds),
        "dep_mdd_max": max(sentence_mdds),
    }

    if len(sentence_mdds) >= 2:
        features["dep_mdd_stdev"] = statistics.stdev(sentence_mdds)
    else:
        features["dep_mdd_stdev"] = 0.0

    # MDD by dependency type (aggregated across sentences)
    type_distances: dict[str, list[float]] = {}
    for sent in sentences:
        for dep_type, mdd_val in sent["mdd_by_type"].items():
            type_distances.setdefault(dep_type, []).append(mdd_val)

    for dep_type in ["advcl", "relcl", "ccomp", "xcomp", "conj", "amod", "prep"]:
        vals = type_distances.get(dep_type, [])
        features[f"dep_mdd_{dep_type}"] = statistics.mean(vals) if vals else 0.0

    # Tree depth
    depths = [s["tree_depth"] for s in sentences]
    features["dep_tree_depth_mean"] = statistics.mean(depths)
    features["dep_tree_depth_max"] = max(depths)

    # Max arc
    max_arcs = [s["max_arc_length"] for s in sentences]
    features["dep_max_arc_max"] = max(max_arcs)

    # Fragment detection
    fragment_count = sum(
        1 for s in sentences
        if s["has_verb_root"] and s["nsubj_count"] == 0
    )
    features["dep_fragment_count"] = float(fragment_count)

    # Multi-root (run-on indicator)
    multi_root_count = sum(1 for s in sentences if s["root_count"] > 1)
    features["dep_multi_root_count"] = float(multi_root_count)

    return features


# ---------------------------------------------------------------------------
# 3. LANGUAGETOOL ANALYSIS
# ---------------------------------------------------------------------------

def extract_lt_matches(tool, text: str) -> list[dict]:
    """Run LanguageTool and return simplified match records."""
    matches = tool.check(text)
    return [
        {
            "rule_id": m.rule_id,
            "category": m.category,
            "message": m.message,
            "offset": m.offset,
            "length": m.error_length,
            "matched_text": text[m.offset:m.offset + m.error_length],
            "replacements": m.replacements[:3] if m.replacements else [],
        }
        for m in matches
    ]


# ---------------------------------------------------------------------------
# 4. DEPENDENCY TREE VISUALIZATION (for debugging)
# ---------------------------------------------------------------------------

def format_dep_tree(doc) -> str:
    """Pretty-print the dependency parse for inspection."""
    lines = []
    for sent in doc.sents:
        lines.append(f"Sentence: {sent.text}")
        for token in sent:
            lines.append(
                f"  {token.i:3d} {token.text:<20s} "
                f"POS={token.pos_:<6s} DEP={token.dep_:<12s} "
                f"HEAD={token.head.text} (i={token.head.i})"
            )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. MAIN VALIDATION RUNNER
# ---------------------------------------------------------------------------

def run_validation(
    spacy_model: str = "en_core_web_trf",
    verbose: bool = True,
    output_path: str = "mdd_validation_results.csv",
    dep_tree_output: str = "mdd_validation_dep_trees.txt",
):
    """
    Run the full validation suite.
    
    For each test case:
    1. Parse clean and corrupted versions with spaCy
    2. Extract MDD features from both
    3. Run LanguageTool on both
    4. Compute deltas and flag unexpected behaviors
    """
    print(f"Loading spaCy model: {spacy_model}")
    nlp = spacy.load(spacy_model)

    print("Initializing LanguageTool...")
    lt = language_tool_python.LanguageTool("en-US")

    cases = build_test_cases()
    results = []
    dep_tree_lines = []

    # MDD feature keys we'll track
    mdd_keys = [
        "dep_mdd_mean", "dep_mdd_max", "dep_mdd_stdev",
        "dep_mdd_advcl", "dep_mdd_relcl", "dep_mdd_ccomp",
        "dep_mdd_xcomp", "dep_mdd_conj", "dep_mdd_amod", "dep_mdd_prep",
        "dep_tree_depth_mean", "dep_tree_depth_max", "dep_max_arc_max",
        "dep_fragment_count", "dep_multi_root_count",
    ]

    print(f"\nRunning {len(cases)} test cases...\n")
    print("=" * 100)

    for case in cases:
        # --- Parse ---
        doc_clean = nlp(case.clean)
        doc_corrupt = nlp(case.corrupted)

        # --- Extract MDD features ---
        feat_clean = extract_doc_mdd_features(doc_clean)
        feat_corrupt = extract_doc_mdd_features(doc_corrupt)

        # --- LanguageTool ---
        lt_clean = extract_lt_matches(lt, case.clean)
        lt_corrupt = extract_lt_matches(lt, case.corrupted)

        # --- Compute deltas ---
        deltas = {}
        for key in mdd_keys:
            v_clean = feat_clean.get(key, 0.0)
            v_corrupt = feat_corrupt.get(key, 0.0)
            deltas[key] = v_corrupt - v_clean

        # --- Classify behavior ---
        # "stable" = MDD didn't shift much (<10% relative change)
        # "distorted" = MDD shifted significantly
        mdd_mean_clean = feat_clean.get("dep_mdd_mean", 0.0)
        mdd_mean_corrupt = feat_corrupt.get("dep_mdd_mean", 0.0)
        if mdd_mean_clean > 0:
            relative_change = abs(mdd_mean_corrupt - mdd_mean_clean) / mdd_mean_clean
        else:
            relative_change = 0.0

        if case.error_type == "none":
            behavior = "baseline"
        elif relative_change < 0.10:
            behavior = "STABLE (good: error doesn't distort parse)"
        elif relative_change < 0.25:
            behavior = "MODERATE_SHIFT (review: some parse distortion)"
        else:
            behavior = "MAJOR_DISTORTION (warning: MDD confounded with error)"

        # --- LT rule matching ---
        lt_rules_found = [m["rule_id"] for m in lt_corrupt]
        lt_rules_expected = set(case.expected_lt_rules)
        lt_rules_actual = set(lt_rules_found)
        lt_match_status = "N/A" if not lt_rules_expected else (
            "MATCH" if lt_rules_expected & lt_rules_actual else "MISS"
        )

        # --- Build result row ---
        row = {
            "case_id": case.id,
            "complexity_tier": case.complexity_tier,
            "structure_type": case.structure_type,
            "error_type": case.error_type,
            "error_description": case.error_description,
            "behavior": behavior,
            "mdd_relative_change": round(relative_change, 4),
            "lt_rules_expected": ", ".join(case.expected_lt_rules),
            "lt_rules_found": ", ".join(lt_rules_found),
            "lt_match_status": lt_match_status,
            "lt_clean_count": len(lt_clean),
            "lt_corrupt_count": len(lt_corrupt),
        }

        # Add clean/corrupt/delta values for each MDD feature
        for key in mdd_keys:
            row[f"clean_{key}"] = round(feat_clean.get(key, 0.0), 4)
            row[f"corrupt_{key}"] = round(feat_corrupt.get(key, 0.0), 4)
            row[f"delta_{key}"] = round(deltas.get(key, 0.0), 4)

        results.append(row)

        # --- Verbose output ---
        if verbose:
            print(f"Case: {case.id}")
            print(f"  Tier: {case.complexity_tier} | Structure: {case.structure_type}")
            print(f"  Error: {case.error_type} — {case.error_description}")
            print(f"  Clean:   \"{case.clean}\"")
            print(f"  Corrupt: \"{case.corrupted}\"")
            print(f"  MDD (clean):   {mdd_mean_clean:.4f}")
            print(f"  MDD (corrupt): {mdd_mean_corrupt:.4f}")
            print(f"  Relative Δ:    {relative_change:.2%}")
            print(f"  Behavior:      {behavior}")
            print(f" LT rules expected:  {', '.join(lt_rules_expected)}")
            print(f"  LT matches (corrupt): {lt_rules_found}")
            print(f"  LT match status: {lt_match_status}")

            # Show per-type deltas for the structure type being tested
            target_key = f"dep_mdd_{case.structure_type}"
            if target_key in deltas and case.structure_type in [
                "advcl", "relcl", "ccomp", "xcomp", "conj", "amod", "prep"
            ]:
                print(f"  Target dep_mdd_{case.structure_type}: "
                      f"clean={feat_clean.get(target_key, 0.0):.4f} "
                      f"corrupt={feat_corrupt.get(target_key, 0.0):.4f} "
                      f"Δ={deltas[target_key]:.4f}")

            print("-" * 100)

        # --- Dep tree output for inspection ---
        dep_tree_lines.append(f"{'=' * 80}")
        dep_tree_lines.append(f"Case: {case.id} ({case.error_type})")
        dep_tree_lines.append(f"{'=' * 80}")
        dep_tree_lines.append(f"\n--- CLEAN ---")
        dep_tree_lines.append(format_dep_tree(doc_clean))
        dep_tree_lines.append(f"\n--- CORRUPTED ---")
        dep_tree_lines.append(format_dep_tree(doc_corrupt))
        dep_tree_lines.append("")

    # --- Save results ---
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    with open(dep_tree_output, "w") as f:
        f.write("\n".join(dep_tree_lines))
    print(f"Dependency trees saved to: {dep_tree_output}")

    # --- Summary report ---
    print("\n" + "=" * 100)
    print("VALIDATION SUMMARY")
    print("=" * 100)

    print("\n1. MDD TIER VALIDATION (do clean baselines show expected complexity ordering?)")
    baselines = df[df["error_type"] == "none"].sort_values("complexity_tier")
    for _, row in baselines.iterrows():
        print(f"   {row['complexity_tier']:>6s} | {row['structure_type']:<15s} | "
              f"MDD={row['clean_dep_mdd_mean']:.4f} | "
              f"depth={row['clean_dep_tree_depth_max']:.0f} | "
              f"max_arc={row['clean_dep_max_arc_max']:.0f}")

    print("\n2. ERROR IMPACT ON MDD (which errors distort the parse?)")
    errors = df[df["error_type"] != "none"].sort_values("mdd_relative_change", ascending=False)
    for _, row in errors.iterrows():
        flag = "⚠️" if "DISTORTION" in row["behavior"] else "  "
        print(f"   {flag} {row['case_id']:<30s} | Δ={row['mdd_relative_change']:>7.2%} | "
              f"{row['behavior']}")

    print("\n3. LANGUAGETOOL COVERAGE (are expected rules firing?)")
    lt_cases = df[df["lt_match_status"] != "N/A"]
    for _, row in lt_cases.iterrows():
        status = "✓" if row["lt_match_status"] == "MATCH" else "✗"
        print(f"   {status} {row['case_id']:<30s} | "
              f"expected: {row.get('lt_rules_expected', 'N/A')} | "
              f"found: {row['lt_rules_found']}")

    print("\n4. CONFOUNDING CHECK (cases where MDD shifted AND LT detected the same error)")
    confounded = df[
        (df["error_type"] != "none") &
        (df["mdd_relative_change"] > 0.10) &
        (df["lt_corrupt_count"] > df["lt_clean_count"])
    ]
    if confounded.empty:
        print("   No confounding detected — MDD and LT appear to measure orthogonal signal.")
    else:
        print("   POTENTIAL CONFOUNDING in the following cases:")
        for _, row in confounded.iterrows():
            print(f"   ⚠️ {row['case_id']}: MDD Δ={row['mdd_relative_change']:.2%}, "
                  f"LT rules={row['lt_rules_found']}")
        print("\n   These cases indicate the MDD features and error features may be")
        print("   capturing overlapping signal. Consider whether the model is")
        print("   double-counting the effect of these error types.")

    lt.close()
    print("\nDone.")
    return df


# ---------------------------------------------------------------------------
# 6. ADDING YOUR OWN TEST CASES
# ---------------------------------------------------------------------------

def add_custom_case(
    cases: list[TestCase],
    clean: str,
    corrupted: str,
    error_type: str,
    description: str,
    structure_type: str = "custom",
    complexity_tier: str = "custom",
    expected_lt_rules: list[str] | None = None,
):
    """
    Helper to add your own test cases for specific patterns you've
    observed in your data.
    
    Example:
        cases = build_test_cases()
        add_custom_case(
            cases,
            clean="The students who had studied hard passed the exam.",
            corrupted="The students who had studied hard past the exam.",
            error_type="homophone",
            description="past/passed confusion in relative clause",
            structure_type="relcl",
            complexity_tier="high",
        )
    """
    idx = len(cases) + 1
    cases.append(TestCase(
        id=f"custom_{idx:02d}",
        complexity_tier=complexity_tier,
        structure_type=structure_type,
        clean=clean,
        corrupted=corrupted,
        error_type=error_type,
        error_description=description,
        expected_lt_rules=expected_lt_rules or [],
    ))


if __name__ == "__main__":
    run_validation(
        spacy_model="en_core_web_trf",  # Change to "en_core_web_sm" for faster testing
        verbose=True,
        output_path="mdd_validation_results.csv",
        dep_tree_output="mdd_validation_dep_trees.txt",
    )
