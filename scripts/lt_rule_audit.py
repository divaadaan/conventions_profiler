"""
LanguageTool Rule Audit
=======================

Runs LanguageTool on every test case from the MDD validation harness
and prints all matches, so we can verify/correct the expected rule IDs.

Usage:
    python lt_rule_audit.py
"""

import language_tool_python
from mdd_validation_harness import build_test_cases


def main():
    print("Initializing LanguageTool...")
    lt = language_tool_python.LanguageTool("en-US")

    cases = build_test_cases()

    print(f"\nAuditing {len(cases)} test cases\n")
    print("=" * 100)

    for case in cases:
        matches_clean = lt.check(case.clean)
        matches_corrupt = lt.check(case.corrupted)

        # Only show detail when there's something interesting
        has_expected = bool(case.expected_lt_rules)
        has_matches = bool(matches_corrupt)
        expected_set = set(case.expected_lt_rules)
        found_set = {m.rule_id for m in matches_corrupt}
        mismatch = expected_set != found_set

        print(f"\n--- {case.id} ({case.error_type}) ---")
        print(f"  Clean:   \"{case.clean}\"")
        print(f"  Corrupt: \"{case.corrupted}\"")
        print(f"  Expected rules: {case.expected_lt_rules or '(none)'}")

        if matches_clean:
            print(f"  LT on CLEAN ({len(matches_clean)} matches):")
            for m in matches_clean:
                print(f"    - {m.rule_id} [{m.category}]")
                print(f"      \"{case.clean[m.offset:m.offset + m.error_length]}\"")
                print(f"      {m.message}")

        if matches_corrupt:
            print(f"  LT on CORRUPT ({len(matches_corrupt)} matches):")
            for m in matches_corrupt:
                print(f"    - {m.rule_id} [{m.category}]")
                print(f"      \"{case.corrupted[m.offset:m.offset + m.error_length]}\"")
                print(f"      {m.message}")
        else:
            print(f"  LT on CORRUPT: (no matches)")

        if has_expected:
            if mismatch:
                missing = expected_set - found_set
                extra = found_set - expected_set
                status = "MISMATCH"
                details = []
                if missing:
                    details.append(f"missing: {sorted(missing)}")
                if extra:
                    details.append(f"unexpected: {sorted(extra)}")
                print(f"  STATUS: {status} -- {', '.join(details)}")
            else:
                print(f"  STATUS: OK")

    lt.close()
    print("\n" + "=" * 100)
    print("Done.")


if __name__ == "__main__":
    main()
