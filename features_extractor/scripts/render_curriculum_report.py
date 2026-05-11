"""Render docs/curriculum_mapping.yaml into three report formats.

Outputs (written to docs/reports/):
  - curriculum_coverage.md    Human-readable markdown with grouped tables
  - curriculum_gaps.md        Filtered view: items with gap: true, grouped by level
  - curriculum_coverage.csv   Flat row-per-item table for spreadsheet review
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "docs" / "curriculum_mapping.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "reports"

# Sections expected in the YAML (top-level keys with a list of items).
SECTIONS = ["capitalization", "punctuation", "syntax", "grammar"]

LEVEL_ORDER = {"low": 0, "medium": 1, "high": 2}


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _detector_summary(detectors: list[dict]) -> str:
    """Render a list of detectors as a short pipe-separated string."""
    if not detectors:
        return "_(none)_"
    parts = []
    for d in detectors:
        dtype = d.get("type", "?")
        did = d.get("id", "?")
        conf = d.get("confidence", "")
        parts.append(f"{dtype}:`{did}`" + (f" ({conf})" if conf else ""))
    return " · ".join(parts)


def _coverage_label(item: dict) -> str:
    if item.get("gap"):
        return "GAP"
    detectors = item.get("detectors") or []
    confidences = [d.get("confidence") for d in detectors]
    if any(c == "high" for c in confidences):
        return "covered (high confidence)"
    if any(c == "medium" for c in confidences):
        return "covered (medium confidence)"
    if confidences:
        return "covered (low confidence)"
    return "GAP"


def render_markdown_full(data: dict) -> str:
    out = []
    out.append("# Curriculum Coverage Report")
    out.append("")
    out.append(f"Source: `docs/curriculum_mapping.yaml` (version {data.get('version', 'unknown')}, "
               f"locale {data.get('locale', 'unknown')}, status {data.get('status', 'unknown')})")
    out.append("")

    total_items = 0
    total_gaps = 0

    for section in SECTIONS:
        items = data.get(section) or []
        if not items:
            out.append(f"## {section.title()}")
            out.append("")
            out.append("_No items mapped yet._")
            out.append("")
            continue

        out.append(f"## {section.title()}")
        out.append("")
        out.append("| Curriculum Item | Level | Coverage | Detectors |")
        out.append("|---|---|---|---|")

        items_sorted = sorted(items, key=lambda x: (LEVEL_ORDER.get(x.get("level", "high"), 99),
                                                     x.get("id", "")))
        for item in items_sorted:
            total_items += 1
            label = _coverage_label(item)
            if "GAP" in label:
                total_gaps += 1
            row = (
                f"| {item.get('curriculum_item', item.get('id', '?'))} "
                f"| {item.get('level', '?')} "
                f"| {label} "
                f"| {_detector_summary(item.get('detectors') or [])} |"
            )
            out.append(row)

        out.append("")

    out.append("## Summary")
    out.append("")
    out.append(f"- Total mapped curriculum items: **{total_items}**")
    out.append(f"- Items with no current detector (gap): **{total_gaps}**")
    coverage_pct = (1 - total_gaps / total_items) * 100 if total_items else 0
    out.append(f"- Estimated coverage: **{coverage_pct:.0f}%**")
    out.append("")
    return "\n".join(out)


def render_gap_audit(data: dict) -> str:
    out = []
    out.append("# Curriculum Gap Audit")
    out.append("")
    out.append("Curriculum items with no current detector. Grouped by level — low-level gaps are highest priority "
               "since the curriculum places more weight on early-mastered concepts.")
    out.append("")

    by_level: dict[str, list[tuple[str, dict]]] = {"low": [], "medium": [], "high": []}
    for section in SECTIONS:
        for item in (data.get(section) or []):
            if item.get("gap"):
                level = item.get("level", "high")
                by_level.setdefault(level, []).append((section, item))

    total_gaps = sum(len(v) for v in by_level.values())

    for level in ["low", "medium", "high"]:
        gaps = by_level.get(level, [])
        out.append(f"## {level.title()} level — {len(gaps)} gap(s)")
        out.append("")
        if not gaps:
            out.append("_No gaps at this level._")
            out.append("")
            continue
        for section, item in gaps:
            out.append(f"- **{item.get('curriculum_item', item.get('id', '?'))}** "
                       f"_({section})_ — example: {item.get('example', '')}")
            notes = item.get("notes")
            if notes:
                out.append(f"  - {notes}")
        out.append("")

    out.append("## Summary")
    out.append("")
    out.append(f"- Total gaps: **{total_gaps}**")
    out.append(f"- Low-level gaps: **{len(by_level.get('low', []))}** "
               f"(highest priority — early-mastered concepts)")
    out.append(f"- Medium-level gaps: **{len(by_level.get('medium', []))}**")
    out.append(f"- High-level gaps: **{len(by_level.get('high', []))}**")
    out.append("")
    return "\n".join(out)


def render_csv(data: dict, output_path: Path) -> None:
    fieldnames = [
        "section", "id", "curriculum_item", "level", "coverage", "gap",
        "n_detectors", "detector_types", "detector_ids", "detector_confidences",
        "example", "notes", "locale_notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for section in SECTIONS:
            for item in (data.get(section) or []):
                detectors = item.get("detectors") or []
                writer.writerow({
                    "section": section,
                    "id": item.get("id", ""),
                    "curriculum_item": item.get("curriculum_item", ""),
                    "level": item.get("level", ""),
                    "coverage": _coverage_label(item),
                    "gap": item.get("gap", False),
                    "n_detectors": len(detectors),
                    "detector_types": "|".join(d.get("type", "") for d in detectors),
                    "detector_ids": "|".join(d.get("id", "") for d in detectors),
                    "detector_confidences": "|".join(d.get("confidence", "") for d in detectors),
                    "example": item.get("example", ""),
                    "notes": item.get("notes", ""),
                    "locale_notes": item.get("locale_notes", ""),
                })


def main() -> None:
    parser = argparse.ArgumentParser(description="Render curriculum mapping into markdown + CSV reports.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Path to curriculum_mapping.yaml")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to write reports to")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = _load(args.input)

    coverage_md = args.output_dir / "curriculum_coverage.md"
    gaps_md = args.output_dir / "curriculum_gaps.md"
    coverage_csv = args.output_dir / "curriculum_coverage.csv"

    coverage_md.write_text(render_markdown_full(data), encoding="utf-8")
    gaps_md.write_text(render_gap_audit(data), encoding="utf-8")
    render_csv(data, coverage_csv)

    print(f"Wrote {coverage_md}")
    print(f"Wrote {gaps_md}")
    print(f"Wrote {coverage_csv}")


if __name__ == "__main__":
    main()
