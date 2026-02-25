"""
LanguageTool processing pass.

Runs essays through a local LanguageTool server for grammar, spelling, and
mechanics error detection. Stores raw match objects with their rule categories
for downstream severity classification.

Intermediate results stored in essay.pass_results["languagetool"]:
    matches: list[dict]
        Each match contains: rule_id, category, message, offset, length,
        context, replacements, and the matched text span.
    match_count: int
        Total number of matches.
    categories: dict[str, int]
        Count of matches per LanguageTool category.
"""

import logging

import language_tool_python as ltp

from conventions_profiler.models import Essay
from conventions_profiler.passes.base import BasePass

logger = logging.getLogger(__name__)


def _match_to_dict(match, essay_text: str) -> dict:
    """Convert a LanguageTool Match object to a serializable dict."""
    offset = match.offset
    length = match.error_length
    matched_text = essay_text[offset:offset + length] if offset + length <= len(essay_text) else ""

    return {
        "rule_id": match.rule_id,
        "category": match.category,
        "message": match.message,
        "offset": offset,
        "length": length,
        "matched_text": matched_text,
        "replacements": match.replacements[:5] if match.replacements else [],
        "context": match.context,
    }


class LanguageToolPass(BasePass):
    """Process essays with LanguageTool for error detection and classification."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.tool = None

    @property
    def name(self) -> str:
        return "languagetool"

    def load_model(self) -> None:
        lt_config = self.config.get("models", {}).get("languagetool", {})
        language = lt_config.get("language", "en-US")

        logger.info(f"Starting LanguageTool server (language={language})...")
        self.tool = ltp.LanguageTool(language)
        logger.info("LanguageTool server ready.")

    def process_essay(self, essay: Essay) -> None:
        if self.tool is None:
            raise RuntimeError(
                "LanguageToolPass.load_model() must be called before processing."
            )

        raw_matches = self.tool.check(essay.text)

        matches = [_match_to_dict(m, essay.text) for m in raw_matches]

        # Count by category
        categories: dict[str, int] = {}
        for m in matches:
            cat = m["category"]
            categories[cat] = categories.get(cat, 0) + 1

        essay.pass_results[self.name] = {
            "matches": matches,
            "match_count": len(matches),
            "categories": categories,
        }

    def unload_model(self) -> None:
        if self.tool is not None:
            self.tool.close()
            self.tool = None
            logger.info("LanguageTool server stopped.")
