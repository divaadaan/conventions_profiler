"""Surprisal pass.

Runs a causal language model (Pythia 70M by default, per Gruteke Klein et al.
2025, arXiv:2502.11150) over each essay and emits per-word surprisal in bits
(log base 2). Words are whitespace-delimited spans of the original text;
sub-word (BPE) surprisals are summed within a word's char range to recover
word-level values.

Stored as:
    essay.pass_results["surprisal"] = {
        "model_id": str,
        "words": [
            {"text": str, "start_char": int, "end_char": int,
             "surprisal": float | None},
            ...
        ],
    }

The first word's surprisal is None: the LM has no left context for the
document-initial BPE token, so any word containing that token gets None.
Downstream aggregates in features/surprisal.py treat None as missing.
"""

import logging
import math
import re

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from feature_extractor.models import Essay
from feature_extractor.passes.base import BasePass

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "EleutherAI/pythia-70m"
_LN2 = math.log(2.0)
_WORD_RE = re.compile(r"\S+")


def _resolve_device(config: dict) -> str:
    device_cfg = config.get("hardware", {}).get("device", "auto")
    if device_cfg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_cfg


class SurprisalPass(BasePass):
    """Per-word surprisal from a causal LM."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.tokenizer = None
        self.model = None
        self.device: str | None = None
        self.model_id = (
            config.get("models", {})
            .get("surprisal", {})
            .get("model", _DEFAULT_MODEL)
        )

    @property
    def name(self) -> str:
        return "surprisal"

    def load_model(self) -> None:
        self.device = _resolve_device(self.config)
        logger.info(
            f"Loading surprisal LM: {self.model_id} (device={self.device})"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()

    def process_essay(self, essay: Essay) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "SurprisalPass.load_model() must be called before processing."
            )

        text = essay.text or ""
        word_spans = [
            (m.group(), m.start(), m.end()) for m in _WORD_RE.finditer(text)
        ]

        if not text.strip() or not word_spans:
            essay.pass_results[self.name] = {
                "model_id": self.model_id,
                "words": [],
            }
            return

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        offsets = encoded["offset_mapping"][0].tolist()

        max_len = getattr(self.model.config, "max_position_embeddings", 2048)
        if input_ids.shape[1] > max_len:
            logger.warning(
                f"Essay {essay.essay_id}: {input_ids.shape[1]} BPE tokens "
                f"exceeds model context {max_len}; truncating."
            )
            input_ids = input_ids[:, :max_len]
            offsets = offsets[:max_len]

        if input_ids.shape[1] < 2:
            # Only one (or zero) BPE token — no predictions possible.
            essay.pass_results[self.name] = {
                "model_id": self.model_id,
                "words": [
                    {
                        "text": w_text,
                        "start_char": int(w_start),
                        "end_char": int(w_end),
                        "surprisal": None,
                    }
                    for (w_text, w_start, w_end) in word_spans
                ],
            }
            return

        with torch.no_grad():
            logits = self.model(input_ids).logits  # (1, T, V)
            log_probs_log2 = F.log_softmax(logits, dim=-1) / _LN2
            targets = input_ids[:, 1:].unsqueeze(-1)  # (1, T-1, 1)
            tgt_log_probs = (
                log_probs_log2[:, :-1, :].gather(-1, targets).squeeze(-1)
            )  # (1, T-1)
            bpe_surprisals = (-tgt_log_probs).squeeze(0).tolist()

        # Align BPE-token i (i >= 1) → bpe_surprisals[i-1]; token 0 → None.
        bpe_token_surp: list[float | None] = [None] + list(bpe_surprisals)

        words_out = []
        for w_text, w_start, w_end in word_spans:
            covered: list[float | None] = []
            for (bpe_start, bpe_end), surp in zip(offsets, bpe_token_surp):
                if bpe_start == bpe_end:
                    # Special / empty token — skip.
                    continue
                if bpe_end <= w_start or bpe_start >= w_end:
                    continue
                covered.append(surp)

            if not covered:
                word_surp = None
            elif any(s is None for s in covered):
                # Word contains the document-initial BPE → no surprisal.
                word_surp = None
            else:
                word_surp = float(sum(covered))

            words_out.append({
                "text": w_text,
                "start_char": int(w_start),
                "end_char": int(w_end),
                "surprisal": word_surp,
            })

        essay.pass_results[self.name] = {
            "model_id": self.model_id,
            "words": words_out,
        }

    def validate_pass_results(self, essay: Essay) -> bool:
        data = essay.pass_results.get(self.name)
        if not isinstance(data, dict):
            return False
        return data.get("model_id") == self.model_id

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        self.tokenizer = None
        if self.device and self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        logger.info("Unloaded surprisal model.")
