"""Registry lớp 3 — chọn correction backend theo tên.

  protonx  -> ProtonX distilled (mặc định; NHỚ license NC cho commercial)
  bmd1905  -> bmd1905/vietnamese-correction-v2 (license thoáng hơn)
  null     -> tắt correction (đo baseline / khi vướng license)
"""
from __future__ import annotations

from typing import Optional

from .base import CorrectionBackend

# Map tên ngắn -> model_id trên HuggingFace
_MODEL_IDS = {
    "protonx": "protonx-models/distilled-protonx-legal-tc",
    "protonx-full": "protonx-models/protonx-legal-tc",
    "protonx-nano": "protonx-models/nano-protonx-legal-tc",
    "bmd1905": "bmd1905/vietnamese-correction-v2",
}


class _NullCorrection(CorrectionBackend):
    name = "null"

    def _correct_raw(self, text: str) -> str:
        return text


def build_correction_backend(name: str, **kwargs) -> CorrectionBackend:
    name = (name or "null").lower()
    if name in ("null", "off", "none"):
        return _NullCorrection()
    model_id = kwargs.pop("model_id", None) or _MODEL_IDS.get(name)
    if model_id is None:
        raise ValueError(
            f"Correction backend không rõ: {name!r}. "
            f"Chọn {list(_MODEL_IDS)} hoặc 'null', hoặc truyền model_id."
        )
    from .seq2seq_backend import Seq2SeqCorrectionBackend

    return Seq2SeqCorrectionBackend(model_id=model_id, name=name, **kwargs)


AVAILABLE_CORRECTION_BACKENDS = ["protonx", "bmd1905", "null"]

__all__ = [
    "CorrectionBackend",
    "build_correction_backend",
    "AVAILABLE_CORRECTION_BACKENDS",
]
