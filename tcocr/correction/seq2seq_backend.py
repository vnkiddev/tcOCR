"""Correction backend cho các model Seq2Seq trên HuggingFace.

Cả ProtonX (protonx-models/*-legal-tc) lẫn bmd1905/vietnamese-correction đều là
kiến trúc Seq2Seq (AutoModelForSeq2SeqLM), nên dùng chung 1 class, chỉ khác model_id.

⚠️ LICENSE: ProtonX phát hành theo license v1.3-NC (Non-Commercial). TCBS dùng
production là commercial -> phải xin bản quyền thương mại từ ProtonX, hoặc dùng
bmd1905 (license thoáng hơn). Backend để pluggable đúng vì lý do này.
"""
from __future__ import annotations

from .base import CorrectionBackend


class Seq2SeqCorrectionBackend(CorrectionBackend):
    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        max_length: int = 512,
        num_beams: int = 4,
        name: str = "seq2seq",
        **kwargs,
    ):
        self.name = name
        self.model_id = model_id
        self.max_length = max_length
        self.num_beams = num_beams
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Cần transformers + torch. Colab: `pip install transformers torch sentencepiece`"
            ) from e
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self._device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self._model.to(self._device)
        self._model.eval()

    def _correct_raw(self, text: str) -> str:
        inputs = self._tokenizer(
            [text], return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self._device)
        with self._torch.no_grad():
            out = self._model.generate(
                **inputs, max_length=self.max_length, num_beams=self.num_beams
            )
        return self._tokenizer.batch_decode(out, skip_special_tokens=True)[0]
