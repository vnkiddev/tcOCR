"""OCR backend dùng PaddleOCR (hỗ trợ cả API 2.x và 3.x qua helper _paddle).

Dùng singleton nên không bị lỗi "PDX has already been initialized" trên Colab.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..types import BoundingBox, TextLine
from ._paddle import get_engine, run_ocr
from .base import OCRBackend


class PaddleOCRBackend(OCRBackend):
    name = "paddle"

    def __init__(self, lang: str = "vi", use_angle_cls: bool = True, **kwargs):
        self._engine = get_engine(lang=lang, use_angle=use_angle_cls)

    def recognize(self, image: np.ndarray) -> List[TextLine]:
        lines: List[TextLine] = []
        for text, conf, (x0, y0, x1, y1) in run_ocr(self._engine, image):
            lines.append(
                TextLine(
                    text=text,
                    confidence=conf,
                    bbox=BoundingBox(x0, y0, x1, y1),
                    source="ocr",
                )
            )
        return lines
