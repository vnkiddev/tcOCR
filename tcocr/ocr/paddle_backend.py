"""OCR backend dùng PaddleOCR (detection + recognition, hỗ trợ tiếng Việt).

Import nặng được lazy-load trong __init__ để `import tcocr` không đòi cài Paddle.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..types import BoundingBox, TextLine
from .base import OCRBackend


class PaddleOCRBackend(OCRBackend):
    name = "paddle"

    def __init__(self, lang: str = "vi", use_angle_cls: bool = True, **kwargs):
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Chưa cài PaddleOCR. Colab: `pip install paddlepaddle-gpu paddleocr`"
            ) from e
        # show_log=False cho đỡ rác log; các tham số khác truyền thẳng.
        self._ocr = PaddleOCR(
            use_angle_cls=use_angle_cls,
            lang=lang,
            show_log=kwargs.pop("show_log", False),
            **kwargs,
        )

    def recognize(self, image: np.ndarray) -> List[TextLine]:
        result = self._ocr.ocr(image, cls=True)
        lines: List[TextLine] = []
        # PaddleOCR trả về [[ [box, (text, conf)], ... ]] (bọc 1 lớp theo page)
        if not result:
            return lines
        page = result[0] if len(result) == 1 and isinstance(result[0], list) else result
        if page is None:
            return lines
        for item in page:
            try:
                box, (text, conf) = item
            except (ValueError, TypeError):
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            lines.append(
                TextLine(
                    text=text,
                    confidence=float(conf),
                    bbox=BoundingBox(min(xs), min(ys), max(xs), max(ys)),
                    source="ocr",
                )
            )
        return lines
