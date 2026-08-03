"""OCR backend dùng VietOCR (recognizer transformer cho tiếng Việt).

VietOCR chỉ RECOGNITION trên ảnh 1 dòng. Detector mượn của PaddleOCR (qua
singleton _paddle nên không đụng lỗi init). Detection giống nhau giữa 2 backend
-> benchmark recognizer công bằng.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..types import BoundingBox, TextLine
from ._paddle import detect_boxes, get_engine
from .base import OCRBackend


class VietOCRBackend(OCRBackend):
    name = "vietocr"

    def __init__(
        self,
        model_name: str = "vgg_transformer",
        device: str = "cuda",
        detector_lang: str = "vi",
        **kwargs,
    ):
        try:
            from vietocr.tool.config import Cfg
            from vietocr.tool.predictor import Predictor
        except ImportError as e:  # pragma: no cover
            raise ImportError("Chưa cài VietOCR. Colab: `pip install vietocr`") from e

        cfg = Cfg.load_config_from_name(model_name)
        # tự lùi về CPU nếu không có GPU để khỏi crash
        try:
            import torch

            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
        except ImportError:
            device = "cpu"
        cfg["device"] = device
        cfg["predictor"]["beamsearch"] = kwargs.get("beamsearch", False)
        self._predictor = Predictor(cfg)
        self._det = get_engine(lang=detector_lang, use_angle=False, det_only=True)

    def recognize(self, image: np.ndarray) -> List[TextLine]:
        from PIL import Image

        lines: List[TextLine] = []
        for (x0, y0, x1, y1) in detect_boxes(self._det, image):
            xi0, yi0, xi1, yi1 = int(x0), int(y0), int(x1), int(y1)
            if xi1 <= xi0 or yi1 <= yi0:
                continue
            crop = image[yi0:yi1, xi0:xi1]
            if crop.size == 0:
                continue
            text, prob = self._predictor.predict(Image.fromarray(crop), return_prob=True)
            lines.append(
                TextLine(
                    text=text,
                    confidence=float(prob),
                    bbox=BoundingBox(x0, y0, x1, y1),
                    source="ocr",
                )
            )
        lines.sort(key=lambda l: (l.bbox.y0 if l.bbox else 0, l.bbox.x0 if l.bbox else 0))
        return lines

    def recognize_crop(self, image: np.ndarray) -> str:
        from PIL import Image

        if image.size == 0:
            return ""
        return self._predictor.predict(Image.fromarray(image))
