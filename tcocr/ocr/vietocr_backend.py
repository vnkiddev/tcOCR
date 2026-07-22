"""OCR backend dùng VietOCR (recognizer transformer cho tiếng Việt).

VietOCR chỉ làm RECOGNITION trên ảnh 1 dòng, không có detector. Nên ta mượn
detector của PaddleOCR (chỉ dò box, tắt rec) rồi đưa từng crop cho VietOCR.
Cách này giữ detection giống nhau giữa 2 backend -> benchmark recognizer công bằng.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..types import BoundingBox, TextLine
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
        # --- Recognizer: VietOCR ---
        try:
            from vietocr.tool.config import Cfg
            from vietocr.tool.predictor import Predictor
        except ImportError as e:  # pragma: no cover
            raise ImportError("Chưa cài VietOCR. Colab: `pip install vietocr`") from e

        cfg = Cfg.load_config_from_name(model_name)
        cfg["device"] = device
        cfg["predictor"]["beamsearch"] = kwargs.get("beamsearch", False)
        self._predictor = Predictor(cfg)

        # --- Detector: mượn PaddleOCR (det only) ---
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "VietOCR cần detector. Cài kèm: `pip install paddlepaddle-gpu paddleocr`"
            ) from e
        self._det = PaddleOCR(
            use_angle_cls=False, lang=detector_lang, det=True, rec=False, show_log=False
        )

    def _detect_boxes(self, image: np.ndarray) -> List[BoundingBox]:
        result = self._det.ocr(image, det=True, rec=False, cls=False)
        boxes: List[BoundingBox] = []
        if not result:
            return boxes
        page = result[0] if len(result) == 1 and isinstance(result[0], list) else result
        if page is None:
            return boxes
        for box in page:
            pts = box if isinstance(box[0], (list, tuple)) else box[0]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            boxes.append(BoundingBox(min(xs), min(ys), max(xs), max(ys)))
        return boxes

    def recognize(self, image: np.ndarray) -> List[TextLine]:
        from PIL import Image

        lines: List[TextLine] = []
        for bb in self._detect_boxes(image):
            x0, y0, x1, y1 = (int(bb.x0), int(bb.y0), int(bb.x1), int(bb.y1))
            if x1 <= x0 or y1 <= y0:
                continue
            crop = image[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            pil = Image.fromarray(crop)
            text, prob = self._predictor.predict(pil, return_prob=True)
            lines.append(
                TextLine(text=text, confidence=float(prob), bbox=bb, source="ocr")
            )
        # sắp theo thứ tự đọc trên->dưới, trái->phải
        lines.sort(key=lambda l: (l.bbox.y0 if l.bbox else 0, l.bbox.x0 if l.bbox else 0))
        return lines

    def recognize_crop(self, image: np.ndarray) -> str:
        from PIL import Image

        if image.size == 0:
            return ""
        return self._predictor.predict(Image.fromarray(image))
