"""Interface lớp 1 — OCR backend. Paddle và VietOCR đều implement cái này
nên pipeline switch qua lại mà không cần biết bên trong."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from ..types import TextLine


class OCRBackend(ABC):
    """Nhận ảnh (numpy BGR/RGB) -> danh sách dòng text kèm bbox + confidence."""

    name: str = "base"

    @abstractmethod
    def recognize(self, image: np.ndarray) -> List[TextLine]:
        """Chạy detection + recognition trên toàn ảnh, trả về các dòng text."""
        raise NotImplementedError

    def recognize_crop(self, image: np.ndarray) -> str:
        """Nhận dạng nhanh 1 crop (1 ô bảng / 1 dòng). Mặc định gộp text các dòng."""
        lines = self.recognize(image)
        return " ".join(l.text for l in lines).strip()

    def warmup(self) -> None:
        """Nạp model trước cho lần chạy đầu đỡ chậm (tùy chọn)."""
        return None
