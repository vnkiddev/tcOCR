"""Registry lớp 1 — chọn OCR backend theo tên."""
from __future__ import annotations

from .base import OCRBackend


def build_ocr_backend(name: str, **kwargs) -> OCRBackend:
    name = (name or "").lower()
    if name == "paddle":
        from .paddle_backend import PaddleOCRBackend

        return PaddleOCRBackend(**kwargs)
    if name == "vietocr":
        from .vietocr_backend import VietOCRBackend

        return VietOCRBackend(**kwargs)
    raise ValueError(f"OCR backend không hỗ trợ: {name!r} (chọn 'paddle' | 'vietocr')")


AVAILABLE_OCR_BACKENDS = ["paddle", "vietocr"]

__all__ = ["OCRBackend", "build_ocr_backend", "AVAILABLE_OCR_BACKENDS"]
