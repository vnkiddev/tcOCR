"""Đọc PDF scan -> ảnh từng trang (numpy RGB). Ưu tiên PyMuPDF (không cần poppler)."""
from __future__ import annotations

from typing import List

import numpy as np


def pdf_to_images(path: str, dpi: int = 200) -> List[np.ndarray]:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return _pdf_to_images_pdf2image(path, dpi)

    images: List[np.ndarray] = []
    zoom = dpi / 72.0
    with fitz.open(path) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                arr = arr[:, :, :3]
            images.append(np.ascontiguousarray(arr))
    return images


def _pdf_to_images_pdf2image(path: str, dpi: int) -> List[np.ndarray]:
    from pdf2image import convert_from_path

    return [np.array(im.convert("RGB")) for im in convert_from_path(path, dpi=dpi)]
