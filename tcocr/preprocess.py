"""Tiền xử lý ảnh trang: khử nghiêng (deskew) + khử nhiễu nhẹ.

Giữ tối giản, an toàn — chỉ can thiệp khi chắc chắn có lợi, tránh làm hỏng chữ số.
"""
from __future__ import annotations

import numpy as np


def deskew(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """Xoay thẳng trang dựa trên hướng text trội nhất. Bỏ qua nếu góc quá lớn (nghi nhiễu)."""
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if coords.shape[0] < 50:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.3 or abs(angle) > max_angle:
        return image
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def denoise(image: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21) if image.ndim == 3 else image


def preprocess_page(image: np.ndarray, do_deskew: bool = True, do_denoise: bool = False) -> np.ndarray:
    out = image
    if do_deskew:
        out = deskew(out)
    if do_denoise:
        out = denoise(out)
    return out
