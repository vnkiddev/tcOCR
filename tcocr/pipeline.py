"""Pipeline chính — ghép 3 lớp + routing + validation.

Luồng mỗi trang:
    ảnh -> tiền xử lý -> [Lớp 1: layout + OCR] -> routing quyết định vùng nào khó
        -> [Lớp 2: VLM] cho vùng khó (nếu bật) -> [Lớp 3: sửa tiếng Việt] cho text
        -> validation nghiệp vụ -> PageResult

Backend của cả 3 lớp đều được dựng qua factory theo config, nên switch tùy ý.
"""
from __future__ import annotations

import os
from typing import List, Optional, Set

import numpy as np

from .config import PipelineConfig
from .correction import build_correction_backend
from .escalation import build_escalation_backend
from .layout import LayoutAnalyzer
from .ocr import build_ocr_backend
from .preprocess import preprocess_page
from .types import (
    DocumentResult,
    PageResult,
    Region,
    RegionType,
    TextLine,
)
from .validation import validate_page


class OCRPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None, lazy: bool = True):
        self.config = config or PipelineConfig()
        self._ocr = None
        self._escalation = None
        self._correction = None
        self._layout = None
        self._stock_whitelist: Optional[Set[str]] = None
        if not lazy:
            self._ensure_backends()

    # --- dựng backend (lazy để UI đổi config không phải nạp lại tất cả) ---
    def _ensure_backends(self) -> None:
        cfg = self.config
        if self._ocr is None:
            self._ocr = build_ocr_backend(cfg.ocr_backend, **cfg.ocr_kwargs)
        if self._escalation is None:
            self._escalation = build_escalation_backend(
                cfg.escalation_backend, **cfg.escalation_kwargs
            )
        if self._correction is None:
            self._correction = build_correction_backend(
                cfg.correction_backend, **cfg.correction_kwargs
            )
        if self._layout is None:
            self._layout = LayoutAnalyzer(use_structure=True)
        if self._stock_whitelist is None and cfg.stock_code_whitelist_path:
            self._stock_whitelist = _load_whitelist(cfg.stock_code_whitelist_path)

    # --- routing: vùng này có cần đẩy lên VLM không? ---
    def _needs_escalation(self, region: Region) -> bool:
        cfg = self.config
        if not self._escalation.enabled:
            return False
        if cfg.escalate_all_tables and region.region_type == RegionType.TABLE:
            return True
        if region.confidence < cfg.escalate_confidence_threshold:
            return True
        if cfg.escalate_on_numeric_parse_fail and region.region_type == RegionType.TABLE:
            return True
        return False

    @staticmethod
    def _crop(image: np.ndarray, region: Region) -> np.ndarray:
        b = region.bbox
        x0, y0, x1, y1 = int(b.x0), int(b.y0), int(b.x1), int(b.y1)
        h, w = image.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1 or w), min(h, y1 or h)
        if x1 <= x0 or y1 <= y0:
            return image
        return image[y0:y1, x0:x1]

    def process_page(self, image: np.ndarray, page_index: int = 0) -> PageResult:
        self._ensure_backends()
        cfg = self.config
        img = preprocess_page(image, do_deskew=True, do_denoise=False)
        h, w = img.shape[:2]

        page = PageResult(
            page_index=page_index,
            width=w,
            height=h,
            ocr_backend=cfg.ocr_backend,
            escalation_backend=cfg.escalation_backend,
            correction_backend=cfg.correction_backend,
        )

        # --- Lớp 1: layout + OCR ---
        regions = self._layout.analyze(img)
        for region in regions:
            crop = self._crop(img, region)
            if region.region_type == RegionType.TABLE and region.table is not None:
                # layout đã cho bảng (PP-Structure); confidence coi như trung bình thấp để dễ escalate
                region.confidence = region.table.confidence
            elif region.region_type in (RegionType.TEXT, RegionType.HEADER):
                lines = self._ocr.recognize(crop)
                region.lines = lines
                region.confidence = min((l.confidence for l in lines), default=1.0)

            # --- Lớp 2: escalate nếu cần ---
            if self._needs_escalation(region):
                try:
                    improved = self._escalation.escalate(crop, region)
                    if improved is not None:
                        region = improved
                except NotImplementedError:
                    page.validation_flags.append("escalation: backend chưa cấu hình")
                except Exception as e:  # pragma: no cover
                    page.validation_flags.append(f"escalation error: {e}")

            page.regions.append(region)

        # --- Lớp 3: sửa tiếng Việt CHỈ trên text (số/bảng bypass) ---
        if cfg.correction_backend not in ("null", "off", "none"):
            for region in page.regions:
                if region.region_type in (RegionType.TEXT, RegionType.HEADER):
                    for ln in region.lines:
                        corrected = self._correction.correct(
                            ln.text, skip_numeric=cfg.correction_skip_numeric
                        )
                        if corrected != ln.text:
                            ln.text = corrected
                            ln.source = "correction"

        # --- Validation nghiệp vụ ---
        if cfg.enable_validation:
            validate_page(page, self._stock_whitelist)

        return page

    def process_pdf(self, path: str) -> DocumentResult:
        from .pdfio import pdf_to_images

        images = pdf_to_images(path, dpi=self.config.pdf_dpi)
        doc = DocumentResult(source_path=path)
        for i, im in enumerate(images):
            doc.pages.append(self.process_page(im, page_index=i))
        return doc

    def process_image(self, image: np.ndarray) -> PageResult:
        return self.process_page(image, page_index=0)


def _load_whitelist(path: str) -> Set[str]:
    if not path or not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip().upper() for line in f if line.strip()}
