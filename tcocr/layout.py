"""Layout analysis — chia trang thành vùng (text / bảng / header ...).

Dùng PaddleOCR PP-Structure nếu có (nhận cả cấu trúc bảng). Nếu không cài
được PP-Structure, fallback: coi cả trang là 1 vùng TEXT (pipeline vẫn chạy,
chỉ mất khả năng tách bảng — sẽ ghi cảnh báo).
"""
from __future__ import annotations

from typing import List

import numpy as np

from .types import BoundingBox, Region, RegionType, Table, TableCell


class LayoutAnalyzer:
    def __init__(self, lang: str = "vi", use_structure: bool = True, **kwargs):
        self._engine = None
        self._fallback = True
        if use_structure:
            try:
                from paddleocr import PPStructure

                self._engine = PPStructure(show_log=False, lang="en", **kwargs)
                self._fallback = False
            except Exception:
                self._engine = None
                self._fallback = True

    @staticmethod
    def _bbox_from_list(b) -> BoundingBox:
        return BoundingBox(float(b[0]), float(b[1]), float(b[2]), float(b[3]))

    def analyze(self, image: np.ndarray) -> List[Region]:
        if self._fallback or self._engine is None:
            h, w = image.shape[:2]
            return [Region(RegionType.TEXT, BoundingBox(0, 0, w, h))]

        regions: List[Region] = []
        for item in self._engine(image):
            box = item.get("bbox")
            rtype_raw = (item.get("type") or "").lower()
            bbox = self._bbox_from_list(box) if box is not None else BoundingBox(0, 0, 0, 0)
            if rtype_raw == "table":
                region = Region(RegionType.TABLE, bbox)
                region.table = self._parse_ppstructure_table(item)
                regions.append(region)
            elif rtype_raw in ("title", "header"):
                regions.append(Region(RegionType.HEADER, bbox))
            elif rtype_raw in ("figure", "image"):
                regions.append(Region(RegionType.FIGURE, bbox))
            else:
                regions.append(Region(RegionType.TEXT, bbox))
        if not regions:
            h, w = image.shape[:2]
            regions.append(Region(RegionType.TEXT, BoundingBox(0, 0, w, h)))
        return regions

    @staticmethod
    def _parse_ppstructure_table(item) -> Table:
        """PP-Structure trả HTML bảng ở item['res']['html']. Parse thô về ma trận ô."""
        res = item.get("res") or {}
        html = res.get("html", "") if isinstance(res, dict) else ""
        rows = _html_table_to_rows(html)
        cells: List[TableCell] = []
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                cells.append(TableCell(text=val, row=i, col=j))
        n_cols = max((len(r) for r in rows), default=0)
        return Table(cells=cells, n_rows=len(rows), n_cols=n_cols, source="ocr")


def _html_table_to_rows(html: str) -> List[List[str]]:
    if not html:
        return []
    try:
        from html.parser import HTMLParser

        class _P(HTMLParser):
            def __init__(self):
                super().__init__()
                self.rows: List[List[str]] = []
                self._cur: List[str] = []
                self._buf = ""
                self._in_cell = False

            def handle_starttag(self, tag, attrs):
                if tag == "tr":
                    self._cur = []
                elif tag in ("td", "th"):
                    self._in_cell = True
                    self._buf = ""

            def handle_endtag(self, tag):
                if tag in ("td", "th"):
                    self._cur.append(self._buf.strip())
                    self._in_cell = False
                elif tag == "tr":
                    self.rows.append(self._cur)

            def handle_data(self, data):
                if self._in_cell:
                    self._buf += data

        p = _P()
        p.feed(html)
        return [r for r in p.rows if r]
    except Exception:
        return []
