"""Kiểu dữ liệu dùng chung xuyên suốt pipeline.

Thiết kế theo hướng "có cấu trúc": mọi lớp trả về cùng một dạng dữ liệu
(text lines + tables + confidence + bbox) để routing và benchmark field-level
làm việc được nhất quán.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


@dataclass
class BoundingBox:
    """Toạ độ vùng trên ảnh trang (pixel)."""
    x0: float
    y0: float
    x1: float
    y1: float

    def as_tuple(self) -> tuple:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass
class TextLine:
    """Một dòng text nhận dạng được."""
    text: str
    confidence: float = 1.0
    bbox: Optional[BoundingBox] = None
    source: str = "ocr"  # ocr | vlm | correction — dấu vết đã đi qua lớp nào


@dataclass
class TableCell:
    text: str
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    confidence: float = 1.0
    is_numeric: bool = False


@dataclass
class Table:
    """Bảng dạng lưới hàng x cột."""
    cells: List[TableCell] = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0
    bbox: Optional[BoundingBox] = None
    confidence: float = 1.0
    source: str = "ocr"

    def to_matrix(self) -> List[List[str]]:
        """Trả về ma trận text để hiển thị / so khớp."""
        grid = [["" for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        for c in self.cells:
            if 0 <= c.row < self.n_rows and 0 <= c.col < self.n_cols:
                grid[c.row][c.col] = c.text
        return grid


class RegionType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    HEADER = "header"
    FIGURE = "figure"
    SIGNATURE = "signature"
    UNKNOWN = "unknown"


@dataclass
class Region:
    """Một vùng trên trang sau bước layout analysis."""
    region_type: RegionType
    bbox: BoundingBox
    confidence: float = 1.0
    # nội dung được điền dần qua các lớp:
    lines: List[TextLine] = field(default_factory=list)
    table: Optional[Table] = None
    escalated: bool = False  # đã đẩy qua lớp 2 (VLM) chưa


@dataclass
class PageResult:
    page_index: int
    width: int = 0
    height: int = 0
    regions: List[Region] = field(default_factory=list)
    # metadata benchmark / debug
    ocr_backend: str = ""
    escalation_backend: str = ""
    correction_backend: str = ""
    validation_flags: List[str] = field(default_factory=list)

    @property
    def min_confidence(self) -> float:
        confs = [r.confidence for r in self.regions] or [1.0]
        return min(confs)

    def full_text(self) -> str:
        parts: List[str] = []
        for r in self.regions:
            if r.table is not None:
                for row in r.table.to_matrix():
                    parts.append("\t".join(row))
            for ln in r.lines:
                parts.append(ln.text)
        return "\n".join(parts)


@dataclass
class DocumentResult:
    source_path: str = ""
    pages: List[PageResult] = field(default_factory=list)

    def full_text(self) -> str:
        return "\n\n".join(p.full_text() for p in self.pages)
