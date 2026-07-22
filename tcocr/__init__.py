"""tcOCR — hệ thống OCR tài chính chứng khoán 3 lớp, độ chính xác cao.

Kiến trúc 3 lớp (pluggable):
  Lớp 1  OCR truyền thống + phát hiện cấu trúc  (PaddleOCR | VietOCR — switch được)
  Lớp 2  Escalate lên Vision-LLM khi cần         (Null | LocalVLM | PrivateAPI — bật/tắt)
  Lớp 3  Sửa tiếng Việt chỉ trên text            (ProtonX | bmd1905 — thay được)
  + Validation nghiệp vụ (chất keo kéo lên 99% field-level)
"""

from .types import (
    BoundingBox,
    TextLine,
    TableCell,
    Table,
    Region,
    RegionType,
    PageResult,
    DocumentResult,
)

__all__ = [
    "BoundingBox",
    "TextLine",
    "TableCell",
    "Table",
    "Region",
    "RegionType",
    "PageResult",
    "DocumentResult",
]

__version__ = "0.1.0"
