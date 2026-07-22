"""Trích ground-truth (gold) từ bản DOC GỐC digital.

Mày có bản gốc digital + bản scan: bản scan là ĐẦU VÀO cho OCR, bản gốc là
NGUỒN nhãn. Dùng pdfplumber lấy text + bảng có cấu trúc từ bản gốc -> gold
field-level, khỏi label tay.

Lưu ý: bản gốc và bản scan phải cùng nội dung/thứ tự trang. Nếu bản gốc là
ảnh (không có text layer) thì không trích tự động được — khi đó cần nhãn tay.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class GoldPage:
    page_index: int
    text: str = ""
    tables: List[List[List[str]]] = field(default_factory=list)  # list[table[row[cell]]]

    def numbers(self) -> List[str]:
        """Tất cả token số trong text + bảng (để chấm điểm field-level trên số)."""
        import re

        pat = re.compile(r"[-+(]?\d[\d.,]*\)?%?")
        found = pat.findall(self.text)
        for tb in self.tables:
            for row in tb:
                for cell in row:
                    found.extend(pat.findall(cell or ""))
        return found


@dataclass
class GoldDocument:
    source_path: str = ""
    pages: List[GoldPage] = field(default_factory=list)


def extract_gold_from_pdf(path: str, has_text_layer: Optional[bool] = None) -> GoldDocument:
    """Trích gold từ PDF gốc digital bằng pdfplumber."""
    try:
        import pdfplumber
    except ImportError as e:  # pragma: no cover
        raise ImportError("Cần pdfplumber để trích gold: `pip install pdfplumber`") from e

    doc = GoldDocument(source_path=path)
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = []
            for tb in page.extract_tables() or []:
                tables.append([[(c or "").strip() for c in row] for row in tb])
            gp = GoldPage(page_index=i, text=text, tables=tables)
            doc.pages.append(gp)

    total_text = sum(len(p.text) for p in doc.pages)
    if total_text < 20 and has_text_layer is not False:
        # gần như không có text -> bản gốc có thể là ảnh, cảnh báo
        import warnings

        warnings.warn(
            f"{path}: trích được rất ít text — bản gốc có thể KHÔNG có text layer "
            "(là ảnh). Gold tự động sẽ không chính xác; cân nhắc nhãn tay."
        )
    return doc
