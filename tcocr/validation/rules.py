"""Validation nghiệp vụ — chất keo kéo độ chính xác lên 99% field-level.

Không dùng model, chỉ dùng LUẬT tận dụng cấu trúc tài chính:
  1. Số parse được không?                (số hỏng -> cần soát / re-escalate)
  2. Dòng "Tổng/Cộng" có khớp tổng cột?   (bắt lỗi chữ số mà confidence bỏ sót)
  3. Mã CK có trong danh mục HOSE/HNX/UPCOM?
  4. Ngày đúng định dạng?

Field nào fail được đưa vào page.validation_flags để đẩy sang human review
hoặc thử lại lớp 2.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Set

from ..types import PageResult, RegionType, Table

_TOTAL_KEYWORDS = ("tổng", "cộng", "total", "tổng cộng", "tổng số")


def _fold_diacritics(s: str) -> str:
    """'Tổng cộng' -> 'tong cong' — OCR hay mất dấu, phải match được cả dạng này."""
    import unicodedata

    return (
        unicodedata.normalize("NFD", s.lower())
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def _is_total_line(text: str) -> bool:
    low = text.lower()
    if any(k in low for k in _TOTAL_KEYWORDS):
        return True
    # dạng mất dấu: chỉ match từ an toàn ("tong", "total") — KHÔNG match "cong"
    # trần vì "cộng" fold trùng với "công" (công ty) -> false positive
    return bool(re.search(r"\btong\b|\btotal\b", _fold_diacritics(text)))
_DATE_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b")
_STOCK_RE = re.compile(r"\b[A-Z]{3}\b")


@dataclass
class ValidationFlag:
    kind: str          # numeric_parse | sum_mismatch | bad_stock_code | bad_date
    detail: str
    region_index: int = -1


def parse_vn_number(s: str) -> Optional[float]:
    """Parse số kiểu VN/quốc tế. Ngoặc () = âm (kế toán). Trả None nếu không phải số."""
    if s is None:
        return None
    t = s.strip()
    if not t:
        return None
    neg = False
    if t.startswith("(") and t.endswith(")"):
        neg, t = True, t[1:-1].strip()
    if t.startswith("-"):
        neg, t = True, t[1:].strip()
    t = t.replace("%", "").replace(" ", "")
    if not re.fullmatch(r"[\d.,]+", t):
        return None
    # Heuristic phân biệt dấu thập phân: dấu cuối cùng trong {, .} là thập phân nếu
    # theo sau <=2 chữ số và chỉ xuất hiện 1 lần kiểu đó.
    if "," in t and "." in t:
        dec_sep = "," if t.rfind(",") > t.rfind(".") else "."
        thou_sep = "." if dec_sep == "," else ","
        t = t.replace(thou_sep, "").replace(dec_sep, ".")
    elif "," in t:
        # nếu ',' đứng cách cuối 3 chữ số và có nhiều nhóm -> phân cách nghìn
        if re.fullmatch(r"\d{1,3}(,\d{3})+", t):
            t = t.replace(",", "")
        else:
            t = t.replace(",", ".")
    elif "." in t:
        if re.fullmatch(r"\d{1,3}(\.\d{3})+", t):
            t = t.replace(".", "")
    try:
        val = float(t)
        return -val if neg else val
    except ValueError:
        return None


def _validate_table(table: Table, region_index: int, tol: float = 1.0) -> List[ValidationFlag]:
    flags: List[ValidationFlag] = []
    grid = table.to_matrix()
    if not grid:
        return flags

    # 1) đánh dấu ô "trông giống số" nhưng parse fail
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell and re.search(r"\d", cell) and re.search(r"[.,]", cell):
                if parse_vn_number(cell) is None:
                    flags.append(
                        ValidationFlag("numeric_parse", f"[{r},{c}]={cell!r}", region_index)
                    )

    # 2) kiểm dòng tổng khớp tổng các dòng phía trên (theo từng cột số)
    total_rows = [
        r for r, row in enumerate(grid)
        if any(_is_total_line(cell or "") for cell in row)
    ]
    for tr in total_rows:
        for c in range(len(grid[tr])):
            declared = parse_vn_number(grid[tr][c])
            if declared is None:
                continue
            above = [parse_vn_number(grid[r][c]) for r in range(tr) if c < len(grid[r])]
            nums = [x for x in above if x is not None]
            if len(nums) >= 2 and abs(sum(nums) - declared) > tol:
                flags.append(
                    ValidationFlag(
                        "sum_mismatch",
                        f"cột {c}: tổng khai báo {declared} ≠ cộng dồn {sum(nums)}",
                        region_index,
                    )
                )
    return flags


_LINE_NUM_RE = re.compile(r"([-+(]?\d[\d.,]*\)?)\s*$")


def _validate_text_lines(lines: List[str], region_index: int, tol: float = 1.0) -> List[ValidationFlag]:
    """Sum-check trên TEXT lines — quan trọng khi layout/bảng TẮT (mặc định Colab).

    Không có cấu trúc bảng thì mỗi dòng 'Chỉ tiêu ... 500.000' vẫn kết thúc bằng
    một con số. Heuristic: dòng chứa 'tổng/cộng' có số cuối N -> so N với tổng
    các số cuối dòng của các dòng liền trước (từ sau dòng tổng gần nhất).
    Chỉ flag khi có >=2 số cộng dồn — hạn chế false positive.
    """
    flags: List[ValidationFlag] = []
    acc: List[float] = []
    for i, line in enumerate(lines):
        m = _LINE_NUM_RE.search(line.strip())
        val = parse_vn_number(m.group(1)) if m else None
        is_total = _is_total_line(line)
        if is_total and val is not None:
            if len(acc) >= 2 and abs(sum(acc) - val) > tol:
                flags.append(
                    ValidationFlag(
                        "sum_mismatch",
                        f"dòng {i} ({line.strip()[:40]!r}): tổng khai báo {val} ≠ cộng dồn {sum(acc)}",
                        region_index,
                    )
                )
            acc = []  # reset sau mỗi dòng tổng
        elif val is not None:
            acc.append(val)
        else:
            acc = []  # dòng không có số cuối -> ngắt chuỗi cộng dồn
    return flags


def validate_page(
    page: PageResult, stock_whitelist: Optional[Set[str]] = None
) -> List[ValidationFlag]:
    flags: List[ValidationFlag] = []
    for idx, region in enumerate(page.regions):
        if region.region_type == RegionType.TABLE and region.table is not None:
            flags.extend(_validate_table(region.table, idx))
        elif region.lines:
            flags.extend(
                _validate_text_lines([l.text for l in region.lines], idx)
            )

        text = " ".join(l.text for l in region.lines)

        # ngày sai format cơ bản (tháng>12 / ngày>31)
        for d, m, _y in _DATE_RE.findall(text):
            if int(m) > 12 or int(d) > 31:
                flags.append(ValidationFlag("bad_date", f"{d}/{m}", idx))

        # mã CK không có trong whitelist (chỉ cảnh báo khi có whitelist)
        if stock_whitelist:
            for code in _STOCK_RE.findall(text):
                if code not in stock_whitelist and code not in _TOTAL_KEYWORDS:
                    # tránh false positive với từ in hoa thường gặp
                    if code not in ("VND", "USD", "CTY", "TNHH", "MCK"):
                        flags.append(ValidationFlag("bad_stock_code", code, idx))

    page.validation_flags = [f"{f.kind}: {f.detail}" for f in flags]
    return flags
