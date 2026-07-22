"""Metrics benchmark — trọng tâm FIELD-LEVEL (số + ô bảng), không chỉ CER.

Với tài chính, 99% phải đo trên field: 1 chữ số sai = cả field sai. CER chỉ là
phụ trợ cho phần text diễn giải.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from ..gold.extract import GoldPage
from ..types import PageResult, RegionType
from ..validation.rules import parse_vn_number

_NUM_RE = re.compile(r"[-+(]?\d[\d.,]*\)?%?")


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, gold: str) -> float:
    """Character Error Rate (0 = hoàn hảo)."""
    gold = gold or ""
    if not gold:
        return 0.0 if not pred else 1.0
    return _levenshtein(pred or "", gold) / len(gold)


def _canon_number(tok: str):
    v = parse_vn_number(tok)
    return v if v is not None else tok.strip()


def number_accuracy(pred_text: str, gold_numbers: List[str]) -> float:
    """Tỉ lệ con số trong gold khớp được trong pred (so theo giá trị đã chuẩn hóa).

    Đây là metric field-level quan trọng nhất cho tài chính.
    """
    if not gold_numbers:
        return 1.0
    pred_nums = [_canon_number(t) for t in _NUM_RE.findall(pred_text or "")]
    pred_pool = list(pred_nums)
    matched = 0
    for g in gold_numbers:
        gv = _canon_number(g)
        if gv in pred_pool:
            pred_pool.remove(gv)
            matched += 1
    return matched / len(gold_numbers)


def table_cell_accuracy(pred_tables: List[List[List[str]]], gold_tables: List[List[List[str]]]) -> float:
    """Tỉ lệ ô bảng khớp chính xác (chuẩn hóa số). Ghép bảng theo thứ tự."""
    if not gold_tables:
        return 1.0
    total = matched = 0
    for gi, gtb in enumerate(gold_tables):
        ptb = pred_tables[gi] if gi < len(pred_tables) else []
        for r, grow in enumerate(gtb):
            for c, gcell in enumerate(grow):
                total += 1
                pcell = ""
                if r < len(ptb) and c < len(ptb[r]):
                    pcell = ptb[r][c]
                gv, pv = _canon_number(gcell), _canon_number(pcell)
                if gv == pv or (gcell or "").strip() == (pcell or "").strip():
                    matched += 1
    return matched / total if total else 1.0


@dataclass
class PageMetrics:
    page_index: int
    cer: float
    number_accuracy: float
    table_cell_accuracy: float
    n_gold_numbers: int
    n_flags: int

    def as_dict(self) -> dict:
        return {
            "page": self.page_index,
            "CER": round(self.cer, 4),
            "number_acc": round(self.number_accuracy, 4),
            "table_cell_acc": round(self.table_cell_accuracy, 4),
            "n_numbers": self.n_gold_numbers,
            "n_flags": self.n_flags,
        }


def _pred_tables(page: PageResult) -> List[List[List[str]]]:
    return [
        r.table.to_matrix()
        for r in page.regions
        if r.region_type == RegionType.TABLE and r.table is not None
    ]


def evaluate_page(pred: PageResult, gold: GoldPage) -> PageMetrics:
    pred_text = pred.full_text()
    return PageMetrics(
        page_index=gold.page_index,
        cer=cer(pred_text, gold.text),
        number_accuracy=number_accuracy(pred_text, gold.numbers()),
        table_cell_accuracy=table_cell_accuracy(_pred_tables(pred), gold.tables),
        n_gold_numbers=len(gold.numbers()),
        n_flags=len(pred.validation_flags),
    )
