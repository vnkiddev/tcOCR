"""Benchmark runner — la bàn để tiến tới 99%.

Cho 1 cặp (PDF scan, PDF gốc) hoặc cả thư mục, chạy pipeline và chấm điểm
field-level. Mỗi khi đổi backend (Paddle<->VietOCR, bật/tắt VLM, đổi correction)
chạy lại cái này để thấy accuracy lên/xuống ngay.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..gold.extract import extract_gold_from_pdf
from ..pipeline import OCRPipeline
from .metrics import PageMetrics, evaluate_page


@dataclass
class BenchmarkReport:
    per_page: List[PageMetrics] = field(default_factory=list)
    config_summary: str = ""

    def aggregate(self) -> Dict[str, float]:
        if not self.per_page:
            return {}
        n = len(self.per_page)
        return {
            "pages": n,
            "avg_CER": round(sum(m.cer for m in self.per_page) / n, 4),
            "avg_number_acc": round(sum(m.number_accuracy for m in self.per_page) / n, 4),
            "avg_table_cell_acc": round(
                sum(m.table_cell_accuracy for m in self.per_page) / n, 4
            ),
            "total_flags": sum(m.n_flags for m in self.per_page),
        }

    def to_markdown(self) -> str:
        agg = self.aggregate()
        lines = [
            f"### Benchmark — `{self.config_summary}`",
            "",
            "| page | CER | number_acc | table_cell_acc | n_numbers | n_flags |",
            "|---|---|---|---|---|---|",
        ]
        for m in self.per_page:
            d = m.as_dict()
            lines.append(
                f"| {d['page']} | {d['CER']} | {d['number_acc']} | "
                f"{d['table_cell_acc']} | {d['n_numbers']} | {d['n_flags']} |"
            )
        lines += [
            "",
            f"**Tổng hợp:** {agg}",
            "",
            f"➡️ number_acc & table_cell_acc là 2 chỉ số quyết định target 99% field-level.",
        ]
        return "\n".join(lines)


def run_pair(
    pipeline: OCRPipeline, scan_pdf: str, gold_pdf: str
) -> BenchmarkReport:
    report = BenchmarkReport(config_summary=pipeline.config.summary())
    pred_doc = pipeline.process_pdf(scan_pdf)
    gold_doc = extract_gold_from_pdf(gold_pdf)
    n = min(len(pred_doc.pages), len(gold_doc.pages))
    for i in range(n):
        report.per_page.append(evaluate_page(pred_doc.pages[i], gold_doc.pages[i]))
    return report


def _pair_files(scan_dir: str, gold_dir: str) -> List[Tuple[str, str]]:
    pairs = []
    for scan in sorted(glob.glob(os.path.join(scan_dir, "*.pdf"))):
        base = os.path.basename(scan)
        gold = os.path.join(gold_dir, base)
        if os.path.exists(gold):
            pairs.append((scan, gold))
    return pairs


def run_dir(pipeline: OCRPipeline, scan_dir: str, gold_dir: str) -> BenchmarkReport:
    """So khớp file theo tên trùng nhau giữa 2 thư mục scan/ và gold/."""
    report = BenchmarkReport(config_summary=pipeline.config.summary())
    for scan, gold in _pair_files(scan_dir, gold_dir):
        r = run_pair(pipeline, scan, gold)
        report.per_page.extend(r.per_page)
    return report
