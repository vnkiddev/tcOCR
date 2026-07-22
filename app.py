"""Gradio UI cho tcOCR — kéo-thả PDF/ảnh, đổi backend ngay trên giao diện.

Chạy local:   python app.py
Chạy Colab:   xem notebooks/tcOCR_colab.ipynb (dùng share=True để lấy link công khai)

⚠️ Data thật production KHÔNG chạy qua link public. UI này để test trên tài liệu
báo cáo tài chính CÔNG KHAI.
"""
from __future__ import annotations

import traceback
from typing import Optional

import numpy as np

from tcocr.config import PipelineConfig
from tcocr.correction import AVAILABLE_CORRECTION_BACKENDS
from tcocr.escalation import AVAILABLE_ESCALATION_BACKENDS
from tcocr.ocr import AVAILABLE_OCR_BACKENDS
from tcocr.pipeline import OCRPipeline
from tcocr.types import RegionType

# cache pipeline theo "chữ ký" config để đổi backend mới nạp lại
_CACHE = {"key": None, "pipeline": None}


def _get_pipeline(cfg: PipelineConfig) -> OCRPipeline:
    key = (
        cfg.ocr_backend,
        cfg.escalation_backend,
        cfg.correction_backend,
        cfg.escalate_confidence_threshold,
        cfg.enable_validation,
    )
    if _CACHE["key"] != key:
        _CACHE["pipeline"] = OCRPipeline(cfg, lazy=True)
        _CACHE["key"] = key
    return _CACHE["pipeline"]


def _render_page(page) -> str:
    out = [f"**Backend:** {page.ocr_backend} / esc={page.escalation_backend} / corr={page.correction_backend}"]
    out.append(f"**Confidence tối thiểu trang:** {page.min_confidence:.3f}\n")
    for i, r in enumerate(page.regions):
        out.append(f"#### Vùng {i} — {r.region_type.value}"
                   + (" _(đã escalate VLM)_" if r.escalated else ""))
        if r.table is not None:
            grid = r.table.to_matrix()
            if grid:
                out.append("| " + " | ".join(grid[0]) + " |")
                out.append("|" + "---|" * len(grid[0]))
                for row in grid[1:]:
                    out.append("| " + " | ".join(row) + " |")
        for ln in r.lines:
            tag = "🤖" if ln.source == "vlm" else ("✍️" if ln.source == "correction" else "")
            out.append(f"- {tag} {ln.text}  `({ln.confidence:.2f})`")
        out.append("")
    if page.validation_flags:
        out.append("### ⚠️ Validation flags (cần soát)")
        for f in page.validation_flags:
            out.append(f"- {f}")
    else:
        out.append("### ✅ Không có cờ validation")
    return "\n".join(out)


def run_ocr(
    file_obj,
    image_obj: Optional[np.ndarray],
    ocr_backend: str,
    escalation_backend: str,
    correction_backend: str,
    threshold: float,
    enable_validation: bool,
):
    try:
        cfg = PipelineConfig(
            ocr_backend=ocr_backend,
            escalation_backend=escalation_backend,
            correction_backend=correction_backend,
            escalate_confidence_threshold=threshold,
            enable_validation=enable_validation,
        )
        pipeline = _get_pipeline(cfg)

        if file_obj is not None:
            path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
            if path.lower().endswith(".pdf"):
                doc = pipeline.process_pdf(path)
                return "\n\n---\n\n".join(_render_page(p) for p in doc.pages)
            import cv2
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            return _render_page(pipeline.process_image(img))
        if image_obj is not None:
            return _render_page(pipeline.process_image(image_obj))
        return "⚠️ Hãy upload PDF hoặc ảnh."
    except Exception:
        return f"❌ Lỗi:\n```\n{traceback.format_exc()}\n```"


def run_benchmark(scan_pdf, gold_pdf, ocr_backend, escalation_backend, correction_backend, threshold):
    try:
        from tcocr.benchmark.runner import run_pair

        cfg = PipelineConfig(
            ocr_backend=ocr_backend,
            escalation_backend=escalation_backend,
            correction_backend=correction_backend,
            escalate_confidence_threshold=threshold,
        )
        pipeline = _get_pipeline(cfg)
        if scan_pdf is None or gold_pdf is None:
            return "⚠️ Cần cả PDF scan và PDF gốc (digital) để chấm điểm field-level."
        report = run_pair(pipeline, scan_pdf.name, gold_pdf.name)
        return report.to_markdown()
    except Exception:
        return f"❌ Lỗi:\n```\n{traceback.format_exc()}\n```"


def build_ui():
    import gradio as gr

    with gr.Blocks(title="tcOCR — OCR tài chính 3 lớp") as demo:
        gr.Markdown("# tcOCR — OCR tài chính chứng khoán (3 lớp, pluggable)")
        with gr.Row():
            ocr_dd = gr.Dropdown(AVAILABLE_OCR_BACKENDS, value="paddle", label="Lớp 1 · OCR")
            esc_dd = gr.Dropdown(AVAILABLE_ESCALATION_BACKENDS, value="null", label="Lớp 2 · Escalation (VLM)")
            corr_dd = gr.Dropdown(AVAILABLE_CORRECTION_BACKENDS, value="protonx", label="Lớp 3 · Sửa tiếng Việt")
        with gr.Row():
            thr = gr.Slider(0.0, 1.0, value=0.80, step=0.05, label="Ngưỡng escalate confidence")
            val = gr.Checkbox(value=True, label="Bật validation nghiệp vụ")

        with gr.Tab("Test 1 tài liệu"):
            with gr.Row():
                file_in = gr.File(label="PDF hoặc ảnh", file_types=[".pdf", ".png", ".jpg", ".jpeg"])
                img_in = gr.Image(label="Hoặc dán ảnh", type="numpy")
            btn = gr.Button("Chạy OCR", variant="primary")
            out = gr.Markdown()
            btn.click(run_ocr, [file_in, img_in, ocr_dd, esc_dd, corr_dd, thr, val], out)

        with gr.Tab("Benchmark field-level"):
            gr.Markdown("Upload **PDF scan** (đầu vào) và **PDF gốc digital** (nguồn gold).")
            with gr.Row():
                scan_in = gr.File(label="PDF scan", file_types=[".pdf"])
                gold_in = gr.File(label="PDF gốc (digital)", file_types=[".pdf"])
            bbtn = gr.Button("Chấm điểm", variant="primary")
            bout = gr.Markdown()
            bbtn.click(run_benchmark, [scan_in, gold_in, ocr_dd, esc_dd, corr_dd, thr], bout)

    return demo


if __name__ == "__main__":
    build_ui().launch()
