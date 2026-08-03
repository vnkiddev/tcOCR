"""Cấu hình pipeline — mọi lựa chọn backend đều đổi được ở đây (hoặc trên UI).

Triết lý: KHÔNG hard-code backend. Mỗi lớp chọn bằng tên string, factory sẽ
dựng đúng backend tương ứng. Nhờ vậy switch Paddle<->VietOCR, bật/tắt VLM,
đổi ProtonX<->bmd1905 chỉ là đổi 1 trường config.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class PipelineConfig:
    # ---- Lớp 1: OCR truyền thống ----
    ocr_backend: str = "paddle"          # "paddle" | "vietocr"
    ocr_kwargs: Dict[str, Any] = field(default_factory=dict)

    # ---- Lớp 2: Escalation (Vision-LLM) ----
    escalation_backend: str = "null"     # "null" | "local_vlm" | "private_api"
    escalation_kwargs: Dict[str, Any] = field(default_factory=dict)
    # ngưỡng routing: vùng có confidence < ngưỡng thì đẩy lên VLM
    escalate_confidence_threshold: float = 0.80
    # bảng luôn escalate (OCR truyền thống làm bảng yếu)
    escalate_all_tables: bool = True
    # ô số parse không ra number thì escalate
    escalate_on_numeric_parse_fail: bool = True

    # ---- Lớp 3: Sửa tiếng Việt ----
    correction_backend: str = "protonx"  # "protonx" | "bmd1905" | "null"
    correction_kwargs: Dict[str, Any] = field(default_factory=dict)
    # CHỈ sửa text diễn giải, KHÔNG đụng số/bảng/mã
    correction_skip_numeric: bool = True

    # ---- Layout ----
    # PP-Structure: mặc định TẮT (né lỗi init lần 2 trên Colab). Bật khi on-prem
    # đã dựng được PP-Structure -> có tách cấu trúc bảng ở lớp 1.
    enable_layout_structure: bool = False

    # ---- Validation nghiệp vụ ----
    enable_validation: bool = True
    stock_code_whitelist_path: str = ""  # file danh sách mã CK (mỗi dòng 1 mã)

    # ---- Render ảnh từ PDF ----
    pdf_dpi: int = 200

    def summary(self) -> str:
        return (
            f"OCR={self.ocr_backend} | escalation={self.escalation_backend} "
            f"(thr={self.escalate_confidence_threshold}) | "
            f"correction={self.correction_backend} | validation={self.enable_validation}"
        )
