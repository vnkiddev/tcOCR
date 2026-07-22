"""Escalation qua API private do ANTT cấp (chỗ trống để điền sau).

Đây là backend dành cho lúc TCBS đã dựng endpoint LLM nội bộ/VPC cô lập.
Chỉ cần điền phần gọi HTTP cho khớp hợp đồng API của ANTT — phần còn lại của
pipeline không đổi vì vẫn tuân theo interface EscalationBackend.

QUAN TRỌNG: endpoint phải nằm trong hạ tầng nội bộ (data không ra ngoài).
"""
from __future__ import annotations

import base64
from typing import Optional

import numpy as np

from ..types import Region
from .base import EscalationBackend


class PrivateAPIBackend(EscalationBackend):
    name = "private_api"
    enabled = True

    def __init__(self, endpoint: str = "", api_key: str = "", model: str = "", **kwargs):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model

    @staticmethod
    def _encode_png(image: np.ndarray) -> str:
        import cv2

        ok, buf = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("Không encode được ảnh crop sang PNG")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def escalate(self, image_crop: np.ndarray, region: Region) -> Optional[Region]:
        if not self.endpoint:
            raise NotImplementedError(
                "Chưa cấu hình endpoint API của ANTT. Set escalation_kwargs={'endpoint': ...}. "
                "Điền phần gọi HTTP trong PrivateAPIBackend.escalate() theo hợp đồng API nội bộ."
            )
        # TODO(ANTT): gọi self.endpoint với ảnh base64 + prompt (xem local_vlm để lấy prompt),
        # parse kết quả về Region giống LocalVLMBackend. Giữ nguyên chữ số, temperature=0.
        raise NotImplementedError("Điền logic gọi API nội bộ ở đây.")
