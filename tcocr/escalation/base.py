"""Interface lớp 2 — Escalation backend.

Khi lớp 1 không chắc (confidence thấp / là bảng / số parse lỗi), vùng đó được
đẩy lên đây. Backend nhận ảnh crop của vùng + gợi ý loại vùng, trả về text /
bảng chính xác hơn. Thiết kế cắm-rút: Null (tắt) / LocalVLM / PrivateAPI(ANTT).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..types import Region


class EscalationBackend(ABC):
    name: str = "base"
    enabled: bool = True

    @abstractmethod
    def escalate(self, image_crop: np.ndarray, region: Region) -> Optional[Region]:
        """Trả về Region đã cải thiện, hoặc None nếu không xử lý được (giữ nguyên lớp 1)."""
        raise NotImplementedError
