"""Escalation TẮT — pipeline chỉ chạy OCR truyền thống + validation.

Dùng để đo baseline (accuracy khi không có VLM) và để chạy khi chưa có
API của ANTT / chưa muốn tốn GPU cho VLM.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..types import Region
from .base import EscalationBackend


class NullEscalationBackend(EscalationBackend):
    name = "null"
    enabled = False

    def escalate(self, image_crop: np.ndarray, region: Region) -> Optional[Region]:
        return None
