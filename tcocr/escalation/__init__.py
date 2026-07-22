"""Registry lớp 2 — chọn escalation backend theo tên."""
from __future__ import annotations

from .base import EscalationBackend


def build_escalation_backend(name: str, **kwargs) -> EscalationBackend:
    name = (name or "null").lower()
    if name in ("null", "off", "none"):
        from .null_backend import NullEscalationBackend

        return NullEscalationBackend()
    if name in ("local_vlm", "vlm", "qwen"):
        from .local_vlm import LocalVLMBackend

        return LocalVLMBackend(**kwargs)
    if name in ("private_api", "antt"):
        from .private_api import PrivateAPIBackend

        return PrivateAPIBackend(**kwargs)
    raise ValueError(
        f"Escalation backend không hỗ trợ: {name!r} (chọn 'null' | 'local_vlm' | 'private_api')"
    )


AVAILABLE_ESCALATION_BACKENDS = ["null", "local_vlm", "private_api"]

__all__ = ["EscalationBackend", "build_escalation_backend", "AVAILABLE_ESCALATION_BACKENDS"]
