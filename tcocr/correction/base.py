"""Interface lớp 3 — Correction backend (sửa tiếng Việt).

Guardrail cốt lõi: CHỈ sửa text diễn giải. Trước khi đưa vào model, ta MASK
các token số/mã (số tiền, mã CK, ngày, %) thành placeholder, sửa xong khôi
phục lại nguyên văn -> model sửa lỗi không bao giờ đụng được vào con số.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import List, Tuple

# Bắt: số có phân cách (1.234.567,89 / 1,234,567.89), %, ngày, mã CK in hoa+số
_NUMERIC_PATTERN = re.compile(
    r"(?:\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b)"   # ngày (ưu tiên trước số)
    r"|(?:[-+(]?\d[\d.,]*\)?%?)"                # số, số âm, ngoặc kế toán, %
    r"|(?:\b[A-Z]{2,}\d[A-Z0-9]*\b)"           # mã kiểu VN30F2312
)


def mask_numeric(text: str) -> Tuple[str, List[str]]:
    """Thay token số/mã bằng ⟦0⟧ ⟦1⟧... trả về (text_đã_mask, danh_sách_gốc)."""
    tokens: List[str] = []

    def _sub(m: re.Match) -> str:
        tokens.append(m.group(0))
        return f"⟦{len(tokens) - 1}⟧"

    return _NUMERIC_PATTERN.sub(_sub, text), tokens


def unmask_numeric(text: str, tokens: List[str]) -> str:
    """Khôi phục các placeholder về nguyên văn con số/mã."""
    def _restore(m: re.Match) -> str:
        idx = int(m.group(1))
        return tokens[idx] if 0 <= idx < len(tokens) else m.group(0)

    return re.sub(r"⟦(\d+)⟧", _restore, text)


class CorrectionBackend(ABC):
    name: str = "base"

    @abstractmethod
    def _correct_raw(self, text: str) -> str:
        """Sửa 1 đoạn text (đã được mask số). Do backend cụ thể cài đặt."""
        raise NotImplementedError

    def correct(self, text: str, skip_numeric: bool = True) -> str:
        if not text or not text.strip():
            return text
        if not skip_numeric:
            return self._correct_raw(text)

        masked, tokens = mask_numeric(text)
        corrected = self._correct_raw(masked)
        # GUARDRAIL: tokenizer của model sửa lỗi có thể nghiền nát placeholder ⟦n⟧.
        # Nếu bất kỳ placeholder nào không sống sót NGUYÊN VẸN đúng 1 lần trong
        # output -> vứt kết quả sửa, giữ text gốc. Thà không sửa còn hơn mất số.
        for i in range(len(tokens)):
            if corrected.count(f"⟦{i}⟧") != 1:
                return text
        return unmask_numeric(corrected, tokens)
