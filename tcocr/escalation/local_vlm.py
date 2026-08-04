"""Escalation bằng Vision-LLM self-host (mặc định Qwen2.5-VL).

Chạy on-prem / trên GPU thuê, KHÔNG gửi data ra cloud. Trên Colab free (T4)
nên dùng bản 3B; production 7B trên GPU 24GB+.

Prompt siết chặt: giữ nguyên con số từng ký tự, xuất có cấu trúc, không bịa.
"""
from __future__ import annotations

import json
import re
from typing import Optional

import numpy as np

from ..types import Region, RegionType, Table, TableCell, TextLine
from .base import EscalationBackend

_TABLE_PROMPT = (
    "Bạn là công cụ OCR tài chính chính xác tuyệt đối. Trích xuất BẢNG trong ảnh.\n"
    "QUY TẮC:\n"
    "- Giữ NGUYÊN từng chữ số, dấu phân cách nghìn, dấu âm/ngoặc. TUYỆT ĐỐI không làm tròn, không suy diễn.\n"
    "- Ô trống để chuỗi rỗng.\n"
    "- Chỉ trả về JSON: {\"rows\": [[\"ô\",\"ô\"], ...]}. Không giải thích."
)
_TEXT_PROMPT = (
    "Bạn là công cụ OCR tiếng Việt chính xác. Trích xuất toàn bộ text trong ảnh, "
    "giữ đúng dấu tiếng Việt và mọi con số. Chỉ trả về text thuần, không giải thích."
)


class LocalVLMBackend(EscalationBackend):
    name = "local_vlm"
    enabled = True

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 1536,
        load_in_4bit: bool = True,
        # Giới hạn vision token — QUAN TRỌNG trên T4 16GB: trang A4 200dpi mà không
        # giới hạn sẽ sinh cả chục nghìn token -> OOM. 1024*28*28 ≈ 1024 token/ảnh.
        max_pixels: int = 1024 * 28 * 28,
        min_pixels: int = 256 * 28 * 28,
        **kwargs,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        try:
            import torch  # noqa: F401
            from transformers import AutoProcessor
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration as _VLModel
            except ImportError:
                from transformers import Qwen2VLForConditionalGeneration as _VLModel
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Cần transformers + torch cho VLM. "
                "Colab: `pip install 'transformers>=4.49' accelerate qwen-vl-utils bitsandbytes`"
            ) from e

        model_kwargs = {"torch_dtype": "auto", "device_map": device}
        if load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                import torch

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
                model_kwargs.pop("torch_dtype", None)
            except Exception:
                pass  # không có bitsandbytes thì chạy full precision

        self._model = _VLModel.from_pretrained(model_id, **model_kwargs)
        self._processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=min_pixels, max_pixels=max_pixels
        )

    def _generate(self, image: np.ndarray, prompt: str) -> str:
        from PIL import Image

        pil = Image.fromarray(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[text], images=[pil], return_tensors="pt").to(
            self._model.device
        )
        out = self._model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
        )
        trimmed = out[:, inputs.input_ids.shape[1]:]
        return self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    @staticmethod
    def _parse_table_json(raw: str) -> Optional[Table]:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        rows = data.get("rows") or []
        if not rows:
            return None
        n_cols = max(len(r) for r in rows)
        cells = []
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                cells.append(TableCell(text=str(val), row=i, col=j))
        return Table(cells=cells, n_rows=len(rows), n_cols=n_cols, source="vlm", confidence=0.95)

    def escalate(self, image_crop: np.ndarray, region: Region) -> Optional[Region]:
        if region.region_type == RegionType.TABLE:
            table = self._parse_table_json(self._generate(image_crop, _TABLE_PROMPT))
            if table is None:
                return None
            region.table = table
            region.escalated = True
            region.confidence = table.confidence
            return region
        # text / header
        text = self._generate(image_crop, _TEXT_PROMPT)
        if not text:
            return None
        region.lines = [TextLine(text=t, confidence=0.95, source="vlm") for t in text.splitlines() if t.strip()]
        region.escalated = True
        region.confidence = 0.95
        return region
