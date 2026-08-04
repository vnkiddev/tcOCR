"""Helper dùng chung cho mọi thứ đụng tới PaddleOCR.

Giải quyết 2 vấn đề trên Colab:
  1) "PDX has already been initialized": PaddleOCR 3.x (nền PaddleX) chỉ được
     init 1 lần / process. Ta dùng SINGLETON theo (lang, use_angle, det_only)
     nên không bao giờ tạo 2 instance -> hết lỗi.
  2) API 2.x vs 3.x khác nhau: bọc lại thành get_engine / run_ocr / detect_boxes
     tự nhận diện phiên bản, phần còn lại của code không phải quan tâm.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

_INSTANCES: dict = {}
_VERSION_MAJOR = None


def paddle_version_major() -> int:
    global _VERSION_MAJOR
    if _VERSION_MAJOR is not None:
        return _VERSION_MAJOR
    try:
        import paddleocr

        v = str(getattr(paddleocr, "__version__", "2"))
        _VERSION_MAJOR = int(v.split(".")[0]) if v[:1].isdigit() else 2
    except Exception:
        _VERSION_MAJOR = 2
    return _VERSION_MAJOR


def get_engine(lang: str = "vi", use_angle: bool = True, det_only: bool = False):
    """Trả về 1 instance PaddleOCR duy nhất cho mỗi cấu hình (singleton)."""
    key = (lang, use_angle, det_only)
    if key in _INSTANCES:
        return _INSTANCES[key]

    try:
        import paddle  # framework — thường là thủ phạm khi "cài rồi vẫn báo thiếu"
    except Exception as e:  # pragma: no cover
        raise ImportError(
            f"Không import được paddlepaddle ({type(e).__name__}: {e}).\n"
            "Trên Colab dùng bản CPU cho chắc (paddlepaddle-gpu trên PyPI hay lỗi build):\n"
            "  1) !pip uninstall -y paddlepaddle-gpu paddlepaddle\n"
            "  2) !pip install paddlepaddle paddleocr\n"
            "  3) Runtime -> Restart session, rồi chạy lại từ đầu."
        ) from e

    try:
        from paddleocr import PaddleOCR
    except Exception as e:  # pragma: no cover
        raise ImportError(
            f"Không import được PaddleOCR ({type(e).__name__}: {e}).\n"
            "Thử: !pip install -U paddleocr rồi Runtime -> Restart session."
        ) from e

    if paddle_version_major() >= 3:
        if det_only:
            # 3.x có module detection riêng — nhẹ hơn hẳn full pipeline
            # (nếu không có thì fallback xuống full PaddleOCR bên dưới).
            try:
                from paddleocr import TextDetection

                eng = TextDetection()
                _INSTANCES[key] = eng
                return eng
            except Exception:
                pass
        # API 3.x: bỏ use_angle_cls/show_log, tắt các module tiền xử lý cho nhẹ.
        eng = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=use_angle,
        )
    else:
        kwargs = dict(use_angle_cls=use_angle, lang=lang, show_log=False)
        if det_only:
            kwargs.update(det=True, rec=False)
        eng = PaddleOCR(**kwargs)

    _INSTANCES[key] = eng
    return eng


def _poly_bbox(poly) -> Tuple[float, float, float, float]:
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _res_dict(res) -> dict:
    """Chuẩn hóa kết quả predict() của 3.x về dict thuần.

    Result của PaddleX thường là dict-subclass (truy cập trực tiếp được), nhưng
    một số version bọc data trong res.json['res']. Cover cả hai cho chắc.
    """
    if isinstance(res, dict) and ("rec_texts" in res or "dt_polys" in res):
        return res
    j = getattr(res, "json", None)
    if isinstance(j, dict):
        inner = j.get("res", j)
        if isinstance(inner, dict):
            return inner
    return res if isinstance(res, dict) else {}


def run_ocr(engine, image: np.ndarray) -> List[Tuple[str, float, tuple]]:
    """Chạy detection+recognition, trả về [(text, conf, (x0,y0,x1,y1)), ...]."""
    out: List[Tuple[str, float, tuple]] = []
    if paddle_version_major() >= 3:
        for raw in engine.predict(image):
            res = _res_dict(raw)
            texts = res.get("rec_texts") or []
            scores = res.get("rec_scores") or []
            polys = res.get("rec_polys") or res.get("dt_polys") or []
            for i, t in enumerate(texts):
                conf = float(scores[i]) if i < len(scores) else 1.0
                bbox = _poly_bbox(polys[i]) if i < len(polys) else (0, 0, 0, 0)
                out.append((t, conf, bbox))
    else:
        result = engine.ocr(image, cls=True)
        if not result:
            return out
        page = result[0] if len(result) == 1 and isinstance(result[0], list) else result
        for item in page or []:
            try:
                box, (t, conf) = item
            except (ValueError, TypeError):
                continue
            out.append((t, float(conf), _poly_bbox(box)))
    return out


def detect_boxes(engine, image: np.ndarray) -> List[tuple]:
    """Chỉ lấy box (cho VietOCR mượn detector). Trả về list (x0,y0,x1,y1)."""
    boxes: List[tuple] = []
    if paddle_version_major() >= 3:
        for raw in engine.predict(image):
            res = _res_dict(raw)
            for poly in res.get("dt_polys") or res.get("rec_polys") or []:
                boxes.append(_poly_bbox(poly))
    else:
        result = engine.ocr(image, det=True, rec=False, cls=False)
        if not result:
            return boxes
        page = result[0] if len(result) == 1 and isinstance(result[0], list) else result
        for box in page or []:
            pts = box if isinstance(box[0], (list, tuple)) else box[0]
            boxes.append(_poly_bbox(pts))
    return boxes
