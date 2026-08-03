# tcOCR — OCR tài chính chứng khoán, độ chính xác cao (3 lớp)

Hệ thống OCR hướng tới **99% field-level** cho tài liệu tài chính (nhiều số + bảng),
thiết kế **pluggable** để benchmark và thay linh kiện dễ dàng. Chạy được ngay trên
**Google Colab + Gradio**.

## Kiến trúc 3 lớp

```
PDF scan ─▶ tiền xử lý ─▶ [Lớp 1] OCR + layout ─▶ routing ─▶ [Lớp 2] VLM (khó) ─▶ [Lớp 3] sửa TV ─▶ validation ─▶ kết quả
```

| Lớp | Vai trò | Backend (switch được) |
|---|---|---|
| **1. OCR truyền thống** | Chạy mọi trang, rẻ; sinh text + bảng + confidence | `paddle` ⇄ `vietocr` |
| **2. Escalation (VLM)** | Chỉ xử lý vùng khó (confidence thấp / bảng / số lỗi) | `null` (tắt) · `local_vlm` (Qwen2.5-VL) · `private_api` (ANTT) |
| **3. Sửa tiếng Việt** | Chỉ sửa **text diễn giải**, số/bảng bypass | `protonx` ⇄ `bmd1905` ⇄ `null` |
| **+ Validation** | Luật nghiệp vụ (tổng khớp, mã CK, ngày) → cờ soát | luôn bật được |

**Vì sao 99% đến từ Lớp validation + human-in-the-loop**, không chỉ OCR: 1 chữ số sai = cả field sai, nên metric là **field-level** (số, ô bảng), và field fail được flag để soát/thử lại.

## Chạy trên Colab (khuyến nghị để test)

Mở `notebooks/tcOCR_colab.ipynb` → chọn GPU T4 → chạy lần lượt. Cell cuối cho **link Gradio công khai**.

## Chạy local

```bash
pip install -r requirements.txt
python app.py
```

## Benchmark field-level (la bàn tới 99%)

Cần **PDF scan** (đầu vào) + **PDF gốc digital** (nguồn gold, trích tự động bằng pdfplumber):

```python
from tcocr.config import PipelineConfig
from tcocr.pipeline import OCRPipeline
from tcocr.benchmark.runner import run_pair, run_dir

pipe = OCRPipeline(PipelineConfig(ocr_backend="paddle", escalation_backend="null"))
print(run_pair(pipe, "scan.pdf", "goc.pdf").to_markdown())
# hoặc cả thư mục (khớp file trùng tên): run_dir(pipe, "data/scan", "data/gold")
```

Chỉ số theo dõi: **`number_acc`** và **`table_cell_acc`** (field-level) — CER chỉ phụ trợ.

## Đổi backend (pluggable)

```python
PipelineConfig(
    ocr_backend="vietocr",          # so với "paddle"
    escalation_backend="local_vlm", # bật VLM; "null" để đo baseline
    escalation_kwargs={"model_id": "Qwen/Qwen2.5-VL-7B-Instruct"},
    correction_backend="bmd1905",   # nếu vướng license ProtonX
)
```

## ⚠️ Hai lưu ý quan trọng

1. **License ProtonX = v1.3-NC (Non-Commercial).** TCBS dùng production là commercial →
   phải **xin bản quyền thương mại từ ProtonX**, hoặc dùng `bmd1905` (license thoáng hơn).
   Lớp 3 để pluggable đúng vì lý do này.
2. **Data residency.** Link Gradio `share=True` và cloud GPU chỉ dùng cho **tài liệu công khai**.
   Data thật của khách → chạy **on-prem** (self-host VLM hoặc API nội bộ của ANTT qua
   `private_api` backend). Không đẩy data thật qua bên thứ ba.

## Khắc phục sự cố (Colab)

- **`PDX has already been initialized`**: PaddleOCR 3.x chỉ init 1 lần/process. Code đã
  dùng singleton để tránh, nhưng nếu vẫn gặp (do lần chạy trước lỗi dở): **Runtime →
  Restart session** rồi chạy lại từ đầu, đừng chạy OCR 2 lần trước khi restart.
- **PaddleOCR 3.x vs 2.x**: code tự nhận diện phiên bản (`tcocr/ocr/_paddle.py`), không
  cần pin. Chạy được với bản Colab cài mặc định.
- **PP-Structure (tách cấu trúc bảng) mặc định TẮT** trên Colab (né init lần 2 + kỵ
  Python 3.12). Bật bằng `PipelineConfig(enable_layout_structure=True)` khi on-prem đã
  dựng ổn định. Khi tắt, chữ trong bảng vẫn được OCR (chỉ mất lưới ô — để lớp 2 VLM lo).

## Cấu trúc mã

```
tcocr/
  types.py          kiểu dữ liệu chung (Region/Table/PageResult…)
  config.py         PipelineConfig — mọi lựa chọn backend
  pipeline.py       ghép 3 lớp + routing + validation
  preprocess.py     deskew / denoise
  layout.py         layout + table detection (PP-Structure)
  pdfio.py          PDF -> ảnh
  ocr/              Lớp 1: base + paddle + vietocr + registry
  escalation/       Lớp 2: base + null + local_vlm + private_api + registry
  correction/       Lớp 3: base(mask số) + seq2seq(protonx/bmd1905) + registry
  validation/       luật nghiệp vụ (tổng khớp, mã CK, ngày)
  gold/             trích gold từ PDF gốc digital
  benchmark/        metrics field-level + runner
app.py              Gradio UI
notebooks/          notebook Colab
```
