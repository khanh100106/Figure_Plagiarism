# Stage 1 - Image Plagiarism Pretraining

Pretrain DINOv2 nhận diện đạo hình (image-image) trên 2 nguồn dữ
liệu gộp lại:

1. **Corpus đạo hình thật** (flowchart, có nhãn thật qua XML hoặc
   ghép tên file) — dạy model các kiểu đạo hình thật (exact copy,
   text plagiarism, text+structure).
2. **Cặp synthetic** sinh từ `paper2fig2026` (cả 7 category) —
   bù domain còn thiếu (figure khoa học màu, đa dạng loại) mà
   corpus thật không có.

Checkpoint sau khi train xong sẽ nạp thẳng vào `image_backbone`
của `ImageTextDualEncoder` ở Giai đoạn 2 (`plagiarism_dinov2/`).

## Cài đặt

```bash
pip install torch torchvision pandas pillow numpy tqdm
```

## Chuẩn bị dữ liệu

Cần 2 nguồn, sửa đường dẫn trong `config.py`:

```python
FLOWCHART_ROOT = Path("data/figure_plagiarism_corpus")   # corpus dao hinh that
PAPER2FIG_ROOT = Path("data/paper2fig2026")               # dataset cua ban
PAPER2FIG_METADATA = PAPER2FIG_ROOT / "metadata" / "paper2fig2026_metadata.csv"
```

`FLOWCHART_GROUPS` trong `config.py` mô tả cấu trúc thư mục
corpus thật — nếu tên thư mục thực tế của bạn khác (viết
hoa/thường, dấu cách), sửa lại 3 dict trong đó cho khớp.

## Chạy — đúng thứ tự 3 bước

### Bước 1: Sinh cặp + mining hard negative

```bash
python build_pairs.py
```

Làm gì: đọc corpus thật (qua `flowchart_corpus.py`), liệt kê
figure từ `paper2fig2026` (qua `synthetic_pairs.py`), trích
embedding toàn bộ pool bằng 1 DINOv2 nhỏ **đóng băng** (chỉ để đo
độ giống nhau, không liên quan gì model sẽ train ở bước 2), rồi
mining hard negative thật sự cho từng anchor.

Kết quả: file `stage1_outputs/pairs_with_hard_negatives.csv`.

Bước này có thể mất vài phút đến vài chục phút tuỳ số lượng ảnh
trong pool (chủ yếu là thời gian trích embedding).

### Bước 2: Train Stage 1

```bash
python train_stage1.py
```

Làm gì: train 1 DINOv2 (Siamese, dùng chung 1 backbone) bằng
triplet loss trên `pairs_with_hard_negatives.csv` vừa tạo ở
Bước 1. Tự tạo thư mục experiment riêng mỗi lần chạy
(`stage1_outputs/exp_0001`, `exp_0002`, ...), giống cơ chế ở
Giai đoạn 2.

Kết quả mỗi experiment:
- `best.pt`, `last.pt` — checkpoint.
- `history.json` — loss/accuracy theo epoch.
- `config_used.json` — snapshot tham số đã dùng.

### Bước 3: Dùng checkpoint ở Giai đoạn 2

Không cần chạy thêm file nào — mở
`plagiarism_dinov2/config.py`, sửa:

```python
STAGE1_CHECKPOINT = Path("../stage1_pretrain/stage1_outputs/exp_0001/best.pt")
```

rồi chạy `python train.py` ở Giai đoạn 2 như bình thường. `train.py`
ở đó sẽ tự nạp backbone đã học từ Stage 1 trước khi bắt đầu
fine-tune tiếp trên `paper2fig2026` (có caption).

*(Tuỳ chọn, không bắt buộc)*: nếu muốn tách riêng 1 file backbone
gọn nhẹ (bỏ optimizer, projection head...) để lưu trữ/chia sẻ:
```bash
python export_backbone.py --checkpoint stage1_outputs/exp_0001/best.pt --output stage1_backbone.pt
```

## Tóm tắt thứ tự chạy

```
1. build_pairs.py      ->  stage1_outputs/pairs_with_hard_negatives.csv
2. train_stage1.py     ->  stage1_outputs/exp_000N/best.pt
3. (Giai doan 2) sua STAGE1_CHECKPOINT trong plagiarism_dinov2/config.py
   -> chay plagiarism_dinov2/train.py nhu binh thuong
```

## Các tham số đáng chú ý trong `config.py`

- `SYNTHETIC_COPIES_PER_FIGURE`: số bản synthetic sinh ra trên
  mỗi figure paper2fig2026 (mặc định 2). Tăng lên nếu muốn nhiều
  dữ liệu synthetic hơn so với dữ liệu thật.
- `HARD_NEGATIVE_TOP_K`: số ứng viên hard negative giữ lại cho
  mỗi anchor trước khi chọn ngẫu nhiên 1 cái mỗi lần train (đa
  dạng hoá, tránh học thuộc đúng 1 negative cố định).
- `FREEZE_BACKBONE = False`: khác với Giai đoạn 2, ở đây mặc định
  **fine-tune toàn bộ** DINOv2 vì mục tiêu chính là để nó học đặc
  trưng đạo hình thật sự — không đóng băng.
- `TRIPLET_MARGIN`: khoảng cách tối thiểu giữa
  (anchor, negative) so với (anchor, positive). Tăng lên nếu muốn
  model phân biệt "quyết liệt" hơn.

## Nếu gặp lỗi

- `CUDA out of memory`: giảm `BATCH_SIZE`, giảm `IMAGE_SIZE`, hoặc
  đổi `BACKBONE_NAME` sang `dinov2_vits14` — áp dụng đúng như đã
  làm ở Giai đoạn 2.
- `build_pairs.py` báo `[Canh bao] Bo qua nhom '...'`: tên thư mục
  trong `FLOWCHART_GROUPS` không khớp cấu trúc thật — sửa lại tên
  cho đúng.
- `build_pairs.py` báo không parse được cặp nào: kiểm tra lại
  `FLOWCHART_ROOT` có trỏ đúng thư mục gốc "Figure Plagiarism
  corpus" không.
