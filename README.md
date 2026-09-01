# Figure-Caption Plagiarism Pretraining (DINOv2)

Pipeline gọn, độc lập với thư mục `retrieval/` cũ. Học một không
gian embedding chung giữa figure và caption, dùng để phát hiện
đạo hình (figure trùng/gần giống, hoặc figure bị gán sai caption).

Hỗ trợ 2 chế độ, chọn bằng `config.CAPTION_MODALITY`:

- **`"text"` (mặc định, dual-encoder kiểu CLIP):** DINOv2 mã hoá
  ảnh figure, text encoder (mặc định SciBERT) mã hoá `caption_text`.
  Đây là hướng bạn đang làm (image-text).
- **`"image"` (Siamese, cách ban đầu):** dùng chung 1 DINOv2 để mã
  hoá cả ảnh figure và ảnh caption (`caption_image_path`).

Cả 2 chế độ dùng chung 1 vòng lặp train/validate, chung loss
(symmetric InfoNCE kiểu CLIP), chỉ khác cách encode caption.

## Cách hoạt động

- Mỗi dòng trong CSV = 1 cặp `(figure, caption)` **thật** (cùng
  xuất hiện trong 1 paper).
- Trong mỗi batch, `figure[i]` phải khớp `caption[i]`; mọi cặp
  khác trong batch đóng vai trò negative (in-batch negatives).
- Sau khi train, embedding này dùng để: tìm figure trùng/gần
  giống nhau (đạo hình) bằng cosine similarity, hoặc kiểm tra một
  figure có "lệch" khỏi caption gốc của nó hay không.

## Giả định về dữ liệu

- `figure_image_path` (và `caption_image_path` nếu dùng chế độ
  `"image"`) là ảnh đã được **crop sẵn** — đã xác nhận đúng.
- Cột `field`, `category`, `published`, `title`, `page` chưa dùng
  đến, chỉ là metadata đi kèm.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chuẩn bị dữ liệu

1. Đặt file CSV/TSV của bạn vào đâu đó, ví dụ `data/metadata.csv`.
2. Sửa `config.py`:
   - `DATA_CSV`: đường dẫn tới file CSV.
   - `IMAGE_ROOT`: thư mục gốc để nối với `figure_image_path` (và
     `caption_image_path` nếu dùng chế độ `"image"`) nếu chúng là
     đường dẫn tương đối. Nếu CSV đã lưu đường dẫn tuyệt đối thì
     để `IMAGE_ROOT = Path("")`.

## Chạy train

```bash
python train.py
```

Mỗi epoch sẽ in ra:

```
Epoch 003/30 | train_loss=1.8421 | val_loss=1.7203 | recall@1=0.412 | recall@5=0.731 | lr=9.51e-05 | time=42.3s
```

Kết thúc training sẽ có trong `outputs/`:
- `best.pt`, `last.pt` — checkpoint (model + optimizer + epoch + metrics).
- `history.json` — toàn bộ loss/recall theo từng epoch.
- `training_curves.png` — biểu đồ Train/Val Loss và Recall@1/@5 theo epoch.

## Đổi backbone / text encoder / tham số

Tất cả nằm trong `config.py`, không cần sửa code:
- `CAPTION_MODALITY`: `"text"` hoặc `"image"`.
- `BACKBONE_NAME`: `dinov2_vits14` (nhỏ, nhanh) → `dinov2_vitg14`
  (lớn nhất, cần nhiều VRAM).
- `TEXT_MODEL_NAME`: tên model trên HuggingFace Hub, ví dụ đổi
  sang `"allenai/specter2"` hoặc `"bert-base-uncased"`.
- `MAX_TEXT_LENGTH`: số token tối đa mỗi caption.
- `FREEZE_BACKBONE` / `FREEZE_TEXT_ENCODER = True`: chỉ train
  projection head, đóng băng encoder tương ứng — hợp khi dataset
  nhỏ hoặc muốn train nhanh để thử nghiệm.
- `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`, `TRAIN_RATIO`...

## Lưu ý phần cứng

- `IMAGE_SIZE` (mặc định 518) phải là bội số của 14 (patch size
  của DINOv2). Muốn giảm VRAM có thể hạ xuống 224 hoặc 252 (vẫn
  chia hết cho 14).
- Batch size lớn hơn thường giúp contrastive loss tốt hơn (nhiều
  negative hơn trong 1 batch) — nếu thiếu VRAM, ưu tiên giảm
  `IMAGE_SIZE` trước khi giảm `BATCH_SIZE`.
- Chế độ `"text"` tải thêm 1 model ngôn ngữ (SciBERT mặc định
  ~440MB) — cần thêm VRAM/RAM so với chế độ `"image"` thuần.

