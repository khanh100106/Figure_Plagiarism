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

Kết thúc training sẽ có trong `outputs/exp_XXXX/` (mỗi lần chạy
`train.py` tự tạo 1 thư mục `exp_0001`, `exp_0002`, ... mới, không
ghi đè lên kết quả lần chạy trước):
- `config_used.json` — snapshot toàn bộ tham số đã dùng cho lần
  chạy này (để sau còn biết `exp_0003` khác `exp_0001` ở chỗ nào).
- `best.pt`, `last.pt` — checkpoint (model + optimizer + scaler + epoch + metrics).
- `history.json` — toàn bộ loss/recall theo từng epoch.
- `training_curves.png` — biểu đồ Train/Val Loss và Recall@1/@5 theo epoch.
- `test_metrics.json` — kết quả đánh giá trên tập test (dùng `best.pt`, chỉ chạy **1 lần duy nhất** sau khi train xong, không dùng tập test cho bất kỳ quyết định nào trong lúc tuning).

Dữ liệu được chia 3 phần theo `paper_id` (`TRAIN_RATIO`/`VAL_RATIO`/`TEST_RATIO` trong `config.py`, mặc định 70/15/15): tập val dùng để chọn "best model" và early stopping trong suốt quá trình train — vì bị dùng lặp đi lặp lại nên không còn khách quan để đánh giá cuối; tập test tách riêng, không đụng tới cho tới bước cuối cùng.

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

- `IMAGE_SIZE` (mặc định **224**) phải là bội số của 14 (patch size
  của DINOv2). 518 là kích thước "chuẩn" của DINOv2 nhưng rất nặng
  (attention tăng bình phương theo số patch) — chỉ nên dùng nếu
  GPU ≥16GB và đã `FREEZE_BACKBONE=True`.
- `FREEZE_BACKBONE = True` (mặc định): chỉ train projection head,
  đóng băng DINOv2 — giảm VRAM rất nhiều. Sau khi train ổn định,
  nếu đủ VRAM/dữ liệu, có thể thử `False` để fine-tune toàn bộ
  cho chất lượng tốt hơn (nhưng cần GPU mạnh hơn nhiều).
- `USE_AMP = True` (mixed precision, fp16): giảm ~40-50% VRAM,
  gần như không đổi chất lượng. Chỉ có tác dụng trên GPU.
- `GRAD_ACCUM_STEPS`: cộng dồn gradient qua nhiều batch nhỏ trước
  khi update trọng số, effective batch = `BATCH_SIZE * GRAD_ACCUM_STEPS`.
  Giúp gradient ổn định hơn khi buộc phải giảm `BATCH_SIZE` vì
  thiếu VRAM — **nhưng lưu ý:** với contrastive loss, negative vẫn
  chỉ lấy trong từng batch vật lý (`BATCH_SIZE`), không phải
  effective batch, nên đây không thay thế hoàn toàn cho việc tăng
  `BATCH_SIZE` thật nếu bạn có đủ VRAM.

### Nếu vẫn gặp lỗi `CUDA out of memory`

Giảm dần theo thứ tự ưu tiên (ít ảnh hưởng chất lượng nhất trước):
1. Giảm `IMAGE_SIZE` (224 → 168 hoặc 112, vẫn phải chia hết 14).
2. Giảm `BATCH_SIZE` (16 → 8 → 4), tăng `GRAD_ACCUM_STEPS` tương ứng
   để giữ effective batch không đổi.
3. Đổi `BACKBONE_NAME` sang `dinov2_vits14` (nhỏ hơn nhiều so với
   `vitb14`).
4. Đặt `FREEZE_TEXT_ENCODER = True` nếu đang dùng chế độ `"text"`.

## Nếu bị overfitting (train_loss về gần 0, val_loss tăng dần)

Dấu hiệu: `train_loss` giảm liên tục xuống rất thấp (< 0.1) trong
khi `val_loss` tăng lên và `recall@1`/`recall@5` không cải thiện
hoặc rất thấp. Thử theo thứ tự:

1. Đảm bảo `FREEZE_BACKBONE = True` và `FREEZE_TEXT_ENCODER = True`
   (mặc định hiện tại) — chỉ train 2 projection head trước, đây là
   baseline ổn định nhất.
2. Tăng `BATCH_SIZE` nếu còn dư VRAM (nhiều negative hơn giúp
   contrastive loss tổng quát hoá tốt hơn, không chỉ nhanh hơn).
3. Tăng `DROPOUT` (0.3 → 0.5) và/hoặc `WEIGHT_DECAY` (1e-2 → 5e-2).
4. Kiểm tra `caption_text` có bị trùng lặp nhiều giữa các dòng
   không (ví dụ nhiều figure dùng chung 1 caption mẫu) — nếu có,
   model dễ "học vẹt" theo caption thay vì học liên hệ thật.
5. Chỉ khi baseline (bước 1) đã ổn và còn dư VRAM/dữ liệu, mới thử
   `FREEZE_TEXT_ENCODER = False` (hoặc `FREEZE_BACKBONE = False`)
   để fine-tune sâu hơn — lúc này `BACKBONE_LR_MULTIPLIER` sẽ tự
   động cho phần encoder LR nhỏ hơn projection head, tránh phá vỡ
   feature đã pretrain.

