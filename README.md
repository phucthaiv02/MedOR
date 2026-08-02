# MedOR
Ontological Reasoning in Medical Knowledge Retrieval

Pipeline fine-tune **Qwen3-8B** để trích xuất thực thể y khoa (triệu chứng, chẩn đoán, xét nghiệm, thuốc...) từ văn bản bệnh án tiếng Việt, dùng **Unsloth** để train, **vLLM** để inference, quản lý package bằng **uv**.

## 1. Cài đặt

Project dùng [`uv`](https://docs.astral.sh/uv/) để quản lý package và virtualenv, Python 3.12 (khai báo ở `.python-version`, `uv` sẽ tự tải nếu máy chưa có).

### 1.1. Cài `uv`

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# hoặc qua pip nếu đã có Python
pip install uv
```

Kiểm tra:

```bash
uv --version
```

### 1.2. Cài dependencies của project

```bash
git clone https://github.com/phucthaiv02/MedOR
cd MedOR

# Chỉ các phụ thuộc cơ bản (pandas, datasets, pydantic, dotenv...)
uv sync

# Máy dùng để TRAIN (cần GPU, cài thêm unsloth/trl/peft/wandb...)
uv sync --extra train

# Máy dùng để INFERENCE (cần GPU, cài thêm vllm)
uv sync --extra infer
```

`uv sync` tự tạo virtualenv tại `.venv/` và cài đúng version đã khoá trong `uv.lock`. Mọi lệnh chạy script trong README này đều qua `uv run ...` — không cần tự `activate` venv.

### 1.3. Cấu hình secrets (`.env`)

Tạo file `.env` ở gốc repo (đã có sẵn, đã gitignore) để chứa secrets:

```
WANDB_API_KEY = <wandb api key>
HF_TOKEN = <huggingface token>
```

`train.py` và `infer_vllm.py` tự động load `.env` khi chạy — không cần `export` tay. `WANDB_API_KEY` được `wandb` tự đọc để log, `HF_TOKEN` được `huggingface_hub` tự đọc khi tải model gated hoặc khi `push_to_hub`.

## 2. Chuẩn bị dữ liệu

```bash
# Tải dataset phucthaiv02/medor từ HF, bỏ trường "position", chia train/test 90/10
uv run python data/medor/download_medor.py
# -> data/medor/train.csv, data/medor/test.csv

# Tách test.csv thành từng cặp file .txt (input) / .json (gold entities) để phục vụ eval
uv run python -m src.medor.prepare_eval
# -> data/medor/eval/txt/{i}.txt, data/medor/eval/gold/{i}.json
```

## 3. Train

Cấu hình nằm ở [`configs/train.yaml`](configs/train.yaml), chỉnh trực tiếp file này thay vì sửa code.

```bash
uv run python -m src.medor.train --config configs/train.yaml
```

Các trường quan trọng trong `configs/train.yaml`:

| Trường | Ý nghĩa |
|---|---|
| `base_model` | Model gốc. Mặc định `unsloth/Qwen3-8B` (bf16). Chỉ đổi sang bản `-bnb-4bit` nếu GPU ít VRAM (<24GB); trên H100/H200 không cần 4-bit. |
| `load_in_4bit` | `false` khi có đủ VRAM để load bf16 (khuyến nghị trên H100/H200). |
| `train_csv` / `val_csv` / `val_split` | Nguồn dữ liệu train. Nếu `val_csv: null` thì tự tách `val_split` (mặc định 5%) từ `train_csv`. |
| `max_seq_length` | Ngưỡng context length (token). Sample có độ dài (prompt+response) ≥ ngưỡng này sẽ bị loại khỏi tập train, log số lượng bị drop ra console. |
| `lora.*` | Tham số LoRA (r, alpha, dropout, target_modules). |
| `output_dir` | Nơi lưu LoRA adapter. |
| `merged_dir` | Nơi lưu **model merge 16-bit đầy đủ** — luôn được lưu sau khi train xong, dùng thẳng làm `model=...` cho vLLM, không cần merge adapter thủ công. |
| `num_train_epochs`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `learning_rate`, ... | Hyperparameter train chuẩn của HF `Trainer`. |
| `report_to`, `wandb_project`, `wandb_run_name` | Bật log W&B (mặc định `wandb`). Cần `WANDB_API_KEY` trong `.env`. |
| `push_to_hub`, `hub_model_id`, `hub_private`, `hub_token` | Nếu `push_to_hub: true`, sau khi train xong sẽ push model đã merge lên HF Hub tại `hub_model_id`. **Phải set `hub_model_id`**, nếu không script sẽ báo lỗi ngay từ đầu. Cần `HF_TOKEN` trong `.env` (hoặc set `hub_token`). |

Kết thúc training sẽ có:
- `outputs/qwen3-medor-lora/` — LoRA adapter + tokenizer
- `outputs/qwen3-medor-merged/` — model merge 16-bit, sẵn sàng nạp vào vLLM

## 4. Inference (vLLM)

Cấu hình ở [`configs/infer.yaml`](configs/infer.yaml). Input là một **thư mục chứa các file `.txt`** (mỗi file 1 văn bản cần trích xuất), output là **thư mục chứa file `.json` cùng tên** (danh sách entity trích xuất được).

```bash
uv run python -m src.medor.infer_vllm --config configs/infer.yaml
```

- **Không dùng prompt/instruction**: model chỉ nhận đúng `input_text` (theo chat template lúc train, không có system prompt), không kèm hướng dẫn hay ví dụ few-shot.
- **Tắt thinking mode của Qwen3**: cả lúc train (`format_for_training`) và lúc infer đều gọi `apply_chat_template(..., enable_thinking=False)` — task này chỉ cần JSON trực tiếp, không cần khối `<think>...</think>` của Qwen3.
- **Guided decoding**: bật theo mặc định (`guided_decoding: true`), ép model sinh đúng JSON schema định nghĩa ở [`src/medor/schema.py`](src/medor/schema.py) (mảng object gồm `text`, `type`, `assertions`, `context`).
- **Tính lại `position`**: model không sinh trực tiếp offset ký tự, thay vào đó output gồm `context` (đoạn văn bản ngắn quanh entity). Sau khi generate, `find_position()` trong [`infer_vllm.py`](src/medor/infer_vllm.py) tính offset qua 2 bước:
  1. Tìm vị trí của `context` trong `input_text`.
  2. Tìm vị trí của `text` trong `context` đó.
  3. Cộng dồn để ra offset cuối cùng của `text` trong `input_text`.
  
  Cách này tránh việc `text` bị lặp lại nhiều lần trong văn bản dài dẫn đến bắt sai vị trí (khác với cách tìm `text` trực tiếp trong toàn bộ `input_text`). Entity nào không khớp được `context` hoặc `text` sẽ bị bỏ qua (log số lượng bị drop ra console).
- File `.json` output cuối cùng gồm `text`, `type`, `assertions`, `position` (không còn `context`, đã được dùng xong để tính position).
- `merged_model_path` (mặc định trỏ tới `outputs/qwen3-medor-merged`) được ưu tiên dùng nếu có sẵn; nếu để `null` thì dùng `base_model` + `lora_path` qua cơ chế LoRA của vLLM.
- `input_dir` / `output_dir`: đổi sang thư mục dữ liệu thực tế khi chạy inference production (ví dụ một thư mục hồ sơ bệnh án mới), không nhất thiết phải là `data/medor/eval/*`.
- Nếu model trả về JSON không hợp lệ, file `.json` tương ứng sẽ là mảng rỗng `[]` thay vì crash.

## 5. Đánh giá

```bash
uv run python -m src.medor.evaluate --config configs/eval.yaml
```

So khớp từng cặp file `predictions_dir/{i}.json` với `gold_dir/{i}.json` (theo tên file), tính precision/recall/F1 theo entity (`match_mode: text_type` hoặc `text_type_assertions`), và tỉ lệ output JSON không hợp lệ. Kết quả in ra console và lưu vào `metrics_output` (mặc định `outputs/eval/metrics.json`).

## 6. Quy trình đầy đủ (tóm tắt)

```bash
uv run python data/medor/download_medor.py
uv run python -m src.medor.prepare_eval

uv sync --extra train
uv run python -m src.medor.train --config configs/train.yaml

uv sync --extra infer
uv run python -m src.medor.infer_vllm --config configs/infer.yaml
uv run python -m src.medor.evaluate --config configs/eval.yaml
```
