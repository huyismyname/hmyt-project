# Tóm tắt bài làm

- Đề tài: 2.7 — Đánh giá chất lượng ảnh đáy mắt (IQA) trên bộ EyeQ
- Nhóm: W2_N03 (MSSV: 20233761, 20233768, 20233774)
- Mô hình chính: FGR-Net (VGG16 Encoder + U-Net Decoder) — baseline tái lập, chống rò rỉ patient-level
- Kết quả nổi bật: Macro-F1 = 0,812 (KTC 95%: 0,805–0,819); Độ nhạy Reject = 0,673

## Cấu trúc thư mục
- `report/` — báo cáo (kiem_tra_lan_2.md)
- `src/` — notebook Colab E1–E4
- `slides/` — slide (nếu có)

## Cách chạy lại (reproducibility)
- Dữ liệu: EyeQ (GitHub HzFu/EyeQ) + ảnh EyePACS; nhãn `eyeq_labels.csv`; external DRIMDB (Kaggle).
- Môi trường: Google Colab (GPU A100), PyTorch/torchvision.
- Notebook:
  - E1 (baseline + chống rò rỉ): https://colab.research.google.com/drive/1IwSlEXpcRXIr5bGAdAwS5IzPwjHwxPeo
  - E2 (calibration): https://colab.research.google.com/drive/1zt5eeZTlz3ZaJDJQCyNCA8KlCW3rrjXr
  - E3 (reject option): https://colab.research.google.com/drive/1TwDI4f4_3NNO06fRP0haqXqBB2PA_qUp
  - E4 (external DRIMDB): https://colab.research.google.com/drive/1LzKvgpRAiwbqM1OWgby0OEgdjI4uVmir
