# Kiểm tra lần 2 — Báo cáo nhóm

> **Nhóm:** W2_N03 · **Đề tài:** 2.7 — Đánh giá chất lượng ảnh đáy mắt (IQA) trên bộ EyeQ
> **GVHD:** TS. Hoàng Quang Huy (Trường Điện-Điện tử, HUST) · **Môn:** ET4248
> **Fork (hmyt-project):** https://github.com/huyismyname/hmyt-project · **Nhánh:** `gd5-proposed-model`
> **Thành viên (MSSV + vai trò, không dùng tên thật):** `20233761 — Project Lead / ML`, `20233768 — Data & Preprocess`, `20233774 — Eval & XAI / Docs & MLOps`
> Quy ước bằng chứng: mỗi số liệu dẫn tới notebook/kết quả trong repo hoặc link Colab/Drive.

---

## A. Tổng quan y văn & khoảng trống nghiên cứu

### A1. Chiến lược tìm kiếm
- **Nguồn:** Consensus/Semantic Scholar, IEEE Xplore, Springer (MICCAI), ScienceDirect, PubMed; cộng mã nguồn GitHub của các mô hình.
- **Truy vấn chính:** "fundus image quality assessment", "retinal image quality deep learning", "gradeability classification fundus", "EyeQ quality".
- **Khoảng năm:** ưu tiên 2020–2026 (ngoại lệ: MCF-Net 2019 là nguồn gốc bộ EyeQ và là baseline kinh điển → giữ làm tham chiếu/origin).
- **Số bài giữ lại:** 10 (xem A2). **Tiêu chí chọn:** đúng miền IQA ảnh đáy mắt, có phương pháp học sâu, có số liệu định lượng; **loại:** trùng lặp, không có đánh giá định lượng, ngoài miền fundus.

### A2. Ma trận so sánh y văn (≥10 bài)
> Nguồn chi tiết + DOI đã kiểm chứng Crossref: xem file `MaTran_YVan_IQA` của nhóm. Bảng dưới là bản cô đọng; **dán DOI đã kiểm chứng vào cột cuối trước khi nộp**.

| # | Bài (năm, venue) | Dataset | Phương pháp | Giao thức chia DL | Kết quả chính (nguồn) | Thẩm định | Hạn chế / khoảng trống | DOI |
|---|---|---|---|---|---|---|---|---|
| 1 | MCF-Net (Fu, 2019, MICCAI) | EyeQ (28.792) | 3× DenseNet-121, hợp nhất đa không gian màu RGB/HSV/Lab | Không nêu rõ patient-level | Acc ~0,876 · F1 ~0,855 (Table paper) | Nội bộ EyeQ | Nặng; không calibration/uncertainty; nguồn gốc EyeQ | `10.1007/978-3-030-32239-7_6` |
| 2 | FGR-Net (Khalid, 2024, ESWA) | EyeQ + EyePACS + nội bộ | Autoencoder (VGG16) + classifier, reconstruction learning, XAI | Không nêu rõ patient-level | Acc 0,8947 · F1 0,8782 (Table 6) | Nội bộ + nội bộ bệnh viện | Yếu ở lớp Usable; không calibration/uncertainty; không external công khai | `10.1016/j.eswa.2023.121644` |
| 3 | VISTA (Khalid, 2024, NCA) | EyeQ | Split-and-reconstruct, ảnh 1280×1280, có giải thích | Không nêu rõ | Acc 0,9066 · F1 0,8868 (abstract/Table 9) | Nội bộ | Rất nặng tài nguyên; không calibration/external | `10.1007/s00521-024-10174-6` |
| 4 | Refined IQA (Guo, 2024, DIGITAL HEALTH) | Nội bộ Thượng Hải + EyeQ (external) | Multi-task Swin-Unet; 3 tiêu chí (Vị trí/Độ nét/Artifact) | Nội bộ | Acc nội bộ ~0,88; EyeQ external ~0,866 | Có external (EyeQ) | Nhãn khác lược đồ 3 lớp; không tái lập trực tiếp | `10.1177/20552076231207582` |
| 5 | FundusQ-Net (2023) | EyePACS-based, thang 1–10 | Hồi quy chất lượng (regression) | — | MAE/thang điểm (paper) | Nội bộ | Bài toán hồi quy, khác lược đồ 3 lớp EyeQ | `10.1016/j.cmpb.2023.107522` |
| 6 | Lightweight mobile IQA (Pérez, 2020) | Riêng | CNN nhẹ cho thiết bị di động | — | (paper) | Nội bộ | Khác dataset; không EyeQ | `10.1117/12.2547126` |
| 7 | ARIA quality (Shi, 2022, Sci Reports) | Kaggle + external | ResNet50; phân biệt kém chất lượng do bệnh vs artefact | Có external nhỏ | (Table paper) | Có external | Khung nhãn khác; không EyeQ 3 lớp | `10.1038/s41598-022-13919-2` |
| 8 | UWF IQA (2025) | Ultra-widefield | Học sâu trên ảnh UWF | — | (paper) | Nội bộ | Khác modality (UWF), không chuyển trực tiếp | `10.1136/bmjopen-2025-100058` |
| 9 | Infant fundus QFS (2025, Sci Reports) | RetCam trẻ sơ sinh | Inception-v3, 3 mức + lỗi thao tác | Nội bộ | (paper) | Nội bộ | Khác quần thể (trẻ sơ sinh), RetCam | `10.1038/s41598-025-10859-5` |
| 10 | ROP IQA CNN (Coyner, 2019) | i-ROP (trẻ sinh non) | CNN, nhị phân chất lượng | Nội bộ | (paper) | Nội bộ | Ngoài cửa sổ 2020–2026; khác quần thể → chỉ tham chiếu bối cảnh | `10.1016/j.oret.2019.01.015` |

### A3. Khoảng trống nghiên cứu (có bằng chứng)
- **G1 — Rò rỉ dữ liệu / kết quả thổi phồng.** Đối chứng cột *giao thức chia* (yếu/không nêu) với *kết quả* (cao) ở các bài #1–#3 cho thấy nghi ngờ chia không theo bệnh nhân. **Cách nhóm giải quyết:** chia **theo bệnh nhân (patient-level)**, có assert chống rò rỉ (xem B1).
- **G2 — Thiếu độ tin cậy có hiệu chỉnh & ước lượng bất định.** Không bài nào (#1–#3) báo cáo ECE/calibration hay cơ chế "từ chối". **Cách giải quyết:** temperature scaling + reject option (B2).
- **G3 — Thiếu kiểm chứng ngoại vi công khai.** #1–#3 chỉ đánh giá cùng phân phối. **Cách giải quyết:** kiểm chứng ngoại vi trên DRIMDB (B2/B3).
- **G4 — Bỏ qua lỗi an toàn quan trọng nhất (Reject lọt cổng).** Các bài tối ưu accuracy/F1 tổng, không đo riêng tỉ lệ ảnh *không đọc được (Reject)* bị xếp thành *đọc được*. **Cách giải quyết:** lấy **độ nhạy phát hiện Reject / số ảnh Reject lọt cổng** làm độ đo an toàn chính (B1).

### A4. Thẩm định PROBAST+AI cho bài quan trọng nhất (FGR-Net, #2)
- **Participants (Rủi ro: CAO/không rõ):** EyeQ là tập con một nguồn (EyePACS), thiếu metadata nhân khẩu; bài không khẳng định chia theo bệnh nhân → nguy cơ chọn mẫu/rò rỉ.
- **Predictors (Thấp–vừa):** đầu vào là ảnh đáy mắt, tiền xử lý nêu rõ; phụ thuộc điều kiện chụp.
- **Outcome (Vừa):** nhãn chất lượng do đồng thuận bác sĩ, **mang tính chủ quan** (nhất là lớp Usable); không nêu việc gán nhãn mù với đặc trưng mô hình.
- **Analysis (CAO về optimism):** cỡ mẫu lớn (tốt) nhưng **không chống rò rỉ rõ ràng, không calibration, không external** → kết quả có thể lạc quan quá mức.
- **Kết luận:** rủi ro sai lệch tổng thể **cao ở miền Analysis** — đây chính là điều dự án của nhóm khắc phục.

---

## B. Kết quả & tiến độ

### B1. Kết quả hiện tại
**Baseline (Milestone 2) — tái lập FGR-Net, chia theo bệnh nhân, chống rò rỉ.** Bằng chứng (Colab E1): https://colab.research.google.com/drive/1IwSlEXpcRXIr5bGAdAwS5IzPwjHwxPeo ; kết quả: Drive `milestone3_e1/e1_results.json`.

- Chia tập (patient-level, official EyeQ split): train/val/test = **11.281 / 1.262 / 16.249 ảnh** (7.752 / 862 / 12.048 bệnh nhân); `[ZERO-LEAKAGE CHECK] … PASS` (3 tập không giao patient_id).
- **Macro-F1 = 0,812** (KTC 95% bootstrap: 0,805–0,819); Accuracy tổng = 0,846.
- Độ nhạy theo lớp: Good 0,95 · Usable 0,77 · **Reject 0,673**.
- **Lỗi an toàn (báo cáo riêng với phân biệt, theo Phụ lục §2):**
  - Theo Phụ lục §7: **860 ảnh không-Good bị "lọt cổng" thành Good** (781 Usable + 79 Reject).
  - **Phân tích sâu hơn (an toàn nghiêm nhất):** theo nguyên tắc chi phí bất đối xứng, ca nguy hiểm nhất là **ảnh Reject (không đọc được) bị chấp nhận là đọc được** = **1.052 ảnh (33% lớp Reject)** bị xếp thành Good/Usable (79 → Good, 973 → Usable). Con số 860 chưa đếm 973 ca Reject→Usable. ⇒ **Độ nhạy phát hiện Reject chỉ 0,673** là độ đo an toàn chính của Milestone 3.

Ma trận nhầm lẫn (hàng = thật, cột = dự đoán), nguồn `[e1_results.json]`:

| Thật \ Dự đoán | Good | Usable | Reject |
|---|---|---|---|
| Good | 8062 | 408 | 1 |
| Usable | 781 | 3509 | 268 |
| Reject | 79 | 973 | 2168 |

**Mô hình đề xuất (Milestone 3) — đang hoàn thiện.** Hướng: cổng chất lượng *nhận biết bất định, có hiệu chỉnh, chi phí bất đối xứng*, ưu tiên giảm rò rỉ Reject.

### B2. Nghiên cứu cắt bỏ (ablation)
Bằng chứng (Colab): E2 https://colab.research.google.com/drive/1zt5eeZTlz3ZaJDJQCyNCA8KlCW3rrjXr · E3 https://colab.research.google.com/drive/1TwDI4f4_3NNO06fRP0haqXqBB2PA_qUp · E4 https://colab.research.google.com/drive/1LzKvgpRAiwbqM1OWgby0OEgdjI4uVmir · kết quả: Drive `milestone3_e2..e4/*.json`. (Thứ tự E1–E4 giả định theo link bạn gửi — đổi nếu khác.)

| ID | Cấu hình | Tập test | Độ đo chính | Kết quả |
|---|---|---|---|---|
| E1 | FGR-Net, patient-level (baseline trung thực) | EyeQ | Macro-F1 / Reject-leak | 0,812 [0,805–0,819] / 1.052 (33%) |
| E2 | + Hiệu chỉnh (temperature scaling) | EyeQ | ECE (nhãn không đổi) | ECE 0,0297 → **0,0114** (−61,7%) |
| E3 | + Cơ chế "từ chối" (selective prediction) | EyeQ | Độ chính xác trên tập được quyết định; lỗi biên | @cov 0,862: acc 0,890; @cov 0,737: acc 0,919 (giảm lỗi biên) |
| E4 | Đánh giá ngoại vi (đóng băng E1+E2) | DRIMDB | AUC / Sens(bad) / ECE shift | AUC 0,663 [0,623–0,703]; Sens(bad) 0,796; **ECE 0,266** |

> Ghi chú E3: notebook hiện đo theo lỗi Usable→Good; **đang cập nhật** để lấy *giảm rò rỉ Reject* (ảnh Reject bị xếp thành Good/Usable) làm độ đo đầu ra chính, khớp lỗi an toàn ở B1.

### B3. So sánh với y văn (và các so sánh KHÔNG công bằng)
- MCF-Net ~0,855 F1 · FGR-Net 0,878 · VISTA 0,887 — **nhưng** đều trên phép chia *không khẳng định patient-level* và *không external*. Baseline của nhóm (0,812) **thấp hơn một cách trung thực** do chia chống rò rỉ → **không so trực tiếp** với các số đó.
- **So sánh không công bằng cần nêu:** (i) khác giao thức chia; (ii) nhóm tái lập FGR-Net (VGG16 encoder + U-Net decoder, có thêm BatchNorm) chứ không chạy đúng mã gốc; (iii) chỉ số external (DRIMDB) không so được với các bài chỉ báo nội bộ.

### B4. Lộ trình tới hoàn thành

| Hạng mục | Chủ (vai trò) | Milestone | Hạn | Trạng thái (bằng chứng) |
|---|---|---|---|---|
| EDA, tổng quan y văn, khoảng trống | `[ID…]` | M1 | Tuần 1–3 | ✓ Đạt (`MaTran_YVan_IQA`, EDA notebook) |
| Chuẩn bị dữ liệu EyeQ 512px | `[ID…]` | M1 | Tuần 3 | ✓ Đạt (`eyeq_512.zip`, `eyeq_labels.csv`) |
| Baseline FGR-Net chống rò rỉ | `[ID…]` | M2 | Tuần 5–6 | ✓ Đạt (`E1 notebook`, `e1_results.json`) |
| Calibration + reject option | `[ID…]` | M3 | Tuần 7–10 | → Đang làm (E2/E3 đã chạy; đang đổi độ đo sang Reject-leak) |
| External DRIMDB | `[ID…]` | M3 | Tuần 10 | ✓ Đã chạy lần 1 (E4) — cần kiểm tra thành phần DRIMDB |
| Mô hình chi phí bất đối xứng / 2 giai đoạn | `[ID…]` | M3 | Tuần 11–12 | ☐ Kế hoạch |
| XAI (Grad-CAM) + phân tích lỗi Reject | `[ID…]` | M4 | Tuần 11–12 | ☐ Kế hoạch |
| Paper IEEE + dọn repo tái lập | `[ID…]` | Chung kết | Tuần 13–15 | ☐ Kế hoạch |

### B5. Rủi ro, vướng mắc & câu hỏi cho giảng viên
- **Rủi ro:** (i) kết quả nằm ở Colab/Drive, cần commit notebook/log vào repo để truy vết; (ii) gói DRIMDB trên Kaggle có **651 ảnh (376 good/275 bad)**, lớn hơn DRIMDB gốc (~216) → nghi có ảnh tăng cường ⇒ KTC ngoại vi chỉ tham khảo; (iii) nhãn Usable chủ quan.
- **Câu hỏi cho giảng viên:** (1) Có bắt buộc giữ official split EyeQ để so sánh, hay được tự chia patient-level 70/15/15 để tăng dữ liệu huấn luyện? (2) Có chấp nhận link Colab/Drive làm bằng chứng số liệu, hay phải commit toàn bộ notebook vào repo?

---

## C. Tự đánh giá theo FAIR, TRIPOD+AI, PROBAST+AI

### C1. FAIR dữ liệu
- **Findable/Accessible:** EyeQ công khai (GitHub HzFu/EyeQ); ảnh gốc EyePACS (Kaggle DR). **Interoperable:** JPEG + CSV nhãn chuẩn. **Reusable:** `[license EyePACS/EyeQ — ĐIỀN]`.
- **Datasheet rút gọn:** nguồn EyePACS, 28.792 ảnh gán nhãn chất lượng, 3 lớp; **không có định danh bệnh nhân** (đã khử nhận dạng); thiếu metadata nhân khẩu (hạn chế). **Không dùng dữ liệu bệnh nhân thật ngoài bộ công khai.**

### C2. Hướng dẫn báo cáo áp dụng: **TRIPOD+AI**
- **Câu hỏi nghiên cứu:** phân loại chất lượng ảnh đáy mắt (Good/Usable/Reject) làm cổng kiểm soát trước chẩn đoán.
- **T0 (thời điểm gốc):** tại thời điểm chụp ảnh, trước bước chẩn đoán tự động.
- **Định nghĩa outcome:** 3 lớp chất lượng theo lược đồ EyeQ (4 chỉ số: mờ, chiếu sáng không đều, tương phản thấp, artefact).
- **Loại mô hình/độ đo:** phân loại đa lớp; độ đo đa chiều (sens/spec theo lớp, Macro-F1, ECE, AUC phát hiện Reject), KTC bootstrap.

### C3. Tự thẩm định PROBAST+AI (4 miền)
- **Participants:** rủi ro vừa — một nguồn (EyePACS), thiếu nhân khẩu; giảm thiểu bằng external DRIMDB.
- **Predictors:** rủi ro thấp — ảnh RGB, tiền xử lý cố định, augment an toàn (không blur/noise).
- **Outcome:** rủi ro vừa — nhãn chủ quan (Usable); nêu rõ, phân tích riêng ranh giới.
- **Analysis:** **rủi ro thấp hơn baseline** — chia patient-level chống rò rỉ, có calibration, có KTC bootstrap, có external; rủi ro còn lại: cỡ mẫu Reject, nhãn nhiễu.

### C4. Checklist nhanh (Phụ lục §7) — *đối chiếu tên mục với sách môn học*

| # | Mục | Trạng thái | Bằng chứng |
|---|---|---|---|
| 1 | Chia patient-level chống rò rỉ | Đạt | assert PASS trong `[E1]` |
| 2 | Seed cố định | Đạt | `SEED=42` trong notebook |
| 3 | requirements ghim phiên bản | `[Đạt/chưa]` | `requirements.txt` |
| 4 | Không PII/dữ liệu bệnh nhân thật | Đạt | chỉ dùng EyeQ/DRIMDB công khai |
| 5 | Code chạy end-to-end | `[Đạt/chưa]` | `[link notebook]` |
| 6 | Độ đo kèm KTC | Đạt | bootstrap 95% trong `[E1/E4]` |
| 7 | Kiểm chứng ngoại vi | Đạt (sơ bộ) | `[E4 DRIMDB]` |
| 8 | Hiệu chỉnh/calibration | Đạt | ECE `[E2]` |
| 9 | Khai báo dùng AI | Đạt | Mục D |

---

## D. Khai báo sử dụng AI
- **Công cụ:** Claude (trợ lý AI).
- **Mục đích:** hỗ trợ dựng khung notebook (tái lập FGR-Net, chia chống rò rỉ, calibration, reject option, external), soạn khung tổng quan y văn, và soạn khung báo cáo này.
- **Cách nhóm kiểm chứng:** mọi số liệu được chạy lại trên Colab và lưu ra file kết quả trong Drive; mọi DOI kiểm chứng qua Crossref; nội dung được nhóm rà soát và chịu trách nhiệm 100%. **Không đưa dữ liệu bệnh nhân thật vào công cụ AI.**
