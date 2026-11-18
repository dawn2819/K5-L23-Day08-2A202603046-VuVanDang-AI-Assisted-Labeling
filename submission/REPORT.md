# Báo cáo Day 08 — 2A202603046 - Vũ Văn Đăng

> Một file duy nhất cho cả bài (đã gộp recipe, QC và ghi chú cá nhân). Mọi con số kèm **mẫu số** và nguồn (file nào). Viết bằng lời của bạn.

## 1. Model và recipe (Part 2–3)

| Lớp | Model + cỡ | Ngưỡng | Vì sao (dẫn `run_id` trong `model_log.csv`) |
|---|---|---|---|
| person / bicycle / motorbike / car / bus / truck | YOLO26m | conf 0.30 (mặc định) | Nhận diện chính xác các phương tiện và người tham gia giao thông chuẩn; ở conf 0.25 (r01) có nhiều box nhiễu, conf 0.50 (r02) bị sót vật ở xa; ngưỡng 0.30 là điểm cân bằng tối ưu giữa precision và recall |
| three_wheeler | YOLOE-26s, prompt: "tuk-tuk, cyclo, cargo tricycle" | conf 0.30 | YOLO26 gọi nhầm tuk-tuk thành car/truck ở r01; YOLOE (r03) phát hiện chính xác 8/8 xe ba bánh ở tập pilot; dùng prompt open-vocabulary chuyên biệt |

- Lệnh `merge_prelabels.py` đã dùng (dán nguyên):
  `python scripts/merge_prelabels.py work/prelabel/yoloe_all/pre.coco.json work/prelabel/y26_all/pre.coco.json --conf 0.3 --iou 0.6 --split "batch,gold" --out work/prelabel/merged_review/pre.coco.json`
- **Phiên bản SAM bạn chọn** (SAM 1 / 2.1 / 3) và lý do: Chọn **SAM 2.1** (trọng số `sam2.1_b.pt`). Lý do: SAM 2.1 cải thiện vượt bậc khả năng phân tách ranh giới các đối tượng phức tạp và bị che khuất so với SAM 1; trọng số bản Base (~154 MB) chạy ổn định trên CPU/RAM phổ thông mà không yêu cầu GPU khủng; không yêu cầu token truy cập bị kiểm duyệt như SAM 3.
- Lỗi nào **không sửa được bằng đổi ngưỡng** (ví dụ tuk-tuk bị gọi `car`)? Sửa thế nào?
  Lỗi phân loại sai về bản chất tri thức (semantic misclassification), điển hình là việc YOLO26 gọi `three_wheeler` (tuk-tuk, xích lô, ba gác) thành `car` hoặc `truck`. Nguyên nhân là tập huấn luyện COCO không có khái niệm xe ba bánh châu Á, nên mô hình buộc phải map sang lớp gần giống nhất có xác suất cao. Đổi ngưỡng conf không thể sửa được vì điểm tin cậy của box `car` thường rất cao (> 0.80). Cách sửa triệt để: kết hợp mô hình open-vocabulary (YOLOE) với prompt văn bản đặc tả, xếp YOLOE ở vị trí ưu tiên số 1 khi chạy `merge_prelabels.py` với ngưỡng IoU chồng lấn 0.6 để ghi đè nhãn `car`/`truck` thành `three_wheeler`.
- Box **điểm cao mà sai** hoặc **điểm thấp mà đúng** (ảnh nào):
  - Điểm cao mà sai: Ở ảnh `img_002.jpg`, chiếc xe tuk-tuk ở tiền cảnh bị YOLO26 gán nhãn `car` với điểm tin cậy rất cao `conf = 0.86`, do khung xe ô tô con có nhiều nét tương đồng với thân xe tuk-tuk.
  - Điểm thấp mà đúng: Ở ảnh `img_011.jpg`, chiếc xe ba bánh ở cự ly xa chỉ đạt điểm tin cậy `conf = 0.28` trong YOLOE nhưng hoàn toàn là một phương tiện thật cần được gán nhãn.
- Giới hạn: chọn ngưỡng trên chính gold (4 ảnh) nên số đo máy trên gold là lạc quan:
  Vì chỉ có 4 ảnh gold (47 đối tượng) và tập quét ngưỡng trùng với tập đánh giá (in-sample evaluation), số đo precision và recall thu được trong `threshold_sweep` mang tính chất lạc quan hơn so với thực tế khi triển khai trên dữ liệu chưa từng thấy. Hơn nữa, các lớp có ít mẫu (như `bus`, `truck`, `bicycle` có < 5 đối tượng) không đủ ý nghĩa thống kê để chọn ngưỡng riêng mà phải dựa vào ngưỡng mặc định 0.30.

## 2. AI error profile (từ `qc/review_stats.md`)

| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted | 440 | 87.0% (440/506) | 506 box máy |
| edited | 51 | 10.1% (51/506) | 506 box máy |
| deleted | 15 | 3.0% (15/506) | 506 box máy |
| added | 16 | 3.2% (16/507) | 507 đối tượng sau duyệt |

- Lớp nào máy tệ nhất? Vì sao (dẫn ảnh)?
  Lớp `person` và `three_wheeler` là hai lớp máy thể hiện kém nhất:
  - `person`: Máy thường xuyên bỏ sót người đi bộ ở khoảng cách xa hoặc người lái xe bị che khuất một phần (như trong `img_007.jpg`, `img_015.jpg`). Người đi bộ chiếm phần lớn số box bị thêm mới (added = 11) và số box cần chỉnh sửa viền ôm sát (edited = 31).
  - `three_wheeler`: YOLO26 hoàn toàn không có khả năng nhận diện, phụ thuộc 100% vào YOLOE. Box tạo ra bởi YOLOE đôi khi bị rộng viền (loose) hoặc sót các xe ba gác tự chế chở hàng cồng kềnh (như `img_003.jpg`).
- So với gold: precision / recall của **máy**:
  - Precision: 16.4% (mẫu số: 12 TP / (12 TP + 61 FP), KTC 95%: 10%–27%)
  - Recall: 25.5% (mẫu số: 12 TP / (12 TP + 35 FN), tổng 47 đối tượng gold, KTC 95%: 15%–40%)
  Con số recall thấp trên gold phản ánh đúng thực tế đường phố châu Á: nhiều phương tiện nhỏ, mật độ dày đặc và người ngồi trên xe máy thường bị máy gộp chung hoặc bỏ qua nếu đặt ngưỡng phát hiện chuẩn.

## 3. Thiên kiến tự động hóa và ảnh bẫy

- Bắt bẫy: 10/12 = 83.3% (KTC 95%: 55%–95%); loại nào lọt: 1 bẫy `loose` ở `img_027.jpg` và 1 bẫy `missing` ở `img_021.jpg`; vì sao (bạn đã bỏ bước nào của checklist)?
  - Ở `img_021.jpg`: Cảnh ngã tư giao thông cực kỳ dày đặc với hàng chục xe máy chen chúc. Tôi đã lơ là bước 0 (Quét tổng thể bằng mắt thường trước khi mở nhãn máy), dẫn tới việc mắt bị "cuốn" theo các box màu xanh hiện sẵn của mô hình và bỏ sót một người đi bộ ở rìa đường mà máy không tạo box.
  - Ở `img_027.jpg`: Bẫy là một box xe máy bị kéo giãn lỏng lẻo (loose). Tôi đã vi phạm bước 4 (Kiểm tra độ ôm sát viền box), vì nhìn lướt qua thấy box đã bao trọn chiếc xe nên vội vã bấm Accept mà không zoom vào để thấy khoảng hở > 8 px ở cạnh trên và cạnh phải.
- Accepted cao mà bắt bẫy thấp (hoặc ngược lại) nói gì về cách bạn duyệt?
  Accepted cao (ví dụ > 95%) đi kèm tỷ lệ bắt bẫy thấp (< 70%) là triệu chứng rõ ràng của hiện tượng **thiên kiến tự động hóa (automation bias)** và hội chứng "bấm gật đầu" (rubber-stamping). Người duyệt bị thôi miên bởi cảm giác máy đã làm phần lớn công việc một cách chính xác, dẫn đến việc hạ thấp mức độ cảnh giác và chấp nhận luôn cả các lỗi sai ngớ ngẩn của mô hình. Ngược lại, bắt bẫy cao (> 80%) chứng minh người duyệt thực sự thực hiện vai trò chốt chặn kiểm định chất lượng độc lập.

## 4. Máy có đáng dùng không?

- Thời gian (từ `time_log.csv`): gán tay 5.5 phút/ảnh (22 phút / 4 ảnh gold) vs duyệt có pre-label 3.0 phút/ảnh (72 phút / 24 ảnh batch). Tiết kiệm ~45% thời gian gán nhãn trên mỗi khung hình. Thiên lệch: Người làm đã quen thuộc với bộ dữ liệu và quy tắc sau khi tự tay gán 4 ảnh gold nên tốc độ nhận diện ngữ cảnh ở tập batch nhanh hơn.
- Kết luận: **RẤT ĐÁNG DÙNG**, nhưng chỉ khi có mô hình phù hợp và quy trình kiểm soát nghiêm ngặt. Mô hình pre-label giải phóng con người khỏi thao tác vẽ hộp thô lặp đi lặp lại cho hàng trăm phương tiện tiêu chuẩn (`car`, `motorbike`, `person`). Tuy nhiên, với các lớp đặc thù địa phương (`three_wheeler`), bắt buộc phải có sự can thiệp của open-vocabulary detector và con người phải luôn là người quyết định cuối cùng.
- **Ba quy tắc giảm thiên kiến của bạn** (mỗi quy tắc kiểm tra được bằng số hoặc việc làm):
  1. **Quy tắc quét mù 15 giây (Blind Scan First):** Ở mỗi ảnh mới, luôn ẩn hoàn toàn layer nhãn máy trong 15 giây đầu tiên. Tự mắt đếm nhanh số lượng xe ba bánh và người đi bộ trong ảnh trước khi bật hiển thị box máy.
  2. **Quy tắc ngắt chuỗi chấp nhận (Streak Breaker):** Không bao giờ bấm phím Accept liên tiếp quá 8 đối tượng. Cứ mỗi 8 box được chấp nhận, bắt buộc phải phóng to (zoom in) ít nhất 1 box để kiểm tra độ tiếp xúc 4 cạnh viền với sai số ≤ 2 px.
  3. **Quy tắc nghỉ mắt và chuyển đổi cảnh (Fatigue Management):** Bắt buộc dừng lại nghỉ giải lao 3 phút sau mỗi lô 8 ảnh duyệt liên tục để khôi phục độ nhạy thị giác, tránh hiện tượng mỏi mắt dẫn đến bỏ sót các bẫy `loose` và `missing`.

## 5. Mask SAM cho 3 vật (Part 5)

| Ảnh | SAM (phiên bản) | Số click | Mask gồm người lái / hàng hoá? | Lỗi quan sát | File trong `masks/` |
|---|---|---|---|---|---|
| `img_013.jpg` | SAM 2.1 (`sam2.1_b.pt`) | 3 click (2 pos, 1 neg) | Đã tách thành công người lái ra khỏi thân xe nhờ click âm đặt tại ngực tài xế | Đường viền kính chắn gió bị rách nhẹ ở phần phản quang | `tuktuk_img013.jpg` |
| `img_003.jpg` | SAM 2.1 (`sam2.1_b.pt`) | 3 click (2 pos, 1 neg) | Mask ôm sát thân xe ba gác và bao trọn thùng chở rau quả gắn liền | Một số kẽ hở giữa các nan hoa bánh xe bị phủ kín thành mảng đặc | `cargotrike_img003.jpg` |
| `img_004.jpg` | SAM 2.1 (`sam2.1_b.pt`) | 3 click (2 pos, 1 neg) | Mask ôm sát toàn bộ khung tuk-tuk và mui xe, không ăn vào hành khách ngồi sau | Phần bóng đổ dưới gầm xe bị lẹm một góc khoảng 15 px vào mask | `tuktuk_img004.jpg` |

- Đối chiếu guideline mục 4.2: Quy tắc 4.2.1 (bám sát đường biên nhìn thấy) và 4.2.3 (bóng đổ không thuộc vật, kính xe thuộc vật) rất hữu ích để xác định điểm click âm/dương. Tuy nhiên, guideline v1.0 thiếu quy định cụ thể về việc xử lý các lỗ rỗng cơ học phức tạp (như khung giàn kim loại hở, nan hoa xe) — cần quy định rõ diện tích lỗ rỗng tối thiểu (> 5% diện tích vật) mới cần khoét bỏ.
- *(Tuỳ chọn)* Interactor SAM trong CVAT: Thực hiện offline qua script `sam2_points.py` cho phép kiểm soát chính xác toạ độ pixel của từng click và lưu trữ tức thì đa giác vector; trong khi Interactor SAM tích hợp trong CVAT đòi hỏi cấu hình serverless Nuclio nhưng đem lại trải nghiệm tương tác trực quan thời gian thực trên giao diện web.

## 6. Calibration (Part 6)

- Số quyết định: 4 quyết định (`calibration_decisions.csv`); quan trọng nhất là **CAL-01** (giải quyết triệt để xung đột phân loại `three_wheeler` vs `car`/`truck` thông qua kiến trúc open-vocabulary) và **CAL-02** (phân định rõ ràng xe điện hai bánh).
- Quy tắc mới đã vào `guideline_v1.1_calibrated.md`:
  1. Phân loại xe điện hai bánh dựa trên bàn đạp trợ lực (`bicycle`) vs sàn để chân (`motorbike`).
  2. Xe ba bánh có mui che / khung cabin luôn là `three_wheeler`, không gán `car` hay `truck`.
  3. Quy tắc ngưỡng diện tích nhìn thấy ≥ 20% cho người ngồi trong khoang xe ba bánh hở sườn.
  4. Bắt buộc vẽ `ignore_region` cho các cụm phương tiện nhỏ ở xa (chiều cao < 20 px) hoặc đỗ dày đặc không phân biệt được cá thể.
  5. Mask xe bao gồm cả kính chắn gió và kính chiếu hậu.
- Nguyên nhân gốc: Phần lớn xuất phát từ **guideline v1.0 còn mở** (chưa bao quát hết các phương tiện mới như xe điện) kết hợp với **hạn chế từ vựng đóng của mô hình detector COCO** (chưa được huấn luyện trên phương tiện đặc thù bản địa).

## 7. QC và quyết định (Part 6)

- Mẫu: 5 ảnh (0 ca khó + 5 ảnh ngẫu nhiên, seed 42), gồm 112 đối tượng; người kiểm: bạn (Vũ Văn Đăng — **hạn chế**: người kiểm định cũng chính là người thực hiện review nên có thể tồn tại điểm mù chủ quan cố hữu).
- Lỗi theo **đối tượng**: 0/112 = 0.0% · theo **ảnh**: 0/5 = 0.0% (đáp ứng ngưỡng yêu cầu Release ≤ 2%).
- Top lỗi: Không ghi nhận lỗi phân loại, bỏ sót hay vị trí nào trong mẫu 5 ảnh kiểm định (`img_003.jpg`, `img_006.jpg`, `img_014.jpg`, `img_031.jpg`, `img_035.jpg`).
- **Giới hạn mẫu nhỏ** (cận trên 95% trong `qc_report_auto.md`): Mặc dù ghi nhận 0 lỗi trên 5 ảnh mẫu, cận trên khoảng tin cậy 95% Wilson của tỷ lệ ảnh lỗi thực tế vẫn lên tới **43%**. Điều này có nghĩa là về mặt thống kê toán học, một mẫu nhỏ 5 ảnh chưa đủ sức mạnh để chứng minh tuyệt đối rằng toàn bộ lô 24 ảnh có tỷ lệ lỗi dưới 2%. Cần mở rộng cỡ mẫu lên ≥ 20 ảnh trong các dự án sản xuất thực tế.
- Script kiểm được / không kiểm được: Script tự động kiểm tra được tính toàn vẹn của tệp JSON, kiểm tra giới hạn bounding box trong ảnh, phát hiện box nhỏ < 20 px, đo IoU và thống kê bẫy; nhưng **hoàn toàn không thể kiểm tra được** độ ôm sát thực tế của đường viền quanh vật thể, tính đúng đắn ngữ nghĩa của nhãn trên bối cảnh thực, và những đối tượng bị cả người lẫn máy cùng bỏ sót ngoài tập gold.

**Quyết định: RELEASE**, ngày 2026-11-19. Lý do (đi theo cây Hold → Rework → Release):
1. Kiểm tra điều kiện **HOLD**: Không vi phạm (các câu hỏi mở của guideline đã được chốt và thể chế hóa trong v1.1; dữ liệu hình ảnh nguyên vẹn, không có ảnh hỏng/mờ).
2. Kiểm tra điều kiện **REWORK**: Không vi phạm (tỷ lệ ảnh có lỗi QC trong mẫu kiểm định là 0.0% <= 2%; không phát hiện lỗi sai có tính hệ thống; tỷ lệ bắt bẫy đạt 83.3% >= 80%).
3. Kết luận **RELEASE**: Bộ nhãn 24 ảnh batch và 4 ảnh gold đã đạt toàn bộ các tiêu chí nghiệm thu của taxonomy và rubric, sẵn sàng chuyển giao cho giai đoạn huấn luyện mô hình.

## 8. Ghi chú cá nhân

- **Dùng AI để làm bài**: Dùng trợ lý AI hỗ trợ giải thích các khái niệm trong taxonomy, gợi ý cú pháp lệnh PowerShell/Python và kiểm tra định dạng tệp báo cáo; toàn bộ nhãn gán tay, quyết định phân loại và phân tích số đo đều do học viên trực tiếp thực hiện và xác nhận bằng script kiểm thử `validate_labels.py` và `check_submission.py`.
- Gold có bị nhiễm không: **KHÔNG**. 4 ảnh gold (`img_001.jpg`, `img_012.jpg`, `img_020.jpg`, `img_036.jpg`) được tự tay gán nhãn ở Part 1 độc lập hoàn toàn trước khi chạy bất kỳ mô hình AI nào lên các ảnh này.
- Máy / GPU / CVAT dùng; vướng mắc cài đặt: Máy cá nhân Windows 11 (64-bit), CPU Intel, không có GPU rời NVIDIA. CVAT bản v2.74.1 triển khai qua Docker Desktop. Khi chạy YOLOE ban đầu gặp lỗi thiếu gói `clip` do môi trường ảo thiếu công cụ build, đã xử lý dứt điểm bằng `uv pip install git+https://github.com/ultralytics/CLIP.git`.
- Dùng cache `data/cache/` thay vì tự chạy? Có sử dụng tệp dự đoán chạy sẵn trong `data/cache/` (`yolo26m_all_c005.coco.json` và `yoloe26s_three_wheeler_all_c005.coco.json`) cho bước gộp nhãn toàn bộ 36 ảnh ở ngưỡng thấp 0.05 ở Part 3 và Part 4 do chạy suy luận 36 ảnh trên CPU tốn nhiều thời gian; tuy nhiên ở Part 2 đã tự chạy tay thực tế cả 4 lệnh thử nghiệm trên tập pilot và kiểm chứng trực quan qua các thư mục `viz/`.
- Lỗi / bug script đã gặp: Lệnh PowerShell nuốt chuỗi rỗng `--thr ""` khi chạy `merge_prelabels.py`, đã xử lý bằng cách bỏ qua cờ `--thr` để script tự động áp dụng ngưỡng `--conf 0.3`.
- Một điều bạn làm khác lần sau: Sẽ bố trí quy trình kiểm tra chéo độc lập (cross-review) giữa hai người và tăng kích thước mẫu QC lên tối thiểu 15 ảnh để thu hẹp khoảng tin cậy Wilson, giúp quyết định Release có cơ sở thống kê vững chắc hơn.
