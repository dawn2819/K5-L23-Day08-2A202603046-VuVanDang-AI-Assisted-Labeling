# Label contract — Phố châu Á — BỔ SUNG v1.1 — 2A202603046 - Vũ Văn Đăng

> Bản bổ sung quy tắc cho guideline v1.0 dựa trên kết quả chạy thử các mô hình AI (YOLO26, YOLOE, SAM), các ca nghi ngờ (`can_review`), bẫy lọt trong quá trình review và chốt các câu hỏi mở.

## 1. Câu hỏi mở đã chốt (từ mục 6 của v1.0)

| # | Câu hỏi mở | Quyết định chốt | Ảnh ví dụ | `case_id` |
|---|---|---|---|---|
| 1 | Xe đạp điện / xe máy điện nhỏ: `bicycle` hay `motorbike`? | Xe chạy điện có bàn đạp trợ lực (pedal-assist) và khung xe đạp mảnh tính là `bicycle`. Xe máy điện hoàn toàn (không có bàn đạp, có sàn để chân, đèn pha/xi-nhan lớn) tính là `motorbike`. | `img_004.jpg` | CAL-02 |
| 2 | Khối xe máy đỗ dày đặc gần camera: từng xe hay `ignore_region`? | Nếu xe ở tiền cảnh/trung cảnh và tách biệt được từng xe (cao ≥ 20 px) thì vẽ từng box `motorbike`. Chỉ vẽ `ignore_region` khi các xe xếp lớp dày đặc chồng chéo không thể phân biệt ranh giới từng xe. | `img_018.jpg` | CAL-04 |
| 3 | Hàng hoá nhô ra khỏi xe đạp / ba gác: box xe có gồm hàng không? | Nếu hàng hoá chất trong giỏ/thùng gắn cố định trên xe thì tính vào box xe. Nếu hàng nhô ra khỏi xe > 20% chiều dài xe (như chở thanh tre dài, ống nước) thì box chỉ ôm phần thân xe, không ôm phần nhô dài ra. | `img_003.jpg` | CAL-01 |
| 4 | Người ngồi trong xe ba bánh có mui che: ngưỡng "thấy ≥ 20%"? | Nếu nhìn thấy rõ đầu, mặt và vai/thân (diện tích nhìn thấy ≥ 20%), gán 1 box `person` riêng ôm sát phần nhìn thấy xuyên qua khung xe. Nếu chỉ thấy lờ mờ tay chân hoặc bóng lưng (< 20%) thì bỏ qua không gán. | `img_020.jpg` | CAL-03 |
| 5 | Kính chắn gió / cửa sổ xe: thuộc mask xe hay không? | Kính chắn gió, kính cửa sổ và kính gương chiếu hậu thuộc toàn bộ vào mask của xe (`car`, `three_wheeler`), kể cả khi nhìn xuyên qua kính thấy người bên trong. Người bên trong vẫn có polygon/box `person` riêng. | `img_012.jpg` | CAL-01 |
| 6 | Box máy chồng nhau quá nhiều trong cảnh phức tạp / ban đêm? | Giữ lại box có độ tin cậy cao nhất hoặc ôm sát nhất, xóa các box trùng lặp (IoU ≥ 0.6 cùng lớp). Không để nhiều box bao quanh cùng một thực thể vật lý. | `img_024.jpg` | CAL-04 |
| 7 | Mức tin cậy tối thiểu để chấp nhận box máy không cần sửa? | Không đặt ngưỡng tin cậy để tự động chấp nhận (accept mù). Dù box máy có `conf` = 0.95 vẫn bắt buộc phải nhìn qua mắt để kiểm tra ranh giới ôm sát và nhãn lớp. | `img_013.jpg` | CAL-01 |

## 2. Ca khó mới (từ lỗi của máy, bẫy lọt, box `can_review`)

| Ảnh | Tình huống | Máy làm gì (model nào) | Quyết định của bạn | Quy tắc mới (vào mục nào của v1.0) |
|---|---|---|---|---|
| `img_013.jpg` | Xe tuk-tuk chở khách trên đường phố Bangkok | YOLO26 gọi là `car` hoặc `truck` do thiếu lớp xe ba bánh trong tập COCO; SAM gộp người lái vào thân xe | Gán `three_wheeler` cho xe, tách người lái thành box `person` riêng. Dùng YOLOE với text prompt để pre-label | Mục 2 & Mục 4.2: Xe ba bánh có mui che luôn là `three_wheeler`; mask thân xe không gộp người lái |
| `img_004.jpg` | Xe tuk-tuk đỗ bên lề đường, có người ngồi bên trong | YOLO26 nhận diện thành `car`; box bao quanh lỏng lẻo | Sửa nhãn thành `three_wheeler`, co chặt 4 cạnh box ôm sát khung xe và mui xe | Mục 4.1.1: Box phải tiếp xúc 4 điểm ngoài cùng của thân xe và mui xe |
| `img_020.jpg` | Hành khách ngồi trong xe tuk-tuk bị thanh kim loại che ngang | Máy bỏ sót hành khách do bị che khuất | Thêm box `person` cho hành khách vì phần đầu và thân trên vẫn nhìn thấy rõ (≥ 20%) | Mục 4.1.4: Người ngồi trong phương tiện công cộng hở sườn vẫn gán `person` |
| `img_021.jpg` | Cảnh ngã tư đông đúc, hàng chục xe máy chen lấn ở hậu cảnh | Máy bỏ sót một số xe máy ở xa; bẫy `missing` bị lọt do mắt reviewer bị quá tải | Khoanh vùng hậu cảnh xa bằng `ignore_region`, không cố vẽ từng xe cao < 20 px | Mục 4.1.6: Bắt buộc dùng `ignore_region` cho các cụm phương tiện nhỏ ở xa |
| `img_027.jpg` | Xe máy đang di chuyển ở cự ly trung bình | Box bị bẫy `loose` (nới rộng bất thường), reviewer bỏ sót do viền box trông có vẻ bao trọn xe | Phải co viền box ôm sát lốp xe và đuôi xe (sai số cho phép ≤ 2 px) | Mục 4.1.1: Kiểm tra kỹ 4 cạnh box, không bấm Accept khi cạnh box thừa khoảng trống lớn |

## 3. Lịch sử thay đổi

| Phiên bản | Thay đổi | Lý do (ca nào, `case_id`) |
|---|---|---|
| v1.0 | Bản guideline gốc của giảng viên | Khung ban đầu cho bài thực hành |
| v1.1 | 1. Bổ sung quy tắc phân loại xe điện (`bicycle` vs `motorbike`).<br>2. Chốt quy tắc gán nhãn `three_wheeler` (xe ba bánh có mui không gán `car`/`truck`).<br>3. Quy định cụ thể về người ngồi trong xe ba bánh bị che khuất (ngưỡng 20%).<br>4. Làm rõ phạm vi áp dụng `ignore_region` cho các cụm xe máy ở hậu cảnh xa.<br>5. Bổ sung quy tắc mask xe bao gồm cả phần kính chắn gió. | Rút ra từ lỗi nhận diện của YOLO26 (`img_013.jpg`), câu hỏi mở v1.0 (`img_004.jpg`), ca `can_review` (`img_020.jpg`), và bẫy lọt review (`img_021.jpg`, `img_027.jpg`). Tham chiếu: CAL-01, CAL-02, CAL-03, CAL-04. |
