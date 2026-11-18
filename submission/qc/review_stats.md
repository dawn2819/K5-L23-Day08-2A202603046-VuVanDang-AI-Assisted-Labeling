# Số đo review (det)

- Ảnh tính: 24 (batch,gold); box máy: 506; đối tượng sau duyệt: 507; ngưỡng accepted: IoU >= 0.9
- Đã **loại 23 đối tượng nằm quanh các bẫy** khỏi bảng này: lỗi nhân tạo không phản ánh chất lượng máy.

## 1. Bốn trạng thái (AI error profile)
| Trạng thái | Số | Tỉ lệ | Mẫu số |
|---|---|---|---|
| accepted (giữ nguyên) | 440 | 440/506 = 87.0% | 506 box máy |
| edited (sửa) | 51 | 51/506 = 10.1% | 506 box máy |
| deleted (xoá) | 15 | 15/506 = 3.0% | 506 box máy |
| added (vẽ thêm) | 16 | 16/507 = 3.2% | 507 đối tượng sau duyệt |

Trong số deleted + added, **0 trường hợp là đổi lớp** (máy gán sai lớp, box đúng chỗ): đó không phải máy bỏ sót. Số vật máy **sót thật** ≈ added − đổi lớp = 16.

### Theo lớp
| Lớp | accepted | edited | deleted | added |
|---|---|---|---|---|
| person | 235 | 31 | 5 | 11 |
| bicycle | 6 | 1 | 1 | 0 |
| motorbike | 122 | 9 | 7 | 4 |
| car | 45 | 9 | 2 | 1 |
| bus | 3 | 0 | 0 | 0 |
| truck | 7 | 0 | 0 | 0 |
| three_wheeler | 22 | 1 | 0 | 0 |

## 2. So với gold (nhãn gán tay từ đầu)
Ảnh gold: 4; đối tượng gold: 47 (mẫu số của recall)
| | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|
| Máy (pre-label) | 12 | 61 | 35 | 16.4% (KTC 95%: 10–27%) | 25.5% (KTC 95%: 15–40%) |

*Lạc quan: ngưỡng được chọn trên chính các ảnh gold này (in-sample), và chỉ có 4 ảnh. Đọc khoảng tin cậy, đừng đọc con số đơn.*


(Ảnh gold không nằm trong nhãn sau duyệt: không tính residual / new error. Đó là chủ ý: duyệt lại ảnh bạn vừa tự gán tay sẽ đo trí nhớ, không đo thiên kiến.)

## 3. Ảnh bẫy
- Bắt được 10/12 = 83.3% (KTC 95%: 55–95%) (ngưỡng đạt: 80%). Chỉ 12 bẫy nên khoảng tin cậy rất rộng: đọc cùng danh sách bẫy lọt, đừng kết luận chỉ từ %.
  - extra: 3/3
  - missing: 2/3
  - wrong_class: 3/3
  - loose: 2/3
  - lọt: img_027.jpg (loose), img_021.jpg (missing)

## Gợi ý đọc số
- `deleted` cao → ngưỡng thấp hoặc ánh xạ nhãn sai; `added` cao → máy bỏ sót (xem lớp nào);
- `accepted` thấp (< 50%) → sửa mỗi box lâu hơn vẽ mới: đổi model/ngưỡng hoặc vẽ tay;
- bắt bẫy < 80% → duyệt lại lô, siết checklist (quét ảnh TRƯỚC khi nhìn nhãn máy).
