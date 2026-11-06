# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy: 6/11**

**Runtime Colab:** GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=Imagenet-1k
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Nói về toàn bộ bức ảnh vì mô hình tổng hợp của ảnh là 640 x 428 quét để dán nhãn chung chứ không khoanh vùng

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

model sử dụng Imagenet-1k có 1000 lớp đã huấn luyện sẵn chứ ko tự nghĩ ra
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
ghi class id để máy nhận dạng còn ghi class name cho con người taxonomy thì nhận dạng model sử dụng

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Vật to nhất thì gán nhãn, ưu tiên chủ thể giữa khung hình hoặc đang thực hiện hành động chính

6. Vì sao không được lấy `score` của model làm ground truth?
vì 0.5~ chỉ là dự đoán độ tự tin từ model chứ không phải quy tắc con người xác lập thực tế guideline ground truth mới là nhãn chuẩn con người đặt ra
## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=oven ; `score`=0.6343 ; `bbox_xyxy`=[489.09, 200.73, 617.81, 344.88] ; `bbox_width`=128.72 ; `bbox_height`=144.15
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Góc trên-trái : X = 489.09 px, Y = 200.73 px.
Góc dưới-phải : X = 617.81 px, Y = 344.88 px.
Chiều rộng: 128.72 px.
Chiều cao: 144.15 px..

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
0.20 có 17 vật , 0.60 có 6 vật

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
reviewer sẽ phải loại bỏ các hộp duplicate trùng lặp và dự đoán sai false positive

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
bounding box là bảo trọn toàn vẹn đối tượng không chừa quá nhiều hay cắt hụt chi tiết chính của vật thể

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
đưa lên hỏi lab coach escalation vì chưa rõ guideline có yêu cầu gắn nhãn cho vật thể chỉ xuất hiện một phần trong ảnh hay không. Cần hỏi Lab Coach để xác nhận có nên đánh dấu vật thể bị cắt mép và chỉ khoanh vùng phần nhìn thấy hay không.
Không nên tự đặt quy tắc mới khi chưa có hướng dẫn thống nhất, vì điều đó có thể khiến các thành viên annotation xử lý dữ liệu không nhất quán.
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=3480 ; 3–5 điểm đầu của `polygon_xy`=446.0, 70.0 / 445.0,71.0 / 444.0,71.0 / 443.0,72.0 / 441.0, 73.0
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
poly gon cung cấp thêm thông tin đa dạng về đường viền và hình dạng thực tế của vật thể
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
trong trường hợp có nhiều hơn một vật thể thì instance id như định dạng cho vật thể đó vd một bếp có 3 ghế
máy tính chỉ hiểu là chair trong khi instance thì nó sẽ thành chair-001 chair-002 chair-003

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
quy tắc vẽ polygon: sát vật thể không thừa không thiếu người gán nhãn dựa vào logic để nối tiếp
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
escalation vì Chưa rõ guideline có yêu cầu gắn nhãn cho vật thể chỉ xuất hiện một phần trong ảnh hay không. Cần hỏi Lab Coach để xác nhận có nên đánh dấu vật thể bị cắt mép và chỉ khoanh vùng phần nhìn thấy hay không.

Không nên tự đặt quy tắc mới khi chưa có hướng dẫn thống nhất, vì điều đó có thể khiến các thành viên annotation xử lý dữ liệu không nhất quán.
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một lớp cho cả ảnh theo taxonomy ImageNet-1K | Top-1 có thể không khớp trực giác cảnh giao thông (prediction ≠ GT) | Đọc guideline chọn đúng 1 lớp cho ảnh | Đúng taxonomy, đúng phạm vi cấp ảnh, không chép score thành nhãn |
| Phát hiện vật thể | Mỗi object: class + box xyxy | Hạ threshold ra thêm hộp nhiễu; object cắt mép | Vẽ box theo rule, đủ object trong guideline | Box có ôm sát không, thiếu/thừa object, ca che khuất |
| Instance segmentation | Mỗi instance: class + polygon | Biên hai vật chạm nhau / vùng mờ | Vẽ mask bám biên từng instance | Mask có tràn object khác không, đủ instance cùng lớp ||

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
   Chỉ dùng ảnh COCO công khai có sẵn trong notebook; không tải ảnh cá nhân, khuôn mặt, biển số hay dữ liệu nội bộ lên Colab/GitHub.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
   Dừng thao tác và báo Lab Coach / GV phụ trách lab.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
