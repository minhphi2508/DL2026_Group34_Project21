# Nghiên cứu hướng 1: ghép detector, giữ nguyên bộ phục hồi Wan

Ngày: 06/10/2026. Trạng thái: thí nghiệm phát triển đã chạy; không đổi pipeline mặc định.

## Câu hỏi và thiết kế

Khi giữ nguyên Wan phục hồi ảnh và mặt, thay nguồn mask có cải thiện chất lượng không? So sánh ba chính sách:

| Nhánh | Mask đưa vào Wan |
|---|---|
| S0 hiện tại | Wan scratch OR V3 missing |
| S1 V3 hai head | V3 scratch OR V3 missing |
| S2 union | Wan scratch OR V3 scratch OR V3 missing |

Cả ba dùng cùng RGB đã được Wan chuẩn hóa kích thước, cùng trọng số và cùng các bước scratch+quality restoration → tìm mặt/align → FaceSR 256 → warp/blend. Chỉ nguồn mask thay đổi. V3 chạy trên ảnh native đã dùng trong các thí nghiệm trước; scratch >=0.4 không nới biên, missing >=0.5 và nới ellipse bán kính 3 pixel native. Resize mask bằng PIL NEAREST; không tìm threshold mới, không fine-tune, không mask tay, không C3/MAT/FFDNet. CPU FP32, TF32 tắt, 2 luồng, cạnh lớn tối đa 512, HR tắt.

Protocol được ghi trước lần suy luận V3 mới, có hash của script, helper và model. Mask Wan và dữ liệu đã chuẩn hóa được tái sử dụng có kiểm tra hash. Mask S0 khớp đúng pixel với kết quả lịch sử trên cả 22 điều kiện. Đây là so sánh chính sách tích hợp cố định, không phải cuộc thi kiến trúc detector ở mọi cấu hình; head scratch V3 không có margin riêng.

## Dữ liệu và giới hạn

Mask được kiểm tra trên 5 chân dung thật + 10 ảnh cảnh người dùng + 5 ảnh còn lành làm đối chứng + 2 probe xước giả lập. Hai probe dùng lại nguồn astronaut/camera của đối chứng, nên 22 điều kiện không phải 22 nguồn ảnh độc lập. Ảnh thật không có ảnh sạch đối chiếu; không tính PSNR/SSIM chất lượng cho chúng. Năm ảnh lành chỉ là đối chứng đại diện, không chứng minh an toàn trên mọi ảnh lành.

Pilot phục hồi 9 ca được chốt trước kết quả mask: cậu bé, người phụ nữ, người đàn ông; thuyền buồm 27, vẹt 30; hai probe xước; astronaut và coffee còn lành. Chạy đủ 3 nhánh, tổng 27 output. Có 22 suy luận global mới và 5 global S0 tái sử dụng đã gắn hash; các bước mặt được chạy riêng cho từng nhánh. Không dùng benchmark test trong nghiên cứu này.

## Kết quả mask

| Chân dung | Wan scratch (%) | V3 scratch (%) | S2 thêm pixel so với S0 |
|---|---:|---:|---:|
| photo_00_boy | 11.4953 | 0.0164 | 11 |
| photo_01_woman | 15.2004 | 0.1364 | 46 |
| photo_01_webp | 7.5763 | 0.0897 | 46 |
| photo_02_man | 16.0772 | 0.0005 | 0 |
| photo_03_girl | 10.3430 | 0.1776 | 24 |

V3 scratch tìm rất ít vùng trên các chân dung đã xem. Diện tích mask lớn hơn không tự chứng minh detector chính xác hơn; ở đây ảnh overlay và output cho thấy S1 thực sự bỏ sót nhiều vết nứt nhìn thấy. Hiện tượng phù hợp với khả năng lệch phân bố giữa xước giả lập và vết rách thật, nhưng thí nghiệm này chưa chứng minh nguyên nhân cụ thể.

| Probe xước | Nhánh | Recall (%) | Precision (%) | Dice (%) |
|---|---|---:|---:|---:|
| scratch_only_astronaut | S0_current | 72.12 | 21.23 | 32.80 |
| scratch_only_astronaut | S1_v3_both | 36.29 | 62.91 | 46.03 |
| scratch_only_astronaut | S2_union | 75.50 | 21.88 | 33.93 |
| scratch_only_camera | S0_current | 53.85 | 21.86 | 31.10 |
| scratch_only_camera | S1_v3_both | 28.01 | 71.07 | 40.19 |
| scratch_only_camera | S2_union | 59.99 | 23.54 | 33.81 |

S2 tăng recall trên hai probe. S1 có precision/Dice cao hơn nhưng recall thấp hơn: mask hẹp ít sửa nhầm cũng bỏ sót nhiều xước. Ground truth là nét mảnh; mask rộng bao quanh xước bị tính false positive ở viền. Vì vậy không chọn pipeline chỉ theo Dice hoặc diện tích mask.

| Đối chứng còn lành | S0 mask (%) | S1 mask (%) | S2 mask (%) | S2 thêm pixel |
|---|---:|---:|---:|---:|
| control_astronaut | 0.7633 | 0.7301 | 0.7912 | 73 |
| control_camera | 0.4662 | 0.4414 | 0.4688 | 7 |
| control_coffee | 2.4914 | 1.2602 | 2.5164 | 43 |
| control_rocket | 4.8677 | 1.4991 | 4.9712 | 178 |
| control_flower | 0.1366 | 0.1366 | 0.1366 | 0 |

S2 thêm tổng 301 pixel trên 4/5 đối chứng còn lành; S0 vốn đã có mask giả dương. Các vùng như bọt cà phê hoặc phản chiếu có thể bị nhầm là hư hại. OR nhiều detector không có khả năng tự loại giả dương của detector trước.

## Kết quả phục hồi có reference

Reference của probe giữ nguyên nền đã nhuộm màu/noise trước khi vẽ thêm xước. PSNR/SSIM đo sự phục hồi về nền này, không phải chất lượng ảnh lịch sử nói chung. Tách output global và output sau phục hồi mặt vì FaceSR có thể thay đổi chi tiết không trùng reference.

| Probe | Nhánh | PSNR global (dB) | SSIM global | PSNR cuối (dB) | MSE vùng xước global |
|---|---|---:|---:|---:|---:|
| scratch_only_astronaut | Input | 18.470 | 0.72858 | — | 15436.22 |
| scratch_only_astronaut | S0_current | 22.082 | 0.69643 | 22.058 | 1333.31 |
| scratch_only_astronaut | S1_v3_both | 21.274 | 0.68018 | 21.264 | 3621.73 |
| scratch_only_astronaut | S2_union | 22.183 | 0.70206 | 22.163 | 1205.78 |
| scratch_only_camera | Input | 19.080 | 0.72928 | — | 13413.88 |
| scratch_only_camera | S0_current | 21.149 | 0.57535 | 21.149 | 3166.10 |
| scratch_only_camera | S1_v3_both | 20.207 | 0.55525 | 20.207 | 4839.63 |
| scratch_only_camera | S2_union | 21.667 | 0.59110 | 21.667 | 2416.60 |

### Bảo toàn ảnh còn lành

Bảng sau đo độ giống input, không phải điểm chất lượng phục hồi. Các ảnh này không cần khử xước; phục hồi toàn ảnh và mặt vẫn có thể làm thay đổi nét và màu.

| Đối chứng | Nhánh | PSNR global (dB) | SSIM global | PSNR cuối (dB) | SSIM cuối |
|---|---|---:|---:|---:|---:|
| control_astronaut | S0_current | 27.041 | 0.82353 | 27.025 | 0.82297 |
| control_astronaut | S1_v3_both | 27.204 | 0.82301 | 27.182 | 0.82200 |
| control_astronaut | S2_union | 26.945 | 0.82302 | 26.946 | 0.82224 |
| control_coffee | S0_current | 25.910 | 0.80395 | 25.910 | 0.80395 |
| control_coffee | S1_v3_both | 26.147 | 0.81755 | 26.147 | 0.81755 |
| control_coffee | S2_union | 25.688 | 0.80297 | 25.688 | 0.80297 |

## Quan sát ảnh thật

Xem `comparison/<id>/FINAL_COMPARISON.png` theo thứ tự Input, S0, S1, S2. `GLOBAL_COMPARISON.png` loại ảnh hưởng FaceSR; `FACE_EFFECTS.png` tách trước/sau mặt. `diagnostics/<id>/MASK_COMPARISON.png` cho đủ 22 trường hợp. Giữ output raw và bản native_display chỉ phóng bằng LANCZOS để dễ xem, không gọi đó là super-resolution.

Nhận xét quan sát của trợ lý được ghi riêng trong `VISUAL_REVIEW.json` sau khi xem ảnh. Chưa thu thập đánh giá mù của người dùng. Số pixel output thay đổi và MAE giữa hai nhánh chỉ đo khác biệt, không chứng minh cải thiện chất lượng.

## Thời gian và kiểm chứng

| Nhánh | Global mới cả batch (giây) | Tìm mặt + FaceSR + blend (giây) |
|---|---:|---:|
| S0_current_new4 | 93.61 | 0.00 |
| S0_current | 0.00 | 21.80 |
| S1_v3_both | 157.23 | 20.14 |
| S2_union | 139.83 | 21.64 |

S0_current_new4 chỉ chạy global cho 4 ca; 5 ảnh thật S0 tái sử dụng global. Không so tốc độ S0 cache với batch 9 ảnh mới của S1/S2. Các thời gian không bao gồm mọi lần khởi động model/V3 hay tạo bảng so sánh.

S0 cuối khớp đúng pixel với cả 5 ảnh thật lịch sử trong pilot. Kiểm tra nguồn input, trọng số, geometry, protocol và strict checkpoint load giữ nguyên. `CHECKS.json` ghi kiểm tra thí nghiệm; `FINAL_AUDIT.json` kiểm tra PNG và 63 file Python chính thức Wan không thay đổi. ZIP có CRC và hash mọi thành viên trong `PACKAGE_CHECKS.json`.

## Ý nghĩa cho bài nghiên cứu pipeline

Thí nghiệm tách riêng đóng góp của lựa chọn detector khỏi bộ phục hồi. Kết quả âm của S1 cho thấy head riêng được huấn luyện không tự động chuyển tốt sang ảnh rách thật. S2 kiểm tra sự bổ sung thông tin, đồng thời có chi phí giả dương. Cần xét đồng thời mask, ảnh sau phục hồi, đối chứng ảnh lành và face refinement; không lấy một chỉ số tổng hợp làm bằng chứng duy nhất.

Đây là nghiên cứu phát triển trên ảnh đã biết. Không tuyên bố S0/S2 tối ưu toàn cục, hiệu quả trên mọi ảnh lịch sử hoặc đạt điểm benchmark mới. Báo cáo giữ cả ca thất bại; không đổi default hoặc push GitHub dựa riêng vào hai probe.

Tài liệu/code chính: Microsoft Bringing Old Photos Back to Life, commit 33875eccf4ebcd3665cf38cc56f3a0ce563d3a9c; model V3 epoch 14 theo descriptor đã kiểm tra. Đóng góp dự án nằm ở dataset/huấn luyện detector, cách tích hợp và thí nghiệm kiểm soát; bộ phục hồi Wan dùng pretrained có ghi nguồn.

Các kết luận cuối của vòng này nằm trong `RESEARCH_DECISION_VI.md`, lập sau khi xem đủ output pilot.
