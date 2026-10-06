# Pipeline cuối được người dùng chọn: full Wan + V3 mask

Quyết định ngày 06/10/2026, sau khi xem các ảnh development và thử thêm nhánh missing C3. **Chốt nhánh B đã chạy trong `five_photo_comparison`: full Wan + V3 missing mask.** C3 không được ghép vào pipeline cuối.

## Luồng xử lý cố định

1. Giải mã ảnh về RGB bằng adapter đầu vào; hỗ trợ các định dạng mà adapter/Pillow thực tế đọc được, gồm JPG/JPEG/JFIF, PNG và WebP đã thử.
2. Tạo bản input Wan với cạnh dài tối đa 512 bằng LANCZOS; detector Wan dùng `input_size=full_size` và xử lý/round kích thước theo upstream.
3. Detector scratch Wan chính thức, ngưỡng 0,4.
4. V3 missing-head trên RGB kích thước native, ngưỡng **>= 0,5**, dilation ellipse bán kính 3 pixel native, rồi resize nearest về frame Wan. **Không OR scratch-head của V3 và không thêm prediction V3 ở scale thứ hai.**
5. OR mask scratch Wan và mask missing V3, đưa vào Wan scratch+quality global restoration.
6. dlib phát hiện mặt và 68 landmarks trên ảnh đã phục hồi; nếu có mặt, dùng `Setting_9_epoch_100` 256 pixel, sau đó warp/blend chính thức. Nếu không phát hiện mặt, giữ output global.

Không thêm C3, MAT prepass/fusion, scratch nhiều tỷ lệ, FFDNet hay Restormer vào luồng mặc định này. Các nhánh đó vẫn là đối chứng hoặc lựa chọn nghiên cứu riêng.

## Cấu hình và nguồn model

Profile đã chạy và kiểm chứng: Windows laptop CPU FP32, TF32 OFF, Torch threads 2, OpenCV threads 1, HR OFF, input Wan tối đa 512. Output raw là kết quả ở frame xử lý; bản cùng kích thước gốc chỉ resize LANCZOS để hiển thị, không phải super-resolution.

Wan: mã nguồn Microsoft pinned commit `33875eccf4ebcd3665cf38cc56f3a0ce563d3a9c`, weights chính thức. V3: checkpoint epoch 14 của nhóm, SHA-256 `73e64cc06d5bcc8960e6b7adc19a25f88cd2ae1a4ddfdc85dd31405dcf4a18d3`. Model face, global, scratch, landmarks và helper bindings nằm trong YAML đi kèm.

Phần của nhóm là model V3 và tích hợp mask/pipeline; không trình bày Wan pretrained là model nhóm tự huấn luyện. Wrapper Windows/CPU và compatibility dtype giữ nguyên như các lần chạy đã kiểm tra.

## Căn cứ chọn và giới hạn

Đây là lựa chọn sản phẩm học thuật dựa trên ảnh development đã biết và quyết định của người dùng. Full Wan + V3 có lợi ích cụ thể ở ca mắt trong `test.jpg`; C3 không cải thiện 4/5 chân dung và nhận nhầm da sáng ở ca bé trai. Không chọn nhánh nhiều module hơn chỉ vì có thêm thành phần.

Vẫn có ca bong tróc nặng chưa đạt, xước/nếp gấp còn sót, ảnh bị làm mượt/đổi màu và chi tiết mặt được sinh lại. Không có ảnh sạch đối chiếu cho ảnh người dùng; không khẳng định đúng danh tính hoặc nội dung lịch sử.

Các điểm, numeric gate và benchmark 1.500 điều kiện của pipeline V3 + Telea cũ **không phải điểm của pipeline Wan mới**. Năm ảnh JOKA đã được người dùng yêu cầu xem trước đây cũng là dữ liệu đã phơi lộ, không dùng làm kiểm tra độc lập cho quyết định này. Không mở hoặc chạy lại benchmark test khi ghi quyết định.

## Trạng thái bàn giao

Recipe này được tích hợp vào `restore.py` của repo mới `minhphi2508/DL2026_Group34_Project21`, có setup và README Windows. Năm điều kiện development gồm JFIF/JPEG, WebP và PNG đã khớp đúng pixel input xử lý, mask, global và final với nhánh lịch sử. Đây là kiểm tra tích hợp, không phải điểm chất lượng hoặc benchmark mới. Xem `VALIDATION.md` để biết phạm vi cài đặt/môi trường/clone đã kiểm chứng.

Quyết định gốc và các freeze/test trước đây được giữ nguyên trong workspace lịch sử. Bản freeze portable của repo này chỉ áp dụng cho kiểm tra tiếp theo sau timestamp của nó; không tuyên bố quyết định có trước các ảnh development đã xem. Không tinh chỉnh cấu hình dựa trên payload benchmark test đã dùng.
