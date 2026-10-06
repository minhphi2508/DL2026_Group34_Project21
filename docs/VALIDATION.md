# Kiểm chứng bản tích hợp Windows

Ngày kiểm tra: **2026-10-06**. Recipe: **full Wan + V3 missing mask**, CPU FP32, TF32 tắt. Đây là kiểm chứng tính đúng của tích hợp và khả năng chạy; không phải một benchmark chất lượng mới.

## Đối chiếu với kết quả nghiên cứu đã chọn

Chạy lại 5 ảnh development đã được cho phép, bao gồm 3 chân dung, một cảnh có vẹt và một ảnh sạch đối chứng. Input bao gồm JPEG/JFIF, WebP và PNG. Cả 5 ca đều **khớp từng pixel** với bản nghiên cứu tại bốn điểm: ảnh xử lý đầu vào, mask kết hợp, global restoration và output final. Ba ca có mặt chạy cả face enhancement/blending; hai ca không có mặt giữ global output.

Biên bản máy đọc được: [`INTEGRATION_PARITY.json`](INTEGRATION_PARITY.json). Không đưa ảnh cá nhân vào Git. Không mở tập test benchmark để tinh chỉnh cấu hình tích hợp. Các thí nghiệm nghiên cứu trước đó có phạm vi riêng, ghi trong [`RESEARCH.md`](RESEARCH.md).

## Môi trường và kiểm tra chức năng

- Windows 11, Python **3.12.14 64-bit**; môi trường `.venv` riêng, không dùng system site-packages.
- PyTorch **2.8.0+cpu**, torchvision **0.23.0+cpu**; các dependency trực tiếp được khóa phiên bản trong `requirements.txt`.
- `pip check`: không có dependency bị thiếu/xung đột.
- `doctor.py --device cpu`: phép tính CPU thật thành công; **76 file source upstream** và **7 model** khớp SHA256.
- Bốn kiểm tra input thành công: JFIF/WebP/tên Unicode và sidecar, trùng stem khác định dạng, ảnh hỏng/quá nhỏ, folder không quét đệ quy.
- Một chân dung được chạy lại trọn pipeline trong môi trường riêng: output final khớp từng pixel với bản đã đối chiếu. [`ISOLATED_CPU_CHECK.json`](ISOLATED_CPU_CHECK.json) xác nhận cả 87 file có hash ràng buộc giữ nguyên byte trong Git index.
- Ba URL archive chính thức phản hồi HTTPS Range hợp lệ. Các archive lưu sẵn có đúng hash; installer hỗ trợ lấy sáu model ngoài Git từ cache đã xác minh.

Danh sách package thực tế của môi trường CPU nằm ở [`ENVIRONMENT_CPU_WINDOWS.txt`](ENVIRONMENT_CPU_WINDOWS.txt); đây là biên bản môi trường, không dùng nó thay profile CUDA khi cài RTX.

## Phạm vi thực sự đã kiểm chứng

Các kiểm tra trên dùng CPU. Chưa chạy suy luận CUDA của bản đóng gói trên máy RTX; `SETUP_RTX.cmd` kiểm tra CUDA thực trên máy người dùng trước khi xác nhận cài thành công. Không hứa output CUDA khớp từng pixel CPU.

Ảnh `native_display` chỉ resize LANCZOS để xem cùng kích thước input. Chạy thành công không bảo đảm mọi vết hỏng đã được sửa, không chứng minh bảo toàn chi tiết mặt, và không thay thế đánh giá trực quan.

Recipe và hash mã/model được ghi trong [`../PIPELINE_FROZEN_BEFORE_TEST.yaml`](../PIPELINE_FROZEN_BEFORE_TEST.yaml) trước các kiểm tra tiếp theo. Đây không phải chứng nhận freeze hồi tố cho các thử nghiệm cũ.
