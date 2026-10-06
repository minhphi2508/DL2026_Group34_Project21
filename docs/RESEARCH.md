# Nghiên cứu pipeline

Đề có hai yêu cầu: phục hồi ảnh cũ bằng deep learning và nghiên cứu ảnh hưởng của các bước/pipeline đến chất lượng cuối. Bản final chọn Wan scratch + V3 missing, Wan global rồi face refinement.

## Các đối chứng đã chạy

| Thí nghiệm | Giữ cố định | Thay đổi | Kết luận phát triển |
|---|---|---|---|
| Ghép detector | Ảnh, frame, Wan global và face weights | Wan scratch + V3 missing / V3 hai head / union cả ba | V3 scratch riêng bỏ sót nhiều trên ảnh thật. Union giúp hai probe xước nhưng thêm giả dương trên đối chứng; giữ nhánh hiện tại |
| Thứ tự denoise | Mask và Wan weights | Wan / FFDNet trước Wan / FFDNet sau Wan | Không có lợi ích nhất quán để thêm FFDNet vào default; noise-only probes cho thấy cần xét nhiệm vụ riêng |
| Face refinement | Cùng global output | Trước và sau FaceSR/blend | Có thể cải thiện diện mạo nhưng đổi chi tiết; false face và mặt không được phát hiện vẫn là giới hạn |

Nguồn số đo, protocol và báo cáo được giữ ở `research/detector_fusion` và `research/pipeline_order`. Đây là nghiên cứu trên ảnh development đã biết. Hai probe xước dùng lại nguồn astronaut/camera của đối chứng; không gọi tổng điều kiện là số nguồn độc lập. Ảnh người dùng không có clean reference; không gán PSNR/SSIM chất lượng cho chúng. Các điểm đối chứng là độ bảo toàn input.

Protocol nghiên cứu là snapshot lịch sử với đường dẫn máy thực hiện, không phải cấu hình cần sửa để chạy repo. Code inference chỉ đọc model/protocol cần thiết ở `provenance/`, không truy cập ảnh benchmark, GT hoặc ID/track/severity để chọn policy. `provenance/SELECTED_PIPELINE.json` ghi recipe portable đang đóng gói.

## Đóng góp và giới hạn

Nhóm xây dữ liệu/sinh suy giảm, huấn luyện và kiểm tra V3, tích hợp mask với backend pretrained, thử các pipeline có kiểm soát và làm workflow Windows. Không nhận Wan pretrained là model tự huấn luyện hoặc U-Net/ResNet34 là kiến trúc mới.

Numeric gate của V3 lịch sử vẫn false; checkpoint được chọn qua review học thuật. Điểm test pipeline cũ V3 + Telea không chuyển sang full Wan + V3. Các ảnh test đã được xem trước đây không trở thành bộ test độc lập mới. Không mở benchmark test trong quá trình đóng gói.

Pipeline final vẫn có ca bong tróc nặng, vùng mất lớn và face artifacts không đạt. Giữ kết quả âm trong báo cáo. Lựa chọn final phục vụ phạm vi bài học và dữ liệu phát triển đã biết, không chứng minh tối ưu toàn cục.
