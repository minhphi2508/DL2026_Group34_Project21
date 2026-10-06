# Nghiên cứu pipeline

Đề có hai yêu cầu: phục hồi ảnh cũ bằng deep learning và nghiên cứu ảnh hưởng của các bước/pipeline đến chất lượng cuối. Bản final chọn Microsoft scratch + Final Version missing, Microsoft global rồi face refinement.

## Các đối chứng đã chạy

| Thí nghiệm | Giữ cố định | Thay đổi | Kết luận phát triển |
|---|---|---|---|
| Ghép detector | Ảnh, frame, Microsoft global và face weights | Microsoft scratch + Final Version missing / Final Version hai head / union cả ba | Final Version scratch riêng bỏ sót nhiều trên ảnh thật. Union giúp hai probe xước nhưng thêm giả dương trên đối chứng; giữ nhánh hiện tại |
| Thứ tự denoise | Mask và Microsoft weights | Microsoft / FFDNet trước Microsoft / FFDNet sau Microsoft | Không có lợi ích nhất quán để thêm FFDNet vào default; noise-only probes cho thấy cần xét nhiệm vụ riêng |
| Face refinement | Cùng global output | Trước và sau FaceSR/blend | Có thể cải thiện diện mạo nhưng đổi chi tiết; false face và mặt không được phát hiện vẫn là giới hạn |

Nguồn số đo, protocol và báo cáo được giữ ở `research/detector_fusion` và `research/pipeline_order`. Đây là nghiên cứu trên ảnh development đã biết. Hai probe xước dùng lại nguồn astronaut/camera của đối chứng; không gọi tổng điều kiện là số nguồn độc lập. Ảnh người dùng không có clean reference; không gán PSNR/SSIM chất lượng cho chúng. Các điểm đối chứng là độ bảo toàn input.

Protocol nghiên cứu là snapshot lịch sử với đường dẫn máy thực hiện, không phải cấu hình cần sửa để chạy repo. Code inference chỉ đọc model/protocol cần thiết ở `provenance/`, không truy cập ảnh benchmark, GT hoặc ID/track/severity để chọn policy. `provenance/SELECTED_PIPELINE.json` ghi recipe portable đang đóng gói.

## Đóng góp và giới hạn

Nhóm xây dữ liệu/sinh suy giảm, huấn luyện và kiểm tra Final Version, tích hợp mask với backend pretrained, thử các pipeline có kiểm soát và làm workflow Windows. Không nhận Microsoft pretrained là model tự huấn luyện hoặc U-Net/ResNet34 là kiến trúc mới.

Numeric gate của Final Version lịch sử vẫn false; checkpoint được chọn qua review học thuật. Điểm test pipeline cũ Final Version + Telea không chuyển sang Microsoft restoration + Final Version. Các ảnh test đã được xem trước đây không trở thành bộ test độc lập mới. Không mở benchmark test trong quá trình đóng gói.

Pipeline final vẫn có ca bong tróc nặng, vùng mất lớn và face artifacts không đạt. Giữ kết quả âm trong báo cáo. Lựa chọn final phục vụ phạm vi bài học và dữ liệu phát triển đã biết, không chứng minh tối ưu toàn cục.
