# Quyết định sau vòng nghiên cứu ghép detector

**Giữ S0: Wan scratch + V3 missing → Wan global → Wan face/blend.** Không thay default hoặc bản freeze hiện tại. S2 giữ như nhánh nghiên cứu có lợi ích giới hạn; S1 không chọn làm phương án phục hồi xước ảnh thật ở cấu hình đã thử.

Đã chạy mask trên 22 điều kiện và phục hồi đủ 3 nhánh trên 9 ca, thành 27 output. Thiết kế được chốt trước suy luận mới; không fine-tune, không tìm threshold, không đánh mask tay, không dùng benchmark test. Các probe và đối chứng chia sẻ một số nguồn, nên không gọi đây là 22 ảnh độc lập.

## Căn cứ

1. **S1 chỉ dùng hai head V3:** bỏ sót nhiều vết nứt rõ trên cậu bé, phụ nữ và hai ảnh cảnh. Mask scratch V3 trên 5 chân dung chỉ phủ 0.0005–0.178% ảnh ở cấu hình native; Wan scratch phủ 7.58–16.08%. Độ phủ không tự chứng minh độ chính xác, nhưng ảnh trước/sau xác nhận nhiều vết xước không được xử lý. Trên hai probe, recall và chất lượng phục hồi thấp hơn S0, dù precision/Dice mask có thể cao hơn do mask hẹp.

2. **S2 OR thêm V3 scratch:** có bổ sung thông tin trên probe. Recall astronaut tăng 72.12% → 75.50%, camera 53.85% → 59.99%. PSNR trước bước mặt tăng tương ứng 22.082 → 22.183 dB và 21.149 → 21.667 dB. MSE trong vùng xước giảm khoảng 9.6% và 23.7%. Đây là cải thiện có đo được trên hai probe đã biết, không nên bỏ qua.

3. **Đánh đổi của S2:** thêm 301 pixel mask trên 4/5 đối chứng còn lành. Trong hai ca đối chứng chạy phục hồi, PSNR bảo toàn input giảm nhẹ so với S0: astronaut 27.041 → 26.945 dB; coffee 25.910 → 25.688 dB. Đây là số đo độ giống input, không phải điểm chất lượng ảnh cũ. Trên 5 ca thật phục hồi, khác biệt thị giác S0/S2 nhỏ; ảnh đàn ông giống đúng pixel. Hai nhánh vẫn để lại các mảng trắng lớn ở thuyền/vẹt. Thêm head scratch không giải quyết được vùng mất chưa được mask nhận ra.

4. **Giới hạn chung:** S0/S2 làm mờ khuôn mặt ảnh đàn ông, còn nhiều vết xước và vùng mất trên ảnh cảnh. Không coi S0 hoàn hảo hoặc thắng từng ảnh. FaceSR cũng cần đánh giá riêng: đối chứng astronaut có một vùng huy hiệu bị dlib nhận nhầm là mặt trong cả ba nhánh; đây là vấn đề chung, không phải lợi thế hay bất lợi riêng của cách ghép mask. Không sửa vấn đề này trong vòng detector để giữ thí nghiệm nhất quán.

## Ý nghĩa của quyết định

Chưa có bằng chứng lợi ích nhất quán trên ảnh thật để đổi cấu hình đã chọn thành OR tất cả. Việc giữ S0 là quyết định phát triển thận trọng trước deadline, không phải khẳng định S2 luôn kém. S2 có thể trình bày như ablation: tăng recall nhưng có chi phí giả dương; lựa chọn detector và bộ phục hồi phải được đánh giá đồng thời.

Thí nghiệm này bổ sung trực tiếp ý 2 của đề: cùng model phục hồi nhưng thay cách phát hiện/ghép vùng hỏng tạo ra đầu ra khác nhau, và chỉ số mask cao chưa chắc đồng nghĩa ảnh phục hồi tốt hơn. Kết quả âm của S1 và ca hỏng chung đều được giữ để báo cáo trung thực.

Ảnh so sánh nằm trong `comparison/`; mask của đủ 22 điều kiện trong `diagnostics/`. Mỗi bảng có Input, S0 hiện tại, S1 V3 hai head, S2 union theo thứ tự trái sang phải. `RESEARCH_RESULTS_VI.md` chứa bảng đầy đủ, `VISUAL_REVIEW.json` ghi nhận xét sau xem ảnh. Báo cáo chưa có đánh giá mù độc lập của người dùng và chưa phải kết quả benchmark test.
