# Nghiên cứu thành phần và thứ tự pipeline — 06/10/2026

## Quyết định sau đợt thử

**Giữ Microsoft restoration + Final Version missing mask làm pipeline mặc định đã chọn.** Hai vị trí ghép FFDNet trước/sau Microsoft chưa mang lại cải thiện đủ rõ để thay baseline trên các ảnh thực của pilot. Chưa mở rộng hai nhánh này lên toàn bộ 15 ảnh development vì điều kiện “có lợi ích rõ” của kế hoạch chưa đạt.

Kết quả quan trọng cho yêu cầu 2 của đề tài: thay đổi mask, bước mặt và thứ tự khử nhiễu có tác động khác nhau; thêm bước không đảm bảo chất lượng cao hơn. Pipeline phù hợp ảnh hư hại hỗn hợp không nhất thiết phù hợp ảnh chỉ có noise. Đây là kết luận từ các điều kiện đã kiểm tra, không khẳng định tối ưu toàn cục.

## Bộ thử và các can thiệp

Tổng hợp cached ablation A/B/C trên đủ 5 chân dung: A Microsoft restoration, B Microsoft restoration + Final Version, C B trước bước mặt. Đã có inference thật và receipt từ lần trước; lần này kiểm tra hash và tái trình bày, không báo cache là inference mới.

Pilot thứ tự gồm 6 điều kiện:

- 3 chân dung thực: bé trai `images.jfif`, người phụ nữ `test.jpg`, người đàn ông `images (2).jfif`.
- 1 cảnh vật: ảnh vẹt 30, dùng mask Microsoft+Final Version cũ, **không dùng C3**.
- 2 probe noise Gaussian luminance sigma 12 đã có: camera/astronaut 512 pixel, seed cố định. Mask rỗng được định nghĩa trước để cô lập bộ phục hồi trên noise, không coi chúng là phép đo end-to-end detector của pipeline thực tế.

| Nhánh | Công thức | Inference mới |
|---|---|---|
| B | RGB raw + mask cố định → Microsoft global → face nếu tìm được | Reuse global đúng input/mask; chạy lại toàn bộ bước face |
| D | FFDNet(RGB raw tại frame Microsoft) + cùng mask → Microsoft global → face | FFDNet + 6 global mới + bước face |
| E | Microsoft global đã cache → FFDNet → face | FFDNet + bước face; global giống B |

FFDNet: luminance, tự ước lượng sigma từ input, blend 0,75, sigma_scale 1, tile 0, CPU threads 2. R: Microsoft scratch+quality, FP32, cùng weights và frame tối đa 512. F: dlib68, `Setting_9_epoch_100` 256px, warp/blend chính thức. Không fine-tune, chỉnh ngưỡng hay redetect mask sau khử nhiễu.

D/E có cùng mask ở từng ảnh nên so sánh vị trí N không bị lẫn việc thay đổi detector. Các bước face dùng cùng thuật toán nhưng phát hiện/căn chỉnh trên output tương ứng; báo cả kết quả trước mặt để tách ảnh hưởng này.

## Final Version đóng góp gì?

| Chân dung | Pixel union mask Final Version bổ sung vào Microsoft | Nhận xét từ ảnh đã kiểm tra |
|---|---:|---|
| Bé trai | 1 | Khác nhỏ, không có lợi ích rõ để quy cho Final Version |
| Người phụ nữ | 157 | Bổ sung vùng mắt; hỗ trợ sửa vùng mất trước bước face |
| Em bé có nơ | 0 | A/B khớp output; Final Version không bổ sung lợi ích ở ca này |
| Người đàn ông | 508 | Chưa cứu được bong tróc nặng; có thay đổi không mong muốn ở chữ ngày giờ |
| Bé gái ảnh hồng | 0 | A/B khớp output; nếp gấp lớn còn tồn tại |

Pixel được thêm không phải thước đo chất lượng. Microsoft là bộ phục hồi toàn ảnh, nên một thay đổi nhỏ của mask có thể ảnh hưởng nhiều pixel output. Bước face còn phụ thuộc landmark/alignment; không diễn giải số pixel thay đổi như số pixel được sửa đúng.

## Bước mặt đóng góp gì?

Trong 5 chân dung, B phát hiện 1 mặt ở mỗi ảnh trong 4 ảnh. Ảnh người đàn ông không được phát hiện mặt, nên output C trước mặt và B cuối giống nhau. Bước mặt giúp tăng chi tiết mắt/miệng/da trên một số ca, nhưng sinh nội dung mới và không tự chứng minh đúng người thật.

Trên bé gái ảnh hồng, bước mặt làm mắt/miệng rõ hơn nhưng nếp gấp lớn vẫn còn. Điều này cho thấy face enhancement không thay thế việc phát hiện/sửa hư hại đúng ở global stage. Trong pilot D/E, số mặt phát hiện giống B: bé trai, người phụ nữ và astronaut mỗi ảnh 1 mặt; camera, người đàn ông và ảnh vẹt không có mặt được phát hiện.

## Thứ tự khử nhiễu: định lượng trên hai probe

PSNR trước bước face để tách tác động của thứ tự khử nhiễu. FFDNet chạy riêng là đối chứng chuyên noise đã cache; audit xác nhận output đó khớp từng pixel với input được đưa vào Microsoft của D.

| Probe | Input | B Microsoft | D FFDNet → Microsoft | E Microsoft → FFDNet | FFDNet riêng |
|---|---:|---:|---:|---:|---:|
| Astronaut PSNR dB | 27,04 | 27,20 | 27,26 | 27,30 | **33,97** |
| Camera PSNR dB | 26,68 | 25,37 | **24,18** | 25,44 | **32,76** |
| Astronaut SSIM | 0,5918 | 0,7630 | 0,7884 | 0,7732 | **0,8877** |
| Camera SSIM | 0,5433 | 0,7888 | 0,7933 | 0,7950 | **0,8615** |

D không thắng nhất quán: trên camera, PSNR giảm khoảng 1,19 dB so với B dù SSIM nhích lên. E tăng PSNR khoảng 0,10/0,07 dB và SSIM nhẹ trên hai probe; mức tăng này chưa chứng minh đáng thêm vào mặc định cho ảnh thực.

FFDNet riêng giữ chi tiết tham chiếu tốt hơn rõ so với các chuỗi có Microsoft ở hai ca noise-only này. Microsoft còn làm mượt/đổi nội dung hoặc tương phản toàn ảnh; bộ phục hồi ảnh cũ không mặc nhiên là lựa chọn thích hợp cho ảnh hiện đại chỉ thêm Gaussian noise. Không suy rộng hai probe thành mọi loại noise, grain, JPEG hay ảnh cũ.

Sau face, astronaut B/D/E có PSNR lần lượt 27,16/27,22/27,25; SSIM 0,7625/0,7881/0,7725. Bước mặt không giúp chỉ số fidelity ở probe hiện đại này. Camera không có bước face nên chỉ số cuối giữ nguyên. Các điểm này không đánh giá đúng danh tính của chân dung thực.

## Nhìn ảnh thực của pilot

| Ca | Quan sát B/D/E | Quyết định |
|---|---|---|
| Bé trai | Khác nhẹ texture mặt, mũi và độ mượt; không thấy cải thiện hư hại rõ từ thêm N | Giữ B |
| Người phụ nữ | Khác nhẹ vùng mặt/nền; còn checker ở đỉnh và seam/texture; không có ưu thế rõ của D/E | Giữ B |
| Người đàn ông | Cả ba vẫn mờ/bong tróc, không phát hiện mặt; N không giải quyết bottleneck | Ca thất bại còn nguyên |
| Vẹt | Cả ba còn hai mảng trắng lớn và nhiều scratch; N không bổ sung vùng mất vào mask | Giữ B; vấn đề chủ yếu ở detection/mask và tái tạo |

Các nhận xét trên là kiểm tra hình ảnh phát triển của người thực hiện, không phải nghiên cứu người dùng độc lập. Đã xuất panel ẩn nhãn P/Q/R với key riêng để người dùng có thể đánh giá tiếp; chưa thu thập điểm blind của người dùng và không giả báo đã có thống kê sở thích.

## Chi phí và độ tin cậy thực nghiệm

- D chạy global 6 ảnh trong 169,28 giây trên laptop CPU, trung bình batch khoảng 28,21 giây/ảnh; không coi đây là thời gian của mọi kích thước ảnh.
- N trước Microsoft: 2,06 giây cho 6 ảnh, cộng 0,05 giây khởi tạo. N sau Microsoft: 2,22 giây, cộng 0,17 giây khởi tạo. Chi phí N khoảng 0,27–0,44 giây/ảnh trong batch này.
- Detect/face/blend cả batch: B 16,79 giây, D 15,76 giây, E 15,60 giây. Chênh lệch nhỏ của các lần chạy không chứng minh phương pháp này nhanh hơn phương pháp kia.
- Global B/E được cache; **không so tổng thời gian cache B/E với global inference mới của D** để tuyên bố speedup. Runtime thật của các bước cached có receipt lịch sử, nhưng các batch/kích thước khác nhau.
- 18 final outputs của 6 điều kiện × 3 nhánh; thêm 5 cached comparison A/B/C. Số lượng output không phải số nguồn ảnh độc lập.
- B chạy lại bước mặt và khớp output cuối cũ từng pixel trên cả 4 ca thực. Masks D khớp byte-for-byte masks B. Cùng frame, input/weights/source giữ nguyên; refs chỉ dùng sau inference.
- `CHECKS.json`: PASS. Audit cuối kiểm tra lại 63 file Python upstream, weights, cached outputs và tính toàn vẹn ảnh/gói. Không đọc payload benchmark test, không dùng các điểm Telea cũ cho Microsoft.

## Kết luận cho yêu cầu 2 và hướng bàn giao

Giữ **Microsoft scratch detector + Final Version missing → Microsoft global → face khi phát hiện được → blend** cho phục hồi ảnh hư hại hỗn hợp. Giữ FFDNet như lựa chọn chuyên noise riêng; chưa thêm N mặc định vào Microsoft và chưa xây một bộ tự động phân loại/routing từ các kết quả này.

Phần nghiên cứu gồm ba câu hỏi có bằng chứng: nguồn mask, đóng góp bước mặt, vị trí khử nhiễu. Các nhánh C3/MAT/scratch nhiều tỷ lệ của đợt trước bổ sung cả cải thiện lẫn phản ví dụ. Kết quả loại một cấu hình không hữu ích cũng là kết luận thực nghiệm, nhưng không gọi nó là đề xuất model mới hoặc chứng minh tối ưu toàn cục.

Bước hoàn thiện là đưa recipe B vào repo Windows và cập nhật tài liệu/báo cáo theo các findings này. Bản baseline đã chốt vẫn giữ nguyên; đợt thử này không thay default GitHub.

## Artifacts

- `cached_ablations/<id>/ABC_COMPARISON.png`, `MASK_COMPARISON.png`: Microsoft/Final Version/face.
- `comparison/<id>/FINAL_COMPARISON.png`, `GLOBAL_COMPARISON.png`, `FACE_EFFECTS.png`: B/D/E cuối, trước mặt và tác động mặt.
- `FACE_DETAIL_CROP.png` ở bé trai/người phụ nữ chỉ là crop chẩn đoán sau inference, không mask tay hay đầu vào scoring.
- `BLINDED_COMPARISON.png`: input/P/Q/R; key ở `BLINDED_KEYS.json`.
- `NOISE_PSNR_GLOBAL.png`, `NOISE_SCORES.csv`, `CACHED_ABLATIONS.csv`, `FACE_AND_OUTPUT_EFFECTS.csv`: biểu đồ và bảng số.
- `PROTOCOL.json`, `INPUT_MANIFEST.json`, `CACHE_PROVENANCE.json`, `TIMINGS.json`, receipts, checks và delivery manifest: cấu hình và audit.
- Native display chỉ resize LANCZOS; raw frame tối đa 512, không SR. Gói không chứa weights/dataset đầy đủ và không phải app standalone.
