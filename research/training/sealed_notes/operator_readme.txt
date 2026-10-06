# Thí nghiệm v3: paired low-severity replay

Ứng viên nghiên cứu cho đồ án, kế thừa đầy đủ hai head của v2 epoch20. V2 giữ nguyên. Không phải resume optimizer/epoch của v2 và không đổi checkpoint production. Kết quả chỉ biết sau chạy và so sánh validation; không hứa đạt mọi tiêu chí.

## Một thay đổi sampling có giới hạn

600 item mỗi epoch, giữ mixture scratch180/missing120/combined150/clean50/noise50/age50. Chọn45 positive non-low của missing và45 của combined, chuyển sang low; ghép cùng source/crop/dihedral với45clean hoặc45noise. 420 ordinary slots giữ nguyên theo counterfactual fresh generation. Low positives tăng missing40→85, combined50→95. 510 unique source mỗi epoch, đủ600 source qua15epochs theo fixture. Cặp cùng context không phải hard-negative mining theo prediction hay nhãn mặt/áo. Native flaws của ảnh historical chưa gán nhãn hoàn chỉnh.

Generation epoch mới0→20, tránh phát lại đúng seed v2 đã dùng. Kiến trúc, BCE+Dice/weights, threshold scratch0.4/missing0.5, validation1600 và tất cả21gates giữ nguyên. Không global morphology, threshold search hoặc GT routing. Warmstart cộng thêm training là confound: không suy cải thiện là hiệu quả nhân quả của sampling so với unpaired fine-tune nếu chưa có ablation đó.

## Ngân sách

Tối đa15epochs mới, early-stop5; budget3600s ở epoch boundary, không gồm setup/preflight/smoke/pack và epoch đang chạy có thể vượt biên. AdamW/lr3e-5/scheduler reset; không dùng optimizer/RNG cũ. CUDA BF16 nếu hỗ trợ, không FP16. FP32 convolution/matmul TF32 off, precision highest. Parent checksum/model-state parity và CUDA batch4 backward smoke phải đạt trước optimizer.

## Nguồn và đầu ra

Parent assets/parent_v2_epoch20.pt SHA797fd66ef68cadf33173843f99ae3041e9b499aadddacf3485cba6315ac12fe9. Parent protocol1bd0ddab9b3d60bd0ef9297ae7ead0aca47aa7c2ac87da2dcd9b0dc6b0447f53, bundle d2530399c8c8cd754592b557ae95a4054061d861cf757d6a99e6d9b880867347. Parent numeric eligibility FALSE được giữ trung thực.

DataRoot chỉ train/val hash-pinned từ original input packet, không test assets. Lệnh runner có preflight/smoke/train/evaluate; full train chỉ explicitCUDA. CPU smoke không optimizer/checkpoint. Các tensor HxW/RGBuint8 và mask binary giữ kích thước, pad32 không resize.

Run riêng candidate_v3_pair_replay. Chọn best_eligible nếu có, nếu không giữ best_unconstrained để phân tích; numeric PASS không tự promote. Không cần inference1600thêm để xuất masks: dùng cached epoch được chọn. LIGHT<250MiB gửi lại; FULL chứa optimizer/RNG/source/parent để Drive backup và resume đúng epochcompleted. Không restore FULL vào candidate/v2 hoặc di chuyển venv.

Gói launcher do final tạo sau CPU proof/seal. Nếu runtime cũ mất, dùng bootstrap original --setup-only trước v3, không chạy lại train v1. Không mở test trong phát triển; cuối cùng chỉ test sau hai freeze docs.
