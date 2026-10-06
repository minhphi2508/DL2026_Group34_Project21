# Quyết định v3 trước huấn luyện — 05/10/2026

V2 có gain mask và phục hồi rõ nhưng 19/21 gates, miss nhỏ/biên và semantic false positives. V3 kiểm một paired-low sampling policy train-only, giữ loss/threshold/validation/gates cũ và warmstart cả2learnedheads. Đây hypothesis, không có v3trainedquality evidence.

CPU train-only audit: low components815→1581, <=64px92→169, <=256px428→803 tại generation20 sample. Full missing pixels giảm3.485m→2.083m do bớt mid/high exposure: phải đo regression high. Negative150 giữ nguyên, paired context chưa phải semantic labels/hard mining. Không mang100source validation hoặc test sangtrain.

15epochs/3600s, earlystop5. Một run research bounded, không sweep nhiều cấu hình. So candidate thực dụng với immutablev2; warmstart/extra epochs ngăn quy causal gain chỉcho sampling. Nếu v3 kém, giữv2 cho đồán và reportlimits. Cảnumericfail vàsemanticfail cầnreport, không viết PASS bằngviệc nớigates. Không cầnperfect-allinputs đểgiữ kết quảhữuích, nhưng khôngđánhđồng prototype với phục hồiidentityđúng.

Maskquality vàinpaintingquality táchriêng. Telea cóblur ởlỗlớn ngayGToracle; MATfrozen làneuralalternative cóevidence hạnchế, không tựđổi default hoặc searchmodels. Stagechoices vàsuppliedmask phải tườngminh, không GT/trackrouting. Root owns integratedqualitydecision/operator; nguồnfrozen/trainingdata/validationkhôngoverwrite. Testchỉsaubothfinalfreeze documents.
