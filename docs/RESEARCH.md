# Pipeline experiments

The project addresses both restoration and the effect of processing steps on final visual quality. The selected pipeline is Microsoft scratch detection + Final Version missing detection, followed by Microsoft global and optional face restoration.

| Study | Held fixed | Compared | Development finding |
|---|---|---|---|
| Mask sources | Inputs, processing geometry and restoration models | Microsoft scratch + group missing; both group heads; union of all three | Expanded union improved two scratch probes but added false positives on controls; retain the selected policy |
| Denoising placement | Damage masks and restoration models | Baseline; FFDNet before; FFDNet after | Benefits were inconsistent; additional denoising is excluded from the default pipeline |
| Face refinement | The same global result | Before and after face restoration/blending | Appearance can improve while facial details change; missed and false detections remain limitations |

[REPRODUCIBILITY.md](REPRODUCIBILITY.md) provides executable Windows commands. Original measurements and protocols are grouped under [results](results). Exact auxiliary fixtures are included under `research/fixtures`.

These are development comparisons. Multiple degraded conditions derived from one photograph are not independent sources. Personal photographs without clean references receive visual assessment, not paired PSNR/SSIM. Original experiment records may contain historical names and absolute paths; these are evidence of the original runs rather than configuration instructions for a new installation.

The detector's original numerical gate remains false; the supplied checkpoint was selected through academic review. Its segmentation metrics do not establish final restoration quality. No new independent held-out restoration benchmark or universal optimum is claimed.
