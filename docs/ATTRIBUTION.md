# Attribution and third-party rights

This repository integrates original student work with pretrained upstream components. It does not claim authorship of the Microsoft restoration networks or pretrained weights.

- **Microsoft:** pinned official source commit `33875eccf4ebcd3665cf38cc56f3a0ce563d3a9c`; original Python source unchanged. MIT notice remains at [`third_party/wan/LICENSE`](../third_party/wan/LICENSE). Official README describes code and pretrained models under that license; obtain weights from its official releases. Cite Microsoft Research for global and face restoration.
- **Synchronized BatchNorm:** Jiayuan Mao, pinned commit `7553990fb9a917cddd9342e89b6dc12a70573f5b`; MIT at [`third_party/sync_batchnorm/LICENSE`](../third_party/sync_batchnorm/LICENSE). Source is copied into the two import locations used by Microsoft.
- **Final Version detector:** group fine-tuned two-head U-Net/ResNet34, epoch 14. segmentation_models.pytorch supplies the MIT architecture implementation; ResNet/ImageNet ancestry and training protocol are recorded in provenance. Fine-tuning does not transfer backbone authorship.
- **dlib:** Boost-licensed library; this Windows setup uses the `dlib-bin` wheel distribution to avoid a local compiler. The 68-point predictor is a separate asset trained on iBUG 300-W. The [official dlib example](https://www.dlib.net/face_landmark_detection_ex.cpp.html) notes that the training dataset excludes commercial use and directs commercial users to resolve rights with Imperial College. This project is an academic demonstration, not a commercial license grant.
- **Dependencies:** PyTorch/torchvision, OpenCV, Pillow, scikit-image and other packages retain their package licenses. Installed wheels include their notices.
- **Data:** historical training data are distributed through the download in DATA.md; personal development portraits are not committed to Git. Per-image rights remain with their original sources.

The Windows compatibility wrapper uses CPU-safe checkpoint loading and restores the historical floating-mask dtype expected by the upstream blender. It does not replace network topology or inference layers. Model and source hashes are checked before inference.

- **FFDNet comparison:** official KAIR gray weights and the group adapter, MIT notice at [`research/denoising/LICENSE_KAIR.txt`](../research/denoising/LICENSE_KAIR.txt). Used only in processing-order experiments.
- **Auxiliary fixtures:** scikit-image sample images and their controlled degradations used in the original development probes. See https://scikit-image.org/docs/stable/api/skimage.data.html for sample source attribution. They are distinct from the Finna historical core.
