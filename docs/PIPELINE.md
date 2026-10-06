# Final restoration pipeline

The selected configuration combines Microsoft's **Bringing Old Photos Back to Life** with the group's **Final Version missing-region detector**.

1. Decode the first image frame into RGB.
2. Create the Microsoft processing image with a maximum side of 512 pixels; upstream code determines the exact processing dimensions.
3. Detect scratches with the official Microsoft detector, threshold 0.4.
4. Predict missing regions at native resolution using the group's epoch-14 checkpoint, threshold 0.5. Expand the mask with a radius-3 ellipse, then align it to the processing image using nearest-neighbour resizing.
5. Combine both masks and run Microsoft's scratch-and-quality restoration model.
6. Detect and align faces. When faces are found, apply the published 256-pixel face model and official blending; otherwise retain the global output.
7. Save final PNGs and optional original-size display copies. Display resizing uses LANCZOS, not learned super-resolution.

Inference uses FP32 with TF32 disabled. The group's scratch head, additional denoising and experimental expanded mask policies are excluded from the default configuration. Model and source identities are checked through the runtime manifests in `provenance/`.

The group contributes the dataset preparation, controlled degradation labels, detector training, mask integration, component experiments and Windows workflow. Microsoft's pretrained restoration networks and the U-Net/ResNet34 architecture remain credited to their authors.

The configuration was selected on examined development cases. Large missing regions, residual scratches, colour changes and generated face details remain limitations. Execution success does not establish restoration quality or a globally optimal pipeline.

See [reproduction commands](REPRODUCIBILITY.md), [experiment findings](RESEARCH.md), [verification scope](VALIDATION.md) and [attribution](ATTRIBUTION.md).
