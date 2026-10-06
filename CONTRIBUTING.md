# Contributing

Use an issue to describe a reproducible problem, Python/Torch versions, CPU/GPU and the stage log. Share images only when you have permission; do not post personal portraits automatically.

Keep the selected model hashes and recipe stable. A change to thresholds, native scaling, mask fusion or a restoration stage needs a separately documented development experiment and review. Never adjust these choices using held-out test payloads. Preserve upstream attribution. New wrapper changes should be checked for matched input/mask/output parity on authorized development cases.
