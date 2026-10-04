# WBC Classification

Classify white blood cell images into four types — **eosinophil, lymphocyte, monocyte, neutrophil** — with PyTorch.

- Fine-tune a pretrained backbone (ResNet, EfficientNet) or train the small `baseline` CNN (IncludeNet).
- Config-driven, reproducible training: every run saves its config, metrics, plots and checkpoints.
- An optional OpenCV step crops the cell before classification.
- One command-line tool: `wbc train | evaluate | predict | segment | download-data`.

Start with [Getting started](getting-started.md).
