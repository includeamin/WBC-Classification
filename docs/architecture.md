# Architecture

```
wbc_classification/
├── config.py          pydantic config + YAML loading + overrides
├── data/              labels, datasets/loaders, transforms, Kaggle download
├── preprocessing/     optional HSV cell segmentation and crop
├── models/            baseline (IncludeNet), torchvision backbones, registry
├── engine/            train, evaluate, predict, metrics, checkpoints, runtime
└── cli.py             `wbc` Typer application
```

Design notes:

- A checkpoint is one `.pt` file holding weights, class names and the full config, so `predict` and `evaluate` need nothing else.
- Errors at the boundaries (config, data, checkpoint) are `WBCError` subclasses; the CLI prints them without a traceback.
- Adding a model means adding an entry in `models/registry.py`.
- The cell crop is optional; when no cell is found the full image is used and a warning is logged.
