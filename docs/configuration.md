# Configuration

Experiments are described by a YAML file (see `configs/`). Unknown keys are rejected so typos fail fast.
Common values have dedicated flags (`wbc train -c cfg.yaml --epochs 5 --lr 0.0003`).
Any field can be overridden with the repeatable `--set section.field=value` option; values are parsed as YAML,
and `--set` takes precedence over the dedicated flags:

```bash
wbc train -c configs/resnet18.yaml --set data.num_workers=4 --set train.seed=1
```

```yaml
preprocessing:
  crop: false            # crop to the detected cell first (HSV segmentation)
data:
  root: data             # folder with TRAIN/ and TEST/
  image_size: 224        # square input size
  batch_size: 32
  val_fraction: 0.2      # stratified split from TRAIN, 0 < x < 0.5
  num_workers: 0
model:
  name: resnet18         # baseline | resnet18 | resnet50 | efficientnet_b0
  pretrained: true       # ImageNet weights (backbones only)
  freeze_backbone_epochs: 0
train:
  name: run              # run folder prefix
  epochs: 30
  lr: 0.001
  weight_decay: 0.0001
  patience: 7            # early-stopping patience (epochs)
  amp: true              # mixed precision (CUDA only)
  seed: 42
  device: auto           # auto | cpu | cuda | mps
  output_dir: runs
```
