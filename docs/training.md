# Training

```bash
poetry run wbc train -c configs/resnet18.yaml            # pretrained ResNet-18
poetry run wbc train -c configs/baseline.yaml            # small baseline CNN
poetry run wbc train -c configs/resnet18.yaml --epochs 5 --device cpu
poetry run wbc -v train -c configs/resnet18.yaml         # log every epoch
```

Each run writes `runs/<name>-<timestamp>/`:

| File | Content |
|---|---|
| `config.yaml` | the exact configuration used |
| `metrics.csv` | per-epoch loss / accuracy / learning rate |
| `curves.png` | training curves |
| `best.pt` / `last.pt` | checkpoints (weights + class names + config) |

Training uses AdamW, a cosine learning-rate schedule, mixed precision on CUDA and early stopping on validation accuracy.
Set `model.freeze_backbone_epochs` to train only the new classification head for the first N epochs.

## GPU

`poetry install` installs the default PyTorch build. For a specific CUDA build follow the
[PyTorch install guide](https://pytorch.org/get-started/locally/) and install that wheel into the Poetry environment.
On Google Colab use `notebooks/train_colab.ipynb`.
Training on CPU is fine for the `baseline` model and for smoke tests, but slow for pretrained backbones.
