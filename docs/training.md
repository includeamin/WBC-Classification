# Training

```bash
poetry run wbc train -c configs/resnet18.yaml            # pretrained ResNet-18
poetry run wbc train -c configs/baseline.yaml            # small baseline CNN
poetry run wbc train -c configs/resnet18.yaml --epochs 5 --device cpu
poetry run wbc train -c configs/resnet18.yaml --set data.num_workers=4 --set train.seed=1
poetry run wbc -v train -c configs/resnet18.yaml         # log every epoch
```

Each run writes `runs/<name>-<timestamp>/`:

| File | Content |
|---|---|
| `config.yaml` | the exact configuration used |
| `metrics.csv` | per-epoch loss / accuracy / learning rate |
| `curves.png` | training curves |
| `confusion_matrix.png` | validation confusion matrix of the best epoch |
| `best.pt` / `last.pt` | checkpoints (weights + class names + config) |

Use `--set section.field=value` (repeatable) to override any configuration field, such as
`train.seed`, `train.patience` or `data.num_workers`; see [Configuration](configuration.md).

Training uses AdamW, a cosine learning-rate schedule, mixed precision on CUDA and early stopping on validation accuracy.
Set `model.freeze_backbone_epochs` to train only the new classification head for the first N epochs.
During the frozen epochs only the classification head's weights are trained; BatchNorm layers in the frozen
backbone still update their running statistics (the model stays in train mode), and early-stopping patience also
counts the frozen epochs.

## GPU

`poetry install` installs the default PyTorch build. For a specific CUDA build follow the
[PyTorch install guide](https://pytorch.org/get-started/locally/) and install that wheel into the Poetry environment.
On Google Colab use `notebooks/train_colab.ipynb`.
Training on CPU is fine for the `baseline` model and for smoke tests, but slow for pretrained backbones.
