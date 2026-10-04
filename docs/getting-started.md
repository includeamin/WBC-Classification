# Getting started

## Install

Requires Python 3.12+ and [Poetry](https://python-poetry.org/).

```bash
git clone https://github.com/includeamin/WBC-Classification.git
cd WBC-Classification
poetry install --extras kaggle
```

## Get the data

```bash
poetry run wbc download-data --dest data
```

See [Data](data.md) for the expected layout and Kaggle credentials.

## Train, evaluate, predict

```bash
poetry run wbc train -c configs/resnet18.yaml
poetry run wbc evaluate runs/resnet18-<timestamp>/best.pt --output runs/resnet18-<timestamp>/test_metrics.json
poetry run wbc predict path/to/cell.jpeg -c runs/resnet18-<timestamp>/best.pt
```

Use `poetry run wbc --help` or the [CLI reference](reference/cli.md) for all options.
Training on a CPU is slow for the pretrained backbones; see [Training](training.md) for GPU and Colab.
