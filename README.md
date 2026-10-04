# WBC Classification

[![CI](https://github.com/includeamin/WBC-Classification/actions/workflows/ci.yml/badge.svg)](https://github.com/includeamin/WBC-Classification/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/includeamin/WBC-Classification)](https://github.com/includeamin/WBC-Classification/releases)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/github/license/includeamin/WBC-Classification)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-mkdocs-informational)](https://includeamin.github.io/WBC-Classification/)

Classify white blood cell images — **eosinophil, lymphocyte, monocyte, neutrophil** — with PyTorch.
Fine-tune a pretrained ResNet / EfficientNet or train the small `baseline` CNN, all from one CLI.

## Quick start

```bash
git clone https://github.com/includeamin/WBC-Classification.git
cd WBC-Classification
poetry install --extras kaggle

poetry run wbc download-data --dest data
poetry run wbc train -c configs/resnet18.yaml
poetry run wbc evaluate runs/<run>/best.pt --output metrics.json
poetry run wbc predict path/to/cell.jpeg -c runs/<run>/best.pt
```

Requires Python 3.12+ and [Poetry](https://python-poetry.org/). Training pretrained models is best done on a GPU
(see the [training guide](https://includeamin.github.io/WBC-Classification/training/) and `notebooks/train_colab.ipynb`).

## Documentation

Full docs: <https://includeamin.github.io/WBC-Classification/> — data setup, configuration reference,
CLI and API reference, architecture, results.

## Dataset

[Blood Cell Images](https://www.kaggle.com/datasets/paultimothymooney/blood-cells) on Kaggle (not stored in this repository).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Releases and the changelog are automated from
[Conventional Commits](https://www.conventionalcommits.org/); see [CHANGELOG.md](CHANGELOG.md).

## License

BSD 3-Clause — see [LICENSE](LICENSE).
