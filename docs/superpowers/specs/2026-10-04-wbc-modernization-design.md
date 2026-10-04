# WBC-Classification Modernization — Design

Date: 2026-10-04
Status: Draft for review

## 1. Goal

Turn WBC-Classification (a Keras CNN that classifies four white-blood-cell types) into a credible open-source project: installable, reproducible, documented, tested, and released automatically. This covers repository structure, the training/inference pipeline, and the model itself.

**Success criteria**

- A stranger can clone the repo, run `poetry install`, and use `wbc train | evaluate | predict | segment | download-data`.
- Training is config-driven and reproducible (seeded, config saved with every run and checkpoint).
- Evaluation is honest: the TEST set is used only for final evaluation.
- Tests, lint, type-check and docs build run in CI; merges to `main` produce a versioned release with changelog.
- A pretrained-backbone model is the main model; a cleaned-up IncludeNet remains as a lightweight baseline. README/docs report metrics only once real numbers exist.

**Non-goals (may come later):** Grad-CAM, ONNX export, multi-GPU, experiment-tracking integrations, rewriting git history, backward compatibility with the old Keras scripts or the `.hdf5` model.

## 2. Decisions (agreed with the project owner)

| Topic | Decision |
|---|---|
| Framework | PyTorch (Keras code is dropped) |
| Dependency/project management | Poetry, `src/` layout, committed `poetry.lock` |
| Python | Develop on 3.14; support `>=3.12,<3.15`; CI matrix 3.12/3.13/3.14 |
| Dependencies | Latest releases at implementation time; caret ranges in `pyproject.toml`, exact pins in the lock |
| Dataset | Removed from the working tree; download script + `data/README.md`; a few fixture images kept for tests; git history not rewritten |
| Model | Pretrained torchvision backbone (main) + modernized IncludeNet baseline (lightweight) |
| Cell cropping | Optional (`preprocessing.crop`), off by default; rewritten, tested module |
| Training compute | Code built and tested on CPU here; full training run by the owner on GPU/Colab |
| Training stack | Plain PyTorch loop, YAML configs validated with pydantic, Typer CLI (not Lightning/Hydra) |
| Docs | MkDocs + Material + mkdocstrings, deployed to GitHub Pages |
| Releases | Conventional Commits + python-semantic-release on merge to `main`; PyPI publish wired but off by default |
| Default branch | Rename `master` → `main` (GitHub-side step done by the owner) |

## 3. Repository layout

```
wbc-classification/
├── src/wbc_classification/
│   ├── config.py            # pydantic config models + YAML loading + CLI overrides
│   ├── data/                # datasets.py, transforms.py, labels.py
│   ├── preprocessing/       # segmentation.py (optional HSV cell crop)
│   ├── models/              # baseline.py (IncludeNet), backbone.py, registry.py
│   ├── engine/              # train.py, evaluate.py, predict.py, metrics.py
│   └── cli.py               # `wbc` Typer app
├── configs/                 # baseline.yaml, resnet18.yaml, efficientnet_b0.yaml
├── data/README.md           # layout + download instructions (data itself is git-ignored)
├── scripts/                 # thin helpers only if not covered by the CLI
├── tests/                   # unit, CLI, CPU smoke test; fixtures in tests/fixtures/
├── notebooks/               # Colab training notebook, WBC-Detection walkthrough
├── docs/                    # MkDocs site sources
├── .github/workflows/       # ci.yml, release.yml, docs.yml, pr-title.yml
├── pyproject.toml, poetry.lock, mkdocs.yml
└── README.md, CONTRIBUTING.md, CHANGELOG.md, LICENSE
```

**Removed:** Keras code (`CNN/`, `learning.py`, `load_test_model.py`), `requirements.txt`, `_config.yml`, stub READMEs, PayPal/hit-counter badges, `TrainedModel/*.hdf5`, committed dataset images. `WBC-Detection/` content moves into `docs/` and a notebook; its sample image becomes a test fixture.

**Packaging:** `[tool.poetry]` with `packages = [{include = "wbc_classification", from = "src"}]`, console script `wbc = "wbc_classification.cli:app"`. Dependency groups:

- main: torch, torchvision, numpy, opencv-python-headless, pillow, scikit-learn, matplotlib, typer, pydantic, pyyaml
- dev: pytest, ruff, mypy
- docs: mkdocs, mkdocs-material, mkdocstrings
- release: python-semantic-release

Plain PyPI torch is declared; the training guide documents installing a CUDA build for GPU users. OpenCV 5 compatibility is verified during implementation (fall back to 4.x only if needed; the new code uses the two-value `findContours` return).

## 4. Models

- `build_model(name, num_classes, pretrained)` registry in `models/registry.py`; names: `baseline`, `resnet18`, `resnet50`, `efficientnet_b0`. Adding a model = adding a registry entry.
- `baseline`: IncludeNet in PyTorch — Conv-BatchNorm-ReLU blocks, global average pooling, dropout; independent of input size (the old hard-coded 50×50 flatten is gone).
- Backbones: torchvision pretrained weights, classifier head replaced; optional `freeze_backbone` for the first N epochs, then unfreeze.

## 5. Data and preprocessing

- Dataset reads `TRAIN/<CLASS>/` and `TEST/<CLASS>/` (Kaggle layout) under `data/`. Class names derive from folder names; `labels.py` is the single source of truth.
- Validation: seeded, stratified split from TRAIN. TEST is used only by `evaluate`.
- Train transforms: resize, flips/rotations, color jitter (stain variation), ImageNet normalization. Eval transforms: resize + normalize. Image size is per-config (50 for baseline; 128–224 for backbones).
- Optional crop (`preprocessing.crop: true`): HSV threshold on a correctly converted (BGR→HSV) image, largest contour, bounding crop with clamped bounds. No cell found → `SegmentationError`; the training pipeline catches it, falls back to the full image, and logs a warning.
- Crop vs. no-crop is compared in the docs results so the default is evidence-based.

## 6. Training, evaluation, inference

- **Training loop** (`engine/train.py`): AdamW, cosine LR schedule, mixed precision on CUDA, early stopping on validation accuracy. Seed controls Python/NumPy/torch.
- **Run artifacts:** `runs/<name>-<timestamp>/` containing config copy, `metrics.csv`, training-curve PNG, confusion matrix, best and last checkpoints.
- **Checkpoint:** a single `.pt` holding weights, model name, class names, and full config; `predict` rebuilds the model from the file alone. Loading validates stored config/class names against the model.
- **Evaluation:** per-class precision/recall/F1 and confusion matrix on TEST, as console output and JSON.
- **Inference:** `wbc predict <image|dir> --checkpoint X` prints class and probabilities; optional annotated-image output. No `cv2.imshow`.
- **Config:** one YAML per experiment (model, image size, epochs, batch size, LR, augmentation, crop flag, seed); every field overridable from the CLI (`wbc train -c configs/resnet18.yaml --epochs 5`). Invalid configs fail fast with clear messages.

## 7. CLI

`wbc train`, `wbc evaluate`, `wbc predict`, `wbc segment`, `wbc download-data`. Nothing runs at import time.

## 8. Documentation

MkDocs Material site: getting started, data setup, training guide (incl. GPU/Colab), configuration reference, model zoo/results, CLI reference (generated from Typer), API reference (mkdocstrings), architecture, WBC-Detection walkthrough, contributing, changelog (included from `CHANGELOG.md`). `mkdocs build --strict` runs in CI; deployed to GitHub Pages on merge to `main`.

The README is short: self-updating badges (CI, release, Python versions, license, docs), quick start, link to docs. Results table lives in the docs.

## 9. CI and releases

- `ci.yml` (PRs and pushes): ruff, mypy, pytest (3.12/3.13/3.14), docs build, CPU smoke test.
- `pr-title.yml`: enforces Conventional Commit PR titles; squash-merge so the title becomes the commit.
- `release.yml` (merge to `main`): python-semantic-release bumps the version in `pyproject.toml`, regenerates `CHANGELOG.md`, tags `vX.Y.Z`, creates the GitHub Release, builds with `poetry build`, attaches artifacts. `docs:`/`chore:`/`ci:` commits do not release. PyPI publishing is configured but disabled until the owner registers the project and adds a token.
- `docs.yml` (merge to `main`): deploys the site.
- First release: `0.1.0`. GitHub Actions use current major versions, verified at write time.

## 10. Testing

- Unit: transforms, label mapping, config validation (bad configs fail clearly), model output shapes for every registry entry, segmentation on fixtures (including the no-cell case), metrics.
- CLI: Typer `CliRunner` for each command.
- Smoke (CPU, under a minute): 1-epoch train of `baseline` on ~20 fixture images → evaluate → save → reload → predict.
- Marked `slow` (not in CI): full training, pretrained-weight downloads.

## 11. Error handling

Validate at boundaries only (config load, dataset discovery, checkpoint load) with actionable messages (e.g. "data dir missing — run `wbc download-data`"). No silent `print` + `return None`. Internal code trusts validated inputs.

## 12. Migration order

1. Housekeeping: branch rename to `main` (owner), remove dataset images from the tree, add `data/README.md` + download command, keep fixtures.
2. Scaffold: Poetry `pyproject.toml`, `src/` layout, ruff/mypy/pytest config, CI skeleton.
3. Core: config, labels, data, transforms, then models and registry.
4. Engine and CLI with tests alongside.
5. Docs: MkDocs site, README, CONTRIBUTING, Colab notebook.
6. Release automation workflows.
7. Cleanup: remove old Keras code and leftover files.

## 13. Risks and open items

- OpenCV 5.0 is a new major version; may need a fallback to 4.x.
- Python 3.14 + torch wheels exist today (cp314); if a transitive dependency lacks 3.14 support the dev version drops to 3.13.
- No pretrained checkpoint or metrics are published until the owner runs GPU training; docs ship with a results template.
- Kaggle download needs the owner's Kaggle credentials; the command documents this and fails with a clear message otherwise.
- Removing images from the tree does not shrink existing history (clone size stays large) — a separate, explicit decision.
