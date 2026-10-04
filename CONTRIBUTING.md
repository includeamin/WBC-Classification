# Contributing

Thanks for helping! This project uses [Poetry](https://python-poetry.org/) and Python 3.12+.

## Setup

```bash
poetry install --with dev,docs --extras kaggle
```

## Checks (all run in CI)

```bash
poetry run ruff check . && poetry run ruff format --check .
poetry run mypy src
poetry run pytest
poetry run typer wbc_classification.cli utils docs --name wbc --output docs/reference/cli.md
poetry run mkdocs build --strict
```

Tests use the small fixture dataset in `tests/fixtures/`; they never download data or pretrained weights.
Mark long-running or network tests with `@pytest.mark.slow` (they are skipped by default).

## Commit messages and PR titles

Releases are automated from [Conventional Commits](https://www.conventionalcommits.org/). PRs are
squash-merged, so the **PR title** must follow the format:

| Prefix | Effect |
|---|---|
| `feat:` | minor release |
| `fix:`, `perf:` | patch release |
| `feat!:` or a `BREAKING CHANGE:` footer | major release (minor while version is 0.x) |
| `docs:`, `chore:`, `ci:`, `test:`, `refactor:` | no release |

Do not edit `CHANGELOG.md` or the version in `pyproject.toml` by hand; the release workflow does it on merge to `main`.

## Adding a model

Add an entry to `models/registry.py`, a config in `configs/`, and a shape test in `tests/test_models.py`.
