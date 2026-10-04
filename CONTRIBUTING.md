# Contributing

Thanks for helping! This project uses [Poetry](https://python-poetry.org/) and Python 3.12+.

## Setup

```bash
make install-docs   # = poetry install --with dev,docs --extras kaggle
```

Run `make help` to list every shortcut (`make check`, `make format`, `make docs`, `make train`, ...).

## Checks (all run in CI)

`make check` runs lint, type-check and tests. `make docs` builds the docs site. Without `make`:

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

## Releases

Every push to `main` runs the `Release` workflow (python-semantic-release). It computes the next
version from the Conventional Commits, updates `project.version` in `pyproject.toml` and
`CHANGELOG.md`, tags `vX.Y.Z` and creates a GitHub Release with the built wheel and sdist.

- The workflow pushes the version-bump commit and tag to `main`, so branch protection must allow
  `github-actions[bot]` to push (or use a PAT / GitHub App token instead of `GITHUB_TOKEN`).
- PyPI publishing is optional: register the project on PyPI with a trusted publisher for this
  repository and the `release.yml` workflow, then set the repository variable `PUBLISH_PYPI` to `true`.
