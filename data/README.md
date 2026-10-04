# Data

The dataset is **not** stored in git. It is the public Kaggle
[Blood Cell Images](https://www.kaggle.com/datasets/paultimothymooney/blood-cells) dataset
(four classes: EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL; 320×240 JPEG).

## Download

```bash
poetry install --extras kaggle
poetry run wbc download-data --dest data
```

If the download fails because Kaggle requires authentication, create an API token at
<https://www.kaggle.com/settings> and expose it as `KAGGLE_USERNAME` / `KAGGLE_KEY`
(or `~/.kaggle/kaggle.json`), then retry. You can also download and unzip the dataset
manually and copy its `TRAIN/` and `TEST/` folders here.

## Expected layout

```
data/
├── TRAIN/
│   ├── EOSINOPHIL/*.jpeg
│   ├── LYMPHOCYTE/*.jpeg
│   ├── MONOCYTE/*.jpeg
│   └── NEUTROPHIL/*.jpeg
└── TEST/            # same four class folders; used only by `wbc evaluate`
```

Class names are taken from the folder names. Files that are not `.jpg/.jpeg/.png` are ignored.
Everything in this folder except this README is git-ignored.
