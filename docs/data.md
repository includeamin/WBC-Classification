# Data

The project uses the public Kaggle
[Blood Cell Images](https://www.kaggle.com/datasets/paultimothymooney/blood-cells) dataset
(320×240 JPEG, four classes). It is not stored in git.

```bash
poetry run wbc download-data --dest data          # add --force to overwrite
```

Expected layout:

    data/
    ├── TRAIN/{EOSINOPHIL,LYMPHOCYTE,MONOCYTE,NEUTROPHIL}/*.jpeg
    └── TEST/{EOSINOPHIL,LYMPHOCYTE,MONOCYTE,NEUTROPHIL}/*.jpeg

- Class names come from the folder names; non-image files are ignored.
- A stratified, seeded validation split is carved from `TRAIN`. `TEST` is only used by `wbc evaluate`.
- If Kaggle asks for authentication, create an API token and set `KAGGLE_USERNAME` / `KAGGLE_KEY`
  (or use `~/.kaggle/kaggle.json`). You can also download the dataset manually and copy `TRAIN/` and `TEST/` into `data/`.

!!! warning "Validation split caveat"
    The Kaggle `TRAIN` folder contains augmented copies of a smaller set of original images, so the
    stratified validation split drawn from `TRAIN` can contain near-duplicates of training images and
    validation accuracy may look optimistic. The held-out `TEST` accuracy from `wbc evaluate` is the
    number to report.
