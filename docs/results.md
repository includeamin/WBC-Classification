# Results

!!! note
    No pretrained checkpoints or benchmark numbers are published yet. Train a model with
    `wbc train` and `wbc evaluate`, then fill in the table below.

!!! warning "Validation split caveat"
    The Kaggle `TRAIN` folder contains augmented copies of a smaller set of original images, so the
    stratified validation split drawn from `TRAIN` can contain near-duplicates of training images and
    validation accuracy may look optimistic. The held-out `TEST` accuracy from `wbc evaluate` is the
    number to report.

| Model | Image size | Crop | Epochs | TEST accuracy | Macro F1 | Notes |
|---|---|---|---|---|---|---|
| baseline | 64 | no | – | – | – | – |
| resnet18 | 128 | no | – | – | – | – |
| resnet18 | 128 | yes | – | – | – | crop vs no-crop comparison |
| efficientnet_b0 | 160 | no | – | – | – | – |
