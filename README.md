# WBC-Classification
Classification of WBC ( White Blood Cells ) with CNN . (Convectional Neural Network)


[![HitCount](http://hits.dwyl.com/includeamin/WBC-Classification.svg)](http://hits.dwyl.com/includeamin/WBC-Classification)

### DATASET
-  [Dataset](https://www.kaggle.com/paultimothymooney/blood-cells/kernels?sortBy=relevance&group=everyone&search=includeamin&page=1&pageSize=20&datasetId=9232)
### Train and save trained model
Use this command to train the model and save model
```bash
git clone 
cd WBC-Classification
python3 learning.py -d ./CNN/datasets/TRAIN/ -m ./TrainedModel/model_epoch_100.hdf5
```
after train you will see result plot:
![](image.png)

### Test the model
```bash
python3 load_test_model.py -d ./CNN/datasets/TEST -m ./TrainedModel/model_epoch_100.hdf5
```

## NOTE
default epoch count is 100. for change it,just edit `cp` variable in [`earning.py`](learning.py)
# Todo
- more customizable