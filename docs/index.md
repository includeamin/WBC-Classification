IncludeNet

### download model
- Download [Dataset](https://www.kaggle.com/paultimothymooney/blood-cells/kernels?sortBy=relevance&group=everyone&search=includeamin&page=1&pageSize=20&datasetId=9232)
- Copy TRAIN folder to dataset directory
### train and save trained model
Use this command to train the model and save model
```shell script
python3 learning.py -d dataset/TRAIN -m sample.hdf5
```


### test model
- copy TEST forlder from downloaded dataset to dataset directory
- run this command
```shell script
python3 Test/load_test_model.py -d dataset/TEST -m SavedModel/150_epoch_model.hdf5
```