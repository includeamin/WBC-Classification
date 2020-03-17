# WBC-Classification
Classification of WBC ( White Blood Cells ) with CNN . (Convolutional Neural Network)

### Download model
- Download [Dataset](https://www.kaggle.com/paultimothymooney/blood-cells/kernels?sortBy=relevance&group=everyone&search=includeamin&page=1&pageSize=20&datasetId=9232)
- Copy TRAIN folder to dataset directory
### Train and save trained model
Use this command to train the model and save model
```shell script
python3 learning.py -d dataset/TRAIN -m sample.hdf5
```
after of train you will see result plot:
![Image of Yaktocat](train_result.png)

### Test the model
- copy TEST forlder from downloaded dataset to dataset directory
- run this command
```shell script
python3 Test/load_test_model.py -d dataset/TEST -m SavedModel/150_epoch_model.hdf5
```
# Todo
- [ ] Create demo api . [ upload and check the result realtime ]
