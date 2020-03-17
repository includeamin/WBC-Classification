from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from Utils.ImageTools import ImageToArrayPreprocessor
from PrePorcessor.Preprocessor import SimplePreprocessor
from dataset.SimpleDatasetLoader import SimpleDatasetLoader
from Model.IncludeNet import IncludeNet
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import argparse
from imutils import paths

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="halloo insert dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())
size = 50
ep = 150
dpt = 3

print("[INFO] loading Images")
imagePaths = list(paths.list_images(args["dataset"]))
sp = SimplePreprocessor(size, size)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
print(labels)
data = data.astype("float") / 255.0
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] compiling model...")
opt = SGD(lr=0.025)
model = IncludeNet.build(width=size, height=size, depth=dpt, classes=4)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=size, epochs=ep, verbose=1)
print("Saving network")
model.save(f'./SavedModel/{args["model"]}')
print("Network have been saved")

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=size)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]))
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, ep), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, ep), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, ep), H.history["accuracy"], label="accuracy")
plt.plot(np.arange(0, ep), H.history["val_accuracy"], label="val_acc")
plt.title("AMINJAMAL")
plt.xlabel("Epoch #")
plt.ylabel("Loss/ACC")
plt.legend()
plt.show()
