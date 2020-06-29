from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from CNN.preprocessing import ImageToArray
from CNN.preprocessing.PreProcessor import PreProcessor
from CNN.datasets.DatasetLoader import DatasetLoader
from CNN.nn.conv import IncludeNet
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import argparse
from imutils import paths

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="insert dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())
size = 50
ep = 50
dpt = 3


class Learning:
    @staticmethod
    def learning():
        test_x, test_y, train_x, train_y = Learning.loading_dataset()

        model = Learning.compile()

        h = Learning.train(model, test_x, test_y, train_x, train_y)

        Learning.save_model(model)

        Learning.evaluating(h, model, test_x, test_y)

    @staticmethod
    def loading_dataset():
        print("[INFO] loading Images")
        image_paths = list(paths.list_images(args["dataset"]))
        sp = PreProcessor(size, size)
        iap = ImageToArray.ImageToArrayPreprocessor()
        sdl = DatasetLoader(preprocessors=[sp, iap])
        (data, labels) = sdl.load(image_paths, verbose=500)
        print(data)
        data = data.astype("float") / 255.0
        (train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.25, random_state=42)
        train_y = LabelBinarizer().fit_transform(train_y)
        test_y = LabelBinarizer().fit_transform(test_y)
        return test_x, test_y, train_x, train_y

    @staticmethod
    def compile():
        print("[INFO] compiling model...")
        opt = SGD(lr=0.025)
        model = IncludeNet.IncludeNet.build(width=size, height=size, depth=dpt, classes=4)
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        return model

    @staticmethod
    def train(model, test_x, test_y, train_x, train_y):
        print("[INFO] training network...")
        h = model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=size, epochs=ep, verbose=1)
        return h

    @staticmethod
    def save_model(model):
        print("Saving network")
        model.save(args["model"])
        print("Network have been saved")

    @staticmethod
    def evaluating(h, model, test_x, test_y):
        print("[INFO] evaluating network...")
        predictions = model.predict(test_x, batch_size=size)
        print(classification_report(test_y.argmax(axis=1),
                                    predictions.argmax(axis=1),
                                    target_names=["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]))
        Learning.show_plot(h)

    @staticmethod
    def show_plot(h):
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, ep), h.history["loss"], label="train_loss")
        plt.plot(np.arange(0, ep), h.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, ep), h.history["acc"], label="acc")
        plt.plot(np.arange(0, ep), h.history["val_acc"], label="val_acc")
        plt.title("AMINJAMAL")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/ACC")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    Learning.learning()
