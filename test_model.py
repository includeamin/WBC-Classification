from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import load_model
from Utils.ImageTools import ImageToArrayPreprocessor
from PrePorcessor.Preprocessor import SimplePreprocessor
from dataset.SimpleDatasetLoader import SimpleDatasetLoader
from keras.optimizers import SGD
from Model.IncludeNet import IncludeNet
import matplotlib.pyplot as plt
import numpy as np
import argparse
from imutils import paths
import cv2


def show(image):
    # Figure size in inches
    plt.figure(figsize=(15, 15))

    # Show image, with nearest neighbour interpolation
    plt.imshow(image, interpolation='nearest')
    plt.show()


def ReadyToUseImage(im):
    image_blur = cv2.GaussianBlur(im, (7, 7), 0)
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)
    min_red = np.array([80, 60, 140])
    max_red = np.array([255, 255, 255])
    image_red1 = cv2.inRange(image_blur_hsv, min_red, max_red)
    big_contour, mask = find_biggest_contour(image_red1)
    overlay_mask(mask, im)
    moments = cv2.moments(mask)
    centre_of_mass = (
        int(moments['m10'] / moments['m00']),
        int(moments['m01'] / moments['m00'])
    )
    image_with_com = im.copy()
    # cv2.rectangle(image_with_com,10,(0,255,0),-1,cv2.LINE_AA)
    cv2.circle(image_with_com, centre_of_mass, 10, (0, 255, 0), -1, cv2.LINE_AA)
    # show(image_with_com)
    image_with_ellipse = im.copy()
    ellipse = cv2.fitEllipse(big_contour)
    # print(centre_of_mass)
    dst = cv2.bitwise_and(im, im, mask=mask)

    return (dst)


def find_biggest_contour(image):
    image = image.copy()
    s, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    biggest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
    return biggest_contour, mask


def overlay_mask(mask, image):
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    # show(img)


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="halloo insert dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())
size = 50
classLabels = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]
print("[INFO] sampling images ...")
imagePaths = np.array(list(paths.list_images(args["dataset"])))
idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]
sp = SimplePreprocessor(size, size)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths)
data = data.astype("float") / 255.0
print("[INFO] loading pre-trained network ...")
model = load_model(args["model"])
print("[INFO] predicting ...")
preds = model.predict(data, batch_size=size).argmax(axis=1)
print(preds)
for (i, imagePath) in enumerate(imagePaths):
    # load the example image, draw the prediction, and display it
    # to our screen
    image = cv2.imread(imagePath)
    # image=ReadyToUseImage(image)
    cv2.putText(image, "Label: {}".format(classLabels[preds[i]]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
