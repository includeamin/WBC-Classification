import numpy as np
import matplotlib.pyplot as plt
import argparse
from imutils import paths
from keras.src.callbacks import EarlyStopping
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.api.optimizers import SGD
from keras.api.callbacks import LearningRateScheduler
from keras.api.utils import to_categorical

from CNN.nn.conv.IncludeNet import IncludeNet
from CNN.preprocessing import PreProcessor
from CNN.datasets import DatasetLoader
from CNN.preprocessing.ImageToArray import ImageToArrayPreprocessor


def step_decay(epoch):
    """Learning rate scheduler function."""
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * (drop ** np.floor((1 + epoch) / epochs_drop))
    return lrate


# Argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to the dataset")
ap.add_argument("-m", "--model", required=True, help="Path to output model")
args = vars(ap.parse_args())

# Parameters
size = 50
epochs = 70
depth = 3
num_classes = 4

# Load and preprocess dataset
print("[INFO] loading and preprocessing data...")
image_paths = list(paths.list_images(args["dataset"]))
sp = PreProcessor.PreProcessor(size, size)
iap = ImageToArrayPreprocessor()
sdl = DatasetLoader.DatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(image_paths, verbose=500)
data = data.astype("float") / 255.0

# Encode labels from strings to integers
le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, num_classes=num_classes)

(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.25, random_state=42
)

# Data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
)

model = IncludeNet.build(width=size, height=size, depth=depth, classes=4)
# Compile model
print("[INFO] compiling model...")
opt = SGD(learning_rate=0.01, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Callbacks

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

callbacks = [LearningRateScheduler(step_decay), early_stopping]

# Train model
print("[INFO] training network...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=size),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // size,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1,
)

# Save model
print("[INFO] serializing network...")
model.save(args["model"])

# Evaluate model
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=size)
print(
    classification_report(
        testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_
    )
)  # Use the classes_ attribute from LabelEncoder

# Plot training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
