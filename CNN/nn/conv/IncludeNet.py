from keras.api.models import Sequential
from keras.api.layers import Conv2D
from keras.api.layers import Activation
from keras.api.layers import Dense, Dropout
from keras import backend as K
from keras.api.layers import MaxPooling2D
from keras.src.layers import BatchNormalization, GlobalAveragePooling2D



class IncludeNet:
    @staticmethod
    def build(
        width, height, depth, classes=4, size=32
    ):  # Default `classes=4` for your use case
        model = Sequential()
        input_shape = (height, width, depth)
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)

        # Conv Block 1
        model.add(Conv2D(size, (3, 3), padding="same", input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Conv Block 2
        model.add(Conv2D(size * 2, (3, 3), padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Conv Block 3
        model.add(Conv2D(size * 4, (3, 3), padding="same"))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        # Adding GlobalAveragePooling2D layer
        model.add(GlobalAveragePooling2D())

        # Dense Layer
        model.add(Dense(size * 8))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        # Output Layer
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model
