from keras.api.models import Sequential
from keras.api.layers import (
    Conv2D,
    Activation,
    Dense,
    Dropout,
    MaxPooling2D,
    BatchNormalization,
    GlobalAveragePooling2D,
)
from keras.api.regularizers import l2
from keras import backend as K


class IncludeNet:
    @staticmethod
    def build(width, height, depth, classes=4, size=32, reg=0.001):
        # Default `classes=4` for your use case, added `reg` parameter for L2 regularization
        model = Sequential()
        input_shape = (height, width, depth)
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)

        # Conv Block 1 with L2 Regularization
        model.add(
            Conv2D(
                size,
                (3, 3),
                padding="same",
                input_shape=input_shape,
                kernel_regularizer=l2(reg),
            )
        )
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Conv Block 2 with L2 Regularization
        model.add(Conv2D(size * 2, (3, 3), padding="same", kernel_regularizer=l2(reg)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Conv Block 3 with L2 Regularization
        model.add(Conv2D(size * 4, (3, 3), padding="same", kernel_regularizer=l2(reg)))
        model.add(BatchNormalization())
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))  # Consider experimenting with dropout rate

        # Adding GlobalAveragePooling2D layer
        model.add(GlobalAveragePooling2D())

        # Dense Layer with L2 Regularization
        model.add(Dense(size * 8, kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))  # Consider experimenting with dropout rate

        # Output Layer
        model.add(
            Dense(classes, kernel_regularizer=l2(reg))
        )  # Adding L2 regularization here as well
        model.add(Activation("softmax"))
        return model
