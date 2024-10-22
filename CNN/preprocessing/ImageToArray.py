from keras.api.preprocessing.image import img_to_array


class ImageToArrayPreprocessor:
    def __init__(self, data_format=None):
        self.dataFormat = data_format

    def preprocess(self, image):
        # print(image)
        return img_to_array(image, data_format=self.dataFormat)
