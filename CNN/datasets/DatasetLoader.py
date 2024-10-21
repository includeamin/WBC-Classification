import numpy as np
import cv2
import os


# datasets_name/class/image.jpg
class DatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, image_paths, verbose=-1):
        data = []
        labels = []
        for i, image_paths in enumerate(image_paths):
            # print(i)
            # try:
            # print(image_paths)
            # raise Exception(image_paths)
            image = cv2.imread(image_paths)
            # print(image_paths.split(os.path.sep))
            label = image_paths.split(os.path.sep)[-2]
            # print(label)
            # print(self.preprocessors)
            # print(image.shape)
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)
            # print("here")
            data.append(image)
            labels.append(label)
            # print("[INFO] processed {}/{}".format(i + 1, len(image_paths)))
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(image_paths)))
        #
        # except Exception as identifier:
        #     print(identifier)

        print(np.array(data), np.array(labels))

        return np.array(data), np.array(labels)
