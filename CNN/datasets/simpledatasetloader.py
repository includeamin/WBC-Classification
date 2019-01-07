import numpy as np 
import cv2
import os
#datasets_name/class/image.jpg
class SimpleDatasetLoader:
    def __init__(self,preprocessors=None):
        self.preprocessors = preprocessors
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self,imagePaths , verbose=-1):
        data = []
        labels = []
        for (i , imagePaths) in enumerate(imagePaths):
            try:
                pass
                image = cv2.imread(imagePaths)
                label = imagePaths.split(os.path.sep)[-2]
                if self.preprocessors is not None:
                    for p in self.preprocessors:
                        image = p.preprocess(image)
                data.append(image)
                labels.append(label)
                if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                    print("[INFO] processed {}/{}".format(i+1,len(imagePaths)))
                
            except Exception as identifier:
                print(identifier)
           
        return (np.array(data),np.array(labels))
   


         