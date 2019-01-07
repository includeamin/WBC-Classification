import cv2
import matplotlib
from matplotlib import colors
from matplotlib import pyplot as plt
import numpy as np
import argparse
import os
from imutils import paths


class SimplePreprocessor:
    def __init__(self,width,height,inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter

        
    def preprocess(self,im):
        try:
            # if in train time with 100 epoch and size 50 get bad result remove all this line und just resize , convert all dataset befor train network
                #im = cv2.imread(imagePaths)
                #Type = imagePaths.split(os.path.sep)[-2]
                image_blur = cv2.GaussianBlur(im, (7, 7), 0)
                image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)
                min_red = np.array([80, 60, 140])
                max_red = np.array([255, 255, 255])
                image_red1 = cv2.inRange(image_blur_hsv, min_red, max_red)
                big_contour, mask = self.find_biggest_contour(image_red1)
                (x,y),radius = cv2.minEnclosingCircle(big_contour)
                center = (int(x),int(y))
                radius = int(radius)
                imCircle = im.copy()
                cv2.circle(imCircle,center,radius,(0,255,0),1)
                height, width, channels = imCircle.shape
                extera = 0
                border= [0,0,0,0]
                if center[0] + radius > width :
                     extera = (center[0] + radius) - width
                     border[3] = extera+1
   
                if(center[0]- radius < 0) :
                     extera =  width - (center[0] + radius)
                     border[2] = extera+1
   
                if center[1]+ radius > height :
                     extera = (center[1] + radius) - height
                     border[1] = extera+1
   
                if center[1] + radius < 0 :
                     extera =  height - (center[1] + radius)
                     border[0]=extera+1
                borderIm = im.copy()
                borderIm = cv2.copyMakeBorder(im,border[0],border[1],border[2],border[3],cv2.BORDER_CONSTANT)
                y = center[1]-radius
                if y <0:
                    y=0
                y2 =center[1]+radius
                x = center[0]-radius
                if x<0 :
                    x=0

                x2 =center[0]+radius
                
                cropedImage = im[y:y2,x:x2]
               
                r = float(self.width) / cropedImage.shape[1]
                
                dim = (self.width, int(cropedImage.shape[0] * r))
                # resized = cv2.resize(cropedImage, dim, interpolation = cv2.INTER_AREA)
                # return resized

            #return self.ConvertToCnnInput(im,self.width)
                final = cv2.cvtColor(cropedImage, cv2.COLOR_BGR2GRAY)
                return cv2.resize(cropedImage,(self.width,self.height),
            interpolation=self.inter)
       
        except Exception as a:
           print ("preprocessor",a)   
    def find_biggest_contour(self,image):
         image = image.copy()
         s, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
         biggest_contour = max(contours, key=cv2.contourArea)
         mask = np.zeros(image.shape, np.uint8)
         cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
         return biggest_contour, mask

    def overlay_mask(self,mask, image):
         rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
         img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
         #show(img)
    