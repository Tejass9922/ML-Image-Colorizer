import numpy as np
import pandas as pa
from random import randint
from cv2 import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from dataclasses import dataclass 
from scipy import misc

class Image:
    
    def __init__(self,source):
        self.source = str(source)
        self.colored_image = self.read_colored_image()
        self.gray_iamge = self.convert_to_grayscale()
        self.left_c,self.right_c = self.split_image_c()
        self.left_g,self.right_g = self.split_image_g()
        


    def read_colored_image(self):
        return cv2.imread(self.source)
       # print(self.colored_image)

    def split_image_c(self):
        
        half = int(len(self.colored_image[0])/2)
    
        self.left_c =  self.colored_image[:,:half]
        self.right_c = self.colored_image[:,half:]

        return (self.left_c,self.right_c)

    def split_image_g(self):
        
        half = int(len(self.colored_image[0])/2)

        self.left_g =  self.gray_iamge[:,:half]
        self.right_g = self.gray_iamge[:,half:]

        return (self.left_g,self.right_g)

    def convert_to_grayscale(self):
      self.gray_iamge = cv2.cvtColor(self.colored_image, cv2.COLOR_BGR2GRAY)
      return self.gray_iamge

src = str('training_image.png')
img_object = Image(src)


cv2.imshow('Original image',img_object.colored_image)

cv2.imshow('Gray image', img_object.gray_iamge)
cv2.imshow('Left Colored', img_object.left_c)
cv2.imshow('Righ Colored', img_object.right_c)
cv2.imshow('Left Gray', img_object.left_g)
cv2.imshow('Right Gray', img_object.right_g)
  
cv2.waitKey(0)
cv2.destroyAllWindows()

