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

#src = str('scenery_training.jpg')
#img_object = Image(src)

"""
cv2.imshow('Original image',img_object.colored_image)

cv2.imshow('Gray image', img_object.gray_iamge)
cv2.imshow('Left Colored', img_object.left_c)
cv2.imshow('Righ Colored', img_object.right_c)
cv2.imshow('Left Gray', img_object.left_g)
cv2.imshow('Right Gray', img_object.right_g)
  
cv2.waitKey(0)
cv2.destroyAllWindows()

"""

c1 = (33, 71, 45)
c2 = (88, 157, 161)
c3 = (99, 111, 87)
c4 = (191, 176, 156)
c5 = (232, 232, 224)

repColors = []
repColors.append(c1)
repColors.append(c2)
repColors.append(c3)
repColors.append(c4)
repColors.append(c5)
#left_original = Image('training_image.png').left_c
#cv2.imwrite('Left_original.png',left_original)
#neural_network_right_half = Image('completed_nerual_network.png')
#cv2.imshow('right-colored',neural_network_right_half.right_c)
#cv2.imwrite('Recolored_Right_Neural_Network.png',neural_network_right_half.right_c)
#cv2.imwrite('completed_neural_network_right_half.png',neural_network_right_half)
#cv2.hconcat()