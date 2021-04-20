import numpy as np
import pandas as pa
from random import randint
from cv2 import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from dataclasses import dataclass 
from scipy import misc

def split_image(preprocessed_image):
   
    # Read the image
    img = misc.imageio.imread(preprocessed_image)
    height, width = img.shape

    # Cut the image in half
    width_cutoff = width // 2
    s1 = img[:, :width_cutoff]
    s2 = img[:, width_cutoff:]

    # Save each half
    #misc.imsave("face1.png", s1)
    #misc.imsave("face2.png", s2)

    return (s1,s2)
def ReadandSplit(picture_link):
    image = cv2.imread(picture_link)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    half = int(len(image[0])/2)
   
    left = img[:,:half]
    right = img[:,half:]

   

    return img

def convert_to_grayscale(image):
    '''
    for row in range(len(image)):
        for col in range(len(image)):
            r = image[row][col][0]
            gray_image[row][col] = 
    '''
    grayscale_image = cv2.imread(image)
    gray_image = cv2.cvtColor(grayscale_image, cv2.COLOR_BGR2GRAY)
    for pixel in gray_image:
        print(pixel)
    return gray_image
'''
image = convert_to_grayscale('training_image.png') 

left.right = ReadandSplit(gray_training_image)

#img = mpimg.imread('training_image2.png')
right = convert_to_grayscale(right)

imgplot = plt.imshow(image)

plt.show()
'''
image = cv2.imread('training_image.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray)
cv2.imshow('Original image',image)
cv2.imshow('Gray image', gray)
  
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
- k - means 5 cluster: 
    - pick 5 random points
    - run loop






'''