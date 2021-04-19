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
def ReadandSplit(picture):
    image = cv2.imread(picture)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    half = int(len(image[0])/2)
   
    left = img[:,:half]
    right = img[:,half:]

   

    return (left, right)

def convert_to_grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

left, right = ReadandSplit('training_image.png') 

#img = mpimg.imread('training_image2.png')
right = convert_to_grayscale(right)
imgplot = plt.imshow(right)

plt.show()
