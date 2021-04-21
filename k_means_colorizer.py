import numpy as np
import pandas as pa
from random import randint
from cv2 import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from dataclasses import dataclass 
from scipy import misc
from preprocessor import *
from k_means_model import *

training_image = Image('training_image.png')
#representative_colors = k_means(training_image.colored_image)
#print(representative_colors)

Recolor = training_image.left_c





for row in range(len(Recolor)):
    for col in range(len(Recolor[0])):
        pixel = Recolor[row][col]
        min_diff = 766
        for c in repColors: 

            print(row, col)     
            r_diff = euclidian(c[0],pixel[0])
            g_diff = euclidian(c[1],pixel[1])
            b_diff = euclidian(c[2],pixel[2])
            total_diff = r_diff + b_diff + g_diff
            chosen_color = c if total_diff < min_diff else chosen_color
            min_diff = min(total_diff,min_diff)
        Recolor[row][col][0] = chosen_color[0]
        Recolor[row][col][1] = chosen_color[1]
        Recolor[row][col][2] = chosen_color[2]


cv2.imshow('Left Colored - New', Recolor)

cv2.waitKey(0)
cv2.destroyAllWindows()