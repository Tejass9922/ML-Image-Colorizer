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

def train_model(training_image):
    print("--TRAINING K-MEANS MODEL--")
    representative_colors = k_means(training_image.colored_image)
    print("--FINISHED TRAINING MODEL--")
    print(representative_colors)
    return representative_colors

def color_left_half(training_image,representative_colors):
    print("\n--RECOLORING LEFT HALF--")
    Recolor = training_image.left_c

    repColors = representative_colors

    for row in range(len(Recolor)):
        for col in range(len(Recolor[0])):
            pixel = Recolor[row][col]
            min_diff = 766
            for c in repColors: 

                #print(row, col)     
                r_diff = euclidian(c[0],pixel[0])
                g_diff = euclidian(c[1],pixel[1])
                b_diff = euclidian(c[2],pixel[2])
                total_diff = r_diff + b_diff + g_diff
                chosen_color = c if total_diff < min_diff else chosen_color
                min_diff = min(total_diff,min_diff)
            Recolor[row][col][0] = chosen_color[0]
            Recolor[row][col][1] = chosen_color[1]
            Recolor[row][col][2] = chosen_color[2]

    print("\n--FINISHED COLORING LEFT HALF--\n")
    return Recolor


training_image = Image('training_image.png')
cv2.imshow('Left Colored - Old', training_image.left_c)
cv2.waitKey(0)
representative_colors = train_model(training_image)

# Sample Colors for trainins_image.png
#representative_colors = [(7.235440566268001, 4.158799121308275, 9.24460824993898), (73.20919616968587, 42.89522142305408, 13.874427956480055), (31.297164913342918, 18.522225081359267, 7.051370619844093), (194.67783309425366, 144.16915590808088, 155.61758501392288), (144.34301092067267, 98.40030035629513, 98.54573126174841)]

#Sample Colors for scenery_image.png
#representative_colors = [(50.06253367689082, 55.07547677539461, 29.799080158022825), (151.63715188952688, 133.68479106819527, 108.21791481071686), (210.91933997171725, 210.22021263344544, 200.62026099253364), (25.178416455599383, 87.07011737492873, 61.44349219299678), (35.477213480856264, 122.63336821860113, 108.12674709383784)]

Recolor = color_left_half(training_image,representative_colors)
cv2.imshow('Left Colored - New', Recolor)
cv2.waitKey(0)
cv2.destroyAllWindows()