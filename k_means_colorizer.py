import math
import sys
import numpy as np
import pandas as pa
from random import randint
from queue import PriorityQueue
from cv2 import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from dataclasses import dataclass 
from scipy import misc
from preprocessor import *
from k_means_model import *

class Patch:
    def __init__(self, row, col, grayscale_value, average_grayscale_value, r_value, g_value, b_value):
        self.row = row
        self.col = col
        self.grayscale_value = grayscale_value
        self.average_grayscale_value = average_grayscale_value
        self.r_value = r_value
        self.g_value = g_value
        self.b_value = b_value

def get_neighbor_values(training_image, row, col):
    rowIndices = [-1, -1, -1, 0, 0, 1, 1, 1]
    colIndices = [-1, 0, 1, -1, 1, -1, 0, 1]
    neighborValues  = []
    for i in range(8):
        x = rowIndices[i] + row
        y = colIndices[i]  + col
        if x >= 0 and x < len(training_image) and y >= 0 and y < len(training_image[0]):
            neighborValues.append(training_image[x][y])
    return neighborValues

def get_six_similar(arr, x, k, n):
    six_similar_patches = []
    pq = PriorityQueue()
    for i in range(k):
        pq.put((-abs(arr[i].average_grayscale_value-x),i))
    for i in range(k,n):
        diff = abs(arr[i].average_grayscale_value-x)
        p,pi = pq.get()
        curr = -p
        if diff>curr:
            pq.put((-curr,pi))
            continue
        else:
            pq.put((-diff,i))
    while(not pq.empty()):
        p,q = pq.get()
        six_similar_patches.append(arr[q])
    return six_similar_patches

def train_model(training_image):
    print("--TRAINING K-MEANS MODEL--")
    representative_colors = k_means(training_image.colored_image)
    print("--FINISHED TRAINING MODEL--")
    print(representative_colors)
    return representative_colors

def color_left_half(training_image,representative_colors):
    print("\n--RECOLORING LEFT HALF--")
    RecolorLeft = training_image.left_c

    repColors = representative_colors

    for row in range(len(RecolorLeft)):
        for col in range(len(RecolorLeft[0])):
            pixel = RecolorLeft[row][col]
            min_diff = 766
            for c in repColors: 

                #print(row, col)     
                r_diff = euclidian(c[0],pixel[0])
                g_diff = euclidian(c[1],pixel[1])
                b_diff = euclidian(c[2],pixel[2])
                total_diff = r_diff + b_diff + g_diff
                chosen_color = c if total_diff < min_diff else chosen_color
                min_diff = min(total_diff,min_diff)
            RecolorLeft[row][col][0] = chosen_color[0]
            RecolorLeft[row][col][1] = chosen_color[1]
            RecolorLeft[row][col][2] = chosen_color[2]

    print("\n--FINISHED COLORING LEFT HALF--\n")
    return RecolorLeft

def color_right_half(training_image, RecolorLeft):
    print("\n--RECOLORING RIGHT HALF--")
    recolor_right = training_image.right_g
    grayscale_left = training_image.left_g

    left_patch_data = []
    for rowG in range(1, len(grayscale_left)-1, 1):
        for colG in range(1, len(grayscale_left[0])-1, 1):
            neighbor_values = get_neighbor_values(grayscale_left, rowG, colG)
            neighbor_values.append(grayscale_left[rowG][colG])
            average_grayscale_value = mean(neighbor_values)
            left_patch_data.append(Patch(rowG, colG, None, average_grayscale_value, None, None, None))


    for row in range(1, len(recolor_right)-1, 1):
        for col in range(1, len(recolor_right[0])-1, 1):
            six_similar_patches = get_six_similar(left_patch_data, recolor_right[row][col], 6, len(left_patch_data))
            six_similar_patches[5]
            six_similar_patches[4]
            six_similar_patches[3]
            six_similar_patches[2]
            six_similar_patches[1]
            six_similar_patches[0]

    print("\n--FINISHED COLORING RIGHT HALF--\n")
    return recolor_right

training_image = Image('scenery_training.png')
cv2.imshow('Left Colored - Old', training_image.left_c)
cv2.waitKey(0)
representative_colors = train_model(training_image)

# Sample Colors for training_image.png
#representative_colors = [(7.235440566268001, 4.158799121308275, 9.24460824993898), (73.20919616968587, 42.89522142305408, 13.874427956480055), (31.297164913342918, 18.522225081359267, 7.051370619844093), (194.67783309425366, 144.16915590808088, 155.61758501392288), (144.34301092067267, 98.40030035629513, 98.54573126174841)]

#Sample Colors for scenery_image.png
#representative_colors = [(50.06253367689082, 55.07547677539461, 29.799080158022825), (151.63715188952688, 133.68479106819527, 108.21791481071686), (210.91933997171725, 210.22021263344544, 200.62026099253364), (25.178416455599383, 87.07011737492873, 61.44349219299678), (35.477213480856264, 122.63336821860113, 108.12674709383784)]

RecolorLeft = color_left_half(training_image,representative_colors)
cv2.imshow('Left Colored - New', RecolorLeft)
cv2.waitKey(0)
cv2.destroyAllWindows()
