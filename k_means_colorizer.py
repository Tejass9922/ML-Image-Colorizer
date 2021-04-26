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
        self.rgb = (r_value,g_value,b_value)

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
    print("start")  
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
        
    print("end")
    return six_similar_patches

def binarySearch(arr, x):
    low = 0
    high = len(arr) - 1
 
    while low <= high:
        mid = low + (high - low) // 2
        if arr[mid].average_grayscale_value < x:
            low = mid + 1
        elif arr[mid].average_grayscale_value > x:
            high = mid - 1
        else:
            return mid      # key found
 
    return low              # key not found
 
 

def findKClosestElements(arr, x, k):
 
   
    i = binarySearch(arr, x)
 
    left = i - 1
    right = i
 
  
    while k > 0:
        if left < 0 or (right < len(arr) and abs(arr[left].average_grayscale_value - x) > abs(arr[right].average_grayscale_value - x)):
            right = right + 1
        else:
            left = left - 1
 
        k = k - 1
 
    # return `k` closest elements
    return arr[left+1: right]

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


def findMajorityElement(A):
 
    m = (-1,-1,-1)
    
    i = 0
 
    for j in range(len(A)):
      
        if i == 0:
            m = A[j]
            i = 1

        elif m.rgb == A[j].rgb:
            i = i + 1
        else:
            i = i - 1
 
    return m
def color_right_half(training_image, RecolorLeft):
    print("\n--RECOLORING RIGHT HALF--")
    grayscale_right = training_image.right_g
    grayscale_left = training_image.left_g
    RecolorRight = training_image.right_c
    left_patch_data = []
    for rowG in range(1, len(grayscale_left)-1, 1):
        for colG in range(1, len(grayscale_left[0])-1, 1):
            neighbor_values = get_neighbor_values(grayscale_left, rowG, colG)
            neighbor_values.append(grayscale_left[rowG][colG])
            average_grayscale_value = float(float(sum(neighbor_values)) / float(len(neighbor_values)))
            left_patch_data.append(Patch(rowG, colG, None, average_grayscale_value, None, None, None))
    left_patch_data.sort(key=lambda x: x.average_grayscale_value, reverse=True)
    print("--COMPLETED AVERAGING LEFT GRAYSCALE--")
    for row in range(1, len(grayscale_right)-1, 1):
        for col in range(1, len(grayscale_right[0])-1, 1):
            #Find average of the patch
            print((row,col))
            neighbor_values = get_neighbor_values(grayscale_right, row, col)
            neighbor_values.append(grayscale_left[row][col])
            average_grayscale_value = float(float(sum(neighbor_values)) / float(len(neighbor_values)))
            six_similar_patches = findKClosestElements(left_patch_data, average_grayscale_value, 6)
            
            for patch in six_similar_patches:
               
                rep_color = RecolorLeft[patch.row][patch.col]
                patch.r_value = rep_color[0]
                patch.g_value = rep_color[1]
                patch.b_value = rep_color[2]
                patch.rgb = (patch.r_value,patch.g_value,patch.b_value)

            #Use Boyer-Moore Voting algo to find majority (Might tweak this to make it super majority)
            
            majority_color = findMajorityElement(six_similar_patches)
            #print("M_color %d"-majority_color)
            if not  majority_color == (-1,-1,-1):
                RecolorRight[row][col] = majority_color.rgb
            else:
                #Assumption: 0th element of six_similar_patches = the most similar of the 6
                most_similar_patch = six_similar_patches[0]
                most_similar_patch_location = (most_similar_patch.row,most_similar_patch.col)
                RecolorRight[row][col] = RecolorLeft[most_similar_patch_location[0]][most_similar_patch_location[1]]
            
            
    print("\n--FINISHED COLORING RIGHT HALF--\n")
    return RecolorRight

training_image = Image('training_image.png')
cv2.imshow('Left Colored - Old', training_image.left_c)
cv2.imshow('Right Colored - Old',training_image.right_c)
cv2.waitKey(0)
#representative_colors = train_model(training_image)

# Sample Colors for training_image.png
representative_colors = [(7.235440566268001, 4.158799121308275, 9.24460824993898), (73.20919616968587, 42.89522142305408, 13.874427956480055), (31.297164913342918, 18.522225081359267, 7.051370619844093), (194.67783309425366, 144.16915590808088, 155.61758501392288), (144.34301092067267, 98.40030035629513, 98.54573126174841)]

#Sample Colors for scenery_image.png
#representative_colors = [(50.06253367689082, 55.07547677539461, 29.799080158022825), (151.63715188952688, 133.68479106819527, 108.21791481071686), (210.91933997171725, 210.22021263344544, 200.62026099253364), (25.178416455599383, 87.07011737492873, 61.44349219299678), (35.477213480856264, 122.63336821860113, 108.12674709383784)]

#RecolorLeft = color_left_half(training_image,representative_colors)
#cv2.imwrite('Recolored_training_left.png',RecolorLeft)
RecolorLeft = cv2.imread('Recolored_training_left.png')
cv2.imshow('Left Colored - New', RecolorLeft)
cv2.waitKey(0)

RecolorRight = color_right_half(training_image,RecolorLeft)
#RecolorFinal = cv2.vconcat(RecolorLeft,RecolorRight)
#cv2.imshow(training_image.colored_image)

cv2.imshow('Right Colored - New',RecolorRight)
#cv2.imshow('Final Colored - New',RecolorFinal)
cv2.waitKey(0)
cv2.destroyAllWindows()
