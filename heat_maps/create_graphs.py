import math
import sys
import numpy as np
import pandas as pa
import random
from random import randint
from queue import PriorityQueue
from cv2 import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from dataclasses import dataclass 
from scipy import misc
from preprocessor import *
from k_means_model import *

import matplotlib.pyplot as plt
import seaborn as sns


def euclidian(pointA, pointB):
    return np.linalg.norm(((pointA)-(pointB)))

def create_heat_map(img1,img2):
    n = len(img1)
    m = len(img1[0])
    arr1 = np.zeros((n,m), dtype='uint8')
    differences = []
    for i in range(n):
        for j in range(m):
            pixel = img1[i][j]
            pixel2 = img2[i][j]

            r1 = int(pixel[0])
            r2 = int(pixel2[0])
            r_diff = abs(r1-r2)
            r_diff_p = abs(r2-r1)

            g1 = int(pixel[0])
            g2 = int(pixel2[0])
            g_diff = abs(r1-r2)
            g_diff_p = abs(r2-r1)

            b1 = int(pixel[0])
            b2 = int(pixel2[0])
            b_diff = abs(r1-r2)
            b_diff_p = abs(r2-r1)

           
            diff = r_diff + b_diff + g_diff
            arr1[i][j] = diff
            print(diff)
            differences.append(diff)
    
    pixels = n*m
    avg = float(sum(differences) / float(pixels))
    print(avg)
    map = sns.heatmap(arr1)
    plt.savefig('heat_map_original_basic.png')


original = Image('training_image.png')
original_right_half = original.right_c

basic_recolored = Image('completed_k_means_colorizer.png')

neural_net_recolored = Image('completed_neural_network.png').right_c



create_heat_map(original.right_c,basic_recolored.right_c)

'''
a = [1,15,14]
b = [23,43,109]

#print(euclidian(a,b))

print(a-b)'''