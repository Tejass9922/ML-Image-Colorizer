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
    def __init__(self, row, col, grayscale_value, average_grayscale_value, r_value, g_value, b_value,patch_data):
        self.row = row
        self.col = col
        self.grayscale_value = grayscale_value
        self.average_grayscale_value = average_grayscale_value
        self.r_value = r_value
        self.g_value = g_value
        self.b_value = b_value
        self.rgb = (r_value,g_value,b_value)
        self.patch_data = patch_data

    def __eq__(self, other):
        for i in range(3):
            for j in range(3):
                pixel_1 = self.patch_data[i][j]
                pixel_2 = other.patch_data[i][j]
                if not pixel_1 == pixel_2:
                    return False

        return True
    def __cmp__(self,other):
        patch_A = self.patch_data
        patch_B = self.patch_data

        return 
    def euclidian(self,other):
        a = self.patch_data
        b = other.patch_data
        dist = np.linalg.norm(a-b)
        return(dist)

def binarySearch(arr, x):
    low = 0
    high = len(arr) - 1
 
    while low <= high:
        mid = low + (high - low) // 2
        if arr[mid] < x:
            low = mid + 1
        elif arr[mid] > x:
            high = mid - 1
        else:
            return mid      # key found
 
    return low              # key not found
 
 
# Function to find the `k` closest elements to `x` in a sorted integer array `arr`
def findKClosestElements(arr, x, k):
 
    # find the insertion point using the binary search algorithm
    i = binarySearch(arr, x)
 
    left = i - 1
    right = i
 
    # run `k` times
    while k > 0:
 
        # compare the elements on both sides of the insertion point `i`
        # to get the first `k` closest elements
 
        if left < 0 or (right < len(arr) and abs(arr[left] - x) > abs(arr[right] - x)):
            right = right + 1
        else:
            left = left - 1
 
        k = k - 1
 
    # return `k` closest elements
    return arr[left+1: right]
'''
arr = [1,2,5,13,18]
x = 2
k = 3
k_closest = findKClosestElements(arr,x,k)
print(k_closest)
'''
'''
arr = [1,2,3,4,45]
arr = arr[0:1]
print(arr)
'''
a1 = [[0,1,4],[3,6,18],[14,5,7]]
a2 = [[19,35,7],[0,14,8],[40,6,7]]
arr = np.array([a2,a1])
print(arr[0]< arr[1])
arr.sort(axis=0)
print(arr)