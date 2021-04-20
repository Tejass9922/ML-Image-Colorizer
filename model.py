import numpy as np
import pandas as pa
from random import randint
from cv2 import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from dataclasses import dataclass 
from scipy import misc

class Cluster:
    
    def __init__(self,centroid_value):
        self.centroid_value= centroid_value
        self.data_points = []
        self.previous_centroid_value = -1

    def __eq__(self, value):
       return self.centroid_value

    def add_data_point(self,point):
        self.data_points.add(point)

    def normalize_centroid(self):
        old_sum_r = sum([point[0] for point in self.data_points])
        old_sum_g = sum([point[1] for point in self.data_points])
        old_sum_b = sum([point[2] for point in self.data_points])

        new_r = float(float(old_sum_r) / float(len(self.data_points)))
        new_g = float(float(old_sum_g) / float(len(self.data_points)))
        new_b = float(float(old_sum_b) / float(len(self.data_points)))

        self.centroid_value = (new_r, new_g, new_b)

    def fetch_color_equivalent(self):
        return self.centroid_value


def chose_random_triples():
    centroids = set()
    iterator = 0
    while iterator < 5:
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        color = (r,g,b)
        if color in centroids:
            continue
        centroids.add(color)
        iterator += 1

    return centroids


def euclidian(pointA, pointB):
    return np.linalg.norm(pointA-pointB)

def fetch_closest_cluster(pixel,clusters):
    chosen_cluster = None
    min_diff = 766
    for c in clusters:
        r_diff = euclidian(c.centroid_value,pixel[0])
        g_diff = euclidian(c.centroid_value,pixel[1])
        b_diff = euclidian(c.centroid_value,pixel[2])
        total_diff = r_diff + b_diff + g_diff
        chosen_cluster = c if total_diff < min_diff else min_diff
        min_diff = min(total_diff,min_diff)

    return chosen_cluster



def k_means(image):
    cluster_values = chose_random_triples()
    
    clusters = set()
    for c in clusters:
        clusters.add(Cluster(c))

    threshold = 5
    convergence_counter = 0

    while convergence_counter < 3:
        for row in range(len(image)):
            for col in range(len(image[0])):
                pixel = image[row][col]
                chosen_cluster = fetch_closest_cluster(pixel,clusters)
                chosen_cluster.add_data_point(pixel)

                change_in_weight = 0
                for cluster in clusters:
                    previous_value = cluster.centroid_value
                    cluster.normalize_centroid()
                    if previous_value == -1:
                        continue
                    cluster.previous_centroid_value = previous_value
                    if abs(cluster.centroid_value - cluster.previous_centroid_value) < threshold:
                        change_in_weight += 1

                if change_in_weight == len(clusters):
                    convergence_counter += 1

    return clusters






                
            
           

            


