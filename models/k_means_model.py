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

    def __repr__(self):
        return str(self.centroid_value)

    def add_data_point(self,point):
        self.data_points.append(point)

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


def chose_random_triples(image):
    '''
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
    '''
    centroids = []
    used_locations = set()
    i = 0
    while i < 5:
        x = random.randint(0,len(image))
        y = random.randint(0,len(image[0]))
        pixel = image[x][y]
        if (x,y) in used_locations:
            continue
        used_locations.add((x,y))
       
      
        centroids.append(pixel)
        i += 1
    #print("Starting 5: ")
    #print(centroids)
    return centroids
   

def euclidian(pointA, pointB):
    return np.linalg.norm(int(pointA)-int(pointB))

def fetch_closest_cluster(pixel,clusters):
    chosen_cluster = None
    min_diff = 766
    for c in clusters:
        
        r_diff = euclidian(c.centroid_value[0],pixel[0])
        g_diff = euclidian(c.centroid_value[1],pixel[1])
        b_diff = euclidian(c.centroid_value[2],pixel[2])
        total_diff = r_diff + b_diff + g_diff
        chosen_cluster = c if total_diff < min_diff else chosen_cluster
        min_diff = min(total_diff,min_diff)
       

    return chosen_cluster



def k_means(image):
    cluster_values = chose_random_triples(image)
    
    clusters = []
    for c in cluster_values:
        clusters.append(Cluster(c))

    threshold = 5
    convergence_counter = 0

    while convergence_counter < 3:
        for row in range(len(image)):
            for col in range(len(image[0])):
                pixel = image[row][col]
                #print(pixel)
                chosen_cluster = fetch_closest_cluster(pixel,clusters)
               
                chosen_cluster.add_data_point(pixel)
                
                change_in_weight = 0
                
        for cluster in clusters:
            #(cluster, cluster.data_points)
            previous_value = cluster.centroid_value
            cluster.normalize_centroid()
            if len(previous_value) != 3:
                continue
            cluster.previous_centroid_value = previous_value
            diff = 0
            for i in range(3):
                diff += abs(euclidian(cluster.centroid_value[i], cluster.previous_centroid_value[i]))
            
            if diff < threshold:
                change_in_weight += 1

        if change_in_weight == len(clusters):
            convergence_counter += 1

    processed_clusters = []
    for c in clusters:
        processed_clusters.append(c.centroid_value)
    return processed_clusters




                


                
            
           

            


