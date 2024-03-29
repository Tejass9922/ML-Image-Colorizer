

import numpy as np
import random
from preprocessor import *
from k_means_model import *
from scipy import signal
import cv2


#Returns sigmoid value at x
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#returns derivative of sigmoid function at x
def sigmoid_derivative(x):
    return x * (1 - x)

def fetch_neighbors_create_patch(gray_image,r,c):
    row = [-1, -1, -1,  0, 0, 0,  1, 1, 1]
    col = [-1,  0,  1, -1, 0, 1, -1, 0, 1]
    patch = []
    for i in range(9):
        x = r + row[i]
        y = c + col[i]
        patch.append([gray_image[x][y]])

    return np.array(patch)
def assign_gray_pixels(img):
    remaining_pixels = set()
    for i in range(len(img)):
        for j in range(len(img[0])):
            remaining_pixels.add((i,j))
    return remaining_pixels
      
def train_network(training_images):
    np.random.seed(1)  
    #Assign random weights for layers. In this case we decided to use negative values initially as well to attain the most accurate possible results when training our network. Change this a bit
    layer1_weights = 2 * np.random.random((255,9)) - 1
    layer2_weights = 2 * np.random.random((255,255)) - 1
    output_weights = 2 * np.random.random((3,255)) - 1
    
    for training_image in training_images:
     
        #for training_image in training_images:
        gray_iamge = training_image.gray_iamge
        training_color_image = training_image.colored_image
        blue,green,red = cv2.split(training_color_image)
        blue_iamge,green_image,red_image = (np.array(blue),np.array(green),np.array(red))


        remaining_pixels = set()
        visited = set()

        padded_image = np.pad(gray_iamge, 1, 'edge')

        completed_evolution = 3
        #Train till completed evolution. Most mean data suggested that training on left half of 1 simple image would be best fitted for 3 training cycles (without overfitting)
        for x in range(completed_evolution):
            print("--Training cycle #: %d --"%x) 
            remaining_pixels = assign_gray_pixels(gray_iamge)
            while len(remaining_pixels) > 0:
                (x_rand,y_rand) = random.choice(tuple(remaining_pixels))
                remaining_pixels.remove((x_rand,y_rand))
                
                current_patch = fetch_neighbors_create_patch(padded_image, x_rand, y_rand) / 255
                
                current_pixel = np.array([[red[x_rand][y_rand]], [green[x_rand][y_rand]], [blue[x_rand][y_rand]]]) / 255


                #Forward Propogation: use sigmoid function to calculate prediction based on current layer weights

                layer1 = np.dot(layer1_weights,current_patch)
                layer1_result = sigmoid(layer1)
                
                layer2 = np.dot(layer2_weights,layer1_result)
                layer2_result = sigmoid(layer2)

                output = np.dot(output_weights,layer2_result)
                output_result = sigmoid(output)
                
                #Error Checking within the weights and Back propogation activated after error calculation 
                
                error = current_pixel - output_result
                error_output_derivative = error * sigmoid_derivative(output_result) 

                output_transpose = output_weights.T 
                layer2_error = output_transpose.dot(error_output_derivative)
                error_layer2_derivative = layer2_error * sigmoid_derivative(layer2_result)
                
                layer2_transpose = layer2_weights.T 
                layer1_error = layer2_transpose.dot(error_layer2_derivative)
                error_layer1_derivative = layer1_error * sigmoid_derivative(layer1_result)

                
                #Update the weights of each layer according to the error calculation
                output_weights += error_output_derivative.dot(layer2_result.T)
                layer2_weights += error_layer2_derivative.dot(layer1_result.T)
                layer1_weights += error_layer1_derivative.dot(current_patch.T)
                
          
            
    np.savetxt('layer1_final_weights_m', layer1_weights)
    np.savetxt('layer2_final_weights_m', layer2_weights)
    np.savetxt('output_final_weights_m', output_weights)

    return layer1_weights, layer2_weights, output_weights


def color_image(network_weights,test_image):
    #Use Layer Weights from input (previously trained network model)
    layer1_weights = network_weights[0]
    layer2_weights = network_weights[1]
    output_weights = network_weights[2]
    grayscale_img = test_image.right_g

    r = len(grayscale_img)
    c = len(grayscale_img[0])

    red = np.zeros((r, c), dtype='uint8')
    green = np.zeros((r, c), dtype='uint8')
    blue = np.zeros((r, c), dtype='uint8')
    

    # Pad image around border to compute all values
    padded_image = np.pad(grayscale_img, 1, 'edge')
    print("--Coloring Image from Grayscale Input--")
    for i in range(r):
        for j in range(c):
            # Fetch 3x3 pixel patch around current pixel
           
            '''
            for i in range(len(current_patch)):
                current_patch[i][0] = current_patch[i][0] / 255
            '''
            current_patch = fetch_neighbors_create_patch(padded_image, i, j) / 255
            # Compute results layer by layer until we get to output        
            layer1 = np.dot(layer1_weights,current_patch)
            layer1_result = sigmoid(layer1)
            
            layer2 = np.dot(layer2_weights,layer1_result)
            layer2_result = sigmoid(layer2)

            output = np.dot(output_weights,layer2_result)
            output_result = sigmoid(output)
           
            try:
                (red[i][j],green[i][j],blue[i][j]) = (output_result[0][0] * 255, output_result[1][0] * 255, output_result[2][0] * 255)
            except:
                print("error")
            

    # Using opencv, merge all three color filters into one image and display it
    final_output = cv2.merge((blue, green, red))
    print("--Finished Coloring Grayscale input--")
  
  
    return final_output


training_image = Image('training_image.png')
training_image2 = Image('training_image_2.png')
testing_image_color = training_image.right_c
testing_image_gray = training_image.right_g
cv2.imshow('gray input',testing_image_gray)
cv2.imshow('original input', testing_image_color)
cv2.waitKey(0)
t1 = Image('training_images/train1.png.png')
t2 = Image('training_images/train2.png.png')
t3 = Image('training_images/train3.png.png')
training_images = [t1,t2,t3]

#Use line below to train the network again with different images
#network_weights = train_network(training_images) 

#Use this line for network to use the saved weights from previous training data
w1,w2,ow = ( np.loadtxt('layer1_final_weights'), np.loadtxt('layer2_final_weights'), np.loadtxt('output_final_weights') )
network_weights = (w1,w2,ow)

#Using right hand side of colored image as testing data
testing_image = Image('training_image.png')
test_image_colored = color_image(network_weights,testing_image)

#Left Side of testing image as training  Right side of image as testing

im_h = cv2.hconcat([training_image.left_c, test_image_colored])
  
# show the output image
cv2.imshow('testing_colored_finished_right.png', im_h)
#cv2.imwrite('completed_nerual_network.png',im_h)
cv2.waitKey(0)
cv2.destroyAllWindows()
