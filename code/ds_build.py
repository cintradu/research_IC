import numpy as np
import cv2
import matplotlib.pyplot as plt 
import random
import os

DATADIR = '/home/luis/Desktop/IC/dataset'
CATEGORIES = ['rgb']


training_data= []

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)

    for img in os.listdir(path):
        
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        img_array = cv2.normalize(img_array, img_array, 0, 255, cv2.NORM_MINMAX)
        training_data.append(img_array)
        
random.shuffle(training_data)

np.save("dataset.npy", training_data)
