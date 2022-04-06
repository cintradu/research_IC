import numpy as np
import cv2
import matplotlib.pyplot as plt 
import random
import os
from tensorflow.keras.utils import Sequence


DATADIR = '/home/luis/Desktop/IC/dataset/rgb'

all_images = []

img_folder = os.listdir(DATADIR)

for img in img_folder:
    img_path = os.path.join(DATADIR, img)
    all_images.append(img_path)

class My_Custom_Generator(Sequence) :

    def __init__(self, image_filenames, batch_size) :
        self.image_filenames = image_filenames
        self.batch_size = batch_size

    def __len__(self) :
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx) :
        batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_x = np.array([np.reshape(cv2.cvtColor(plt.imread(str(file_name)), cv2.COLOR_RGB2GRAY),(270,440,1)) for file_name in batch_x])/255.

        return batch_x, batch_x

random.shuffle(all_images)

dataset = all_images
