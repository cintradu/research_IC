from tensorflow.image import stateless_random_brightness, stateless_random_contrast, rot90
import random
import numpy as np
import matplotlib.pyplot as plt

import ds_build as ds

img_paths = ds.dataset[:1000]
img_set = np.array([plt.imread(str(file_name)) for file_name in img_paths])

seed = (1,2)
for i in range(1000):
    img_set[i] = stateless_random_brightness(img_set[i], random.random(), seed)
    img_set[i] = stateless_random_contrast(img_set[i], random.uniform(0.0, 0.5), random.uniform(0.51, 1.0), seed)
    img_set[i] = rot90(img_set[i], k= random.choice([0,2]))

evaluation_ds = img_set