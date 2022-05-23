from base64 import decode
from simplejson import load

from tensorflow.keras.models import load_model
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import cv2

import evaluation_ds as evds

eval_img = evds.evaluation_ds[randint(0,1000)]

eval_img = cv2.cvtColor(eval_img, cv2.COLOR_BGR2GRAY)
eval_img = eval_img.reshape(-1, 270, 440, 1)

#model_ec = load_model("")
model_ac = load_model("AC_v2_255k")

#encoded_img = 255.*(model_ec.predict([eval_img]))
decoded_img = 255.*(model_ac.predict([eval_img]))

decoded_img = decoded_img.reshape(270,440,1)
eval_img = eval_img.reshape(270,440,1)


result = plt.figure()
result.add_subplot(1,2,1)
plt.imshow(eval_img, cmap= 'gray')
result.add_subplot(1,2,2)
plt.imshow(decoded_img, cmap= 'gray')
plt.show()
