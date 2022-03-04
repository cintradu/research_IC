import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.layers import Rescaling


# creating a dataset with the robot images
img_height = 270
img_width = 440

# 80% for training and 20% for validation
train_ds = image_dataset_from_directory(
    '/home/luis/Desktop/IC/dataset',  
    labels= None,
    batch_size= None,
    image_size= (img_height, img_width),
    seed = 123,
    validation_split= 0.2,
    subset= "training"
)
val_ds = image_dataset_from_directory(
    '/home/luis/Desktop/IC/dataset',
    labels= None,
    batch_size= None,
    image_size= (img_height, img_width),
    seed = 123,
    validation_split= 0.2,
    subset= "validation"
)

# normalizating the database
normalization_layer = Rescaling(1./255)
normalized_ds = train_ds.map(lambda x: (normalization_layer(x)))

# configuring the dataset for better performing (prefetch overlaps data preprocessing and model execution during training)
AUTOTUNE = tf.data.AUTOTUNE 

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)