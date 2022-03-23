from json import decoder
from cupshelpers import activateNewPrinter
import numpy as np

from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, MaxPool2D, UpSampling2D, Reshape
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay


dataset = np.load("dataset.npy")

#lr = PolynomialDecay(1e-5, 10000, 1e-6, power= 0.5)
opt = Adam(learning_rate= 1e-5)
loss = BinaryCrossentropy()

batchsize = 64
epochs = 100


encoder_input = Input(shape= (270,440,1), dtype= "float32", name= "input_img")
x = Conv2D(3, (3,2), activation= "relu", padding= "SAME")(encoder_input)
x = MaxPool2D((3,2), padding= "SAME")(x)
x = Conv2D(3, (3,2), activation= "relu", padding= "SAME")(x)
x = MaxPool2D((3,2), padding= "SAME")(x)
encoder_output = Flatten()(x)

encoder = Model(encoder_input, encoder_output, name= "encoder")

decoder_input = Reshape((30,110,3))(encoder_output)
y = UpSampling2D((3,2))(decoder_input)
y = Conv2DTranspose(3, (3,2), activation= "relu", padding= "SAME")(y) 
y = UpSampling2D((3,2))(y)
decoder_output = Conv2DTranspose(1, (3,2), activation= "relu", padding= "SAME")(y) 

autoencoder = Model(encoder_input, decoder_output, name= "AutoEncoder")


autoencoder.compile(optimizer= opt, loss= loss, metrics=['accuracy'])
autoencoder.fit(dataset, dataset, epochs= epochs, batch_size= batchsize, validation_split= 0.2, verbose= 2)

