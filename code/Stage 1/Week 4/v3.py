from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, MaxPool2D, UpSampling2D, Reshape
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import ds_build as ds


epochs = 150
batchsize = 32
training_ds = ds.My_Custom_Generator(ds.dataset[:6461], batchsize)
validation_ds = ds.My_Custom_Generator(ds.dataset[6461:], batchsize)

lr_schedule = ExponentialDecay(1e-3, decay_steps=10000, decay_rate=0.96, staircase=True)
opt = Adam(learning_rate= lr_schedule, amsgrad= True)
loss = MeanSquaredError()


encoder_input = Input(shape= (270,440,1), dtype= "float32", name= "input_img")
x = Conv2D(32, (8,4), activation= "relu", padding= "VALID")(encoder_input)
x = Conv2D(16, (4,2), activation= "relu", padding= "VALID")(x)
x = Conv2D(8, (2,1), activation= "relu", padding= "VALID")(x)
encoder_output = Flatten()(x)

encoder = Model(encoder_input, encoder_output, name= "encoder")

decoder_input = Reshape((259,436,8))(encoder_output)
y = Conv2DTranspose(16, (2,1), activation= "relu", padding= "VALID")(decoder_input) 
y = Conv2DTranspose(32, (4,2), activation= "relu", padding= "VALID")(y)
decoder_output = Conv2DTranspose(1, (8,4), activation= "relu", padding= "VALID")(y) 

autoencoder = Model(encoder_input, decoder_output, name= "AutoEncoder")


autoencoder.compile(optimizer= opt, loss= loss, metrics=['accuracy'])
autoencoder.fit(x= training_ds, epochs= epochs, verbose= 2, validation_data= validation_ds)