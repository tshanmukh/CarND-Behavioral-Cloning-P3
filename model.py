from keras.models import Sequential, Model
from keras.layers import Lambda, Dense, Conv2D, BatchNormalization, Activation, Flatten

# defining the network
# Using the Nvidia architechture as explained in the classroom that it achived best accuracy

model = Sequential() # getting the instance of a model
model.add(Lambda(lambda x : (x/255)-1 , input_shape=(66,200,3)))    #Normalization layer

# First convolution layer
model.add(Conv2D(filters=24,kernel_size=5,strides=(2,2)))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Second convolution layer
model.add(Conv2D(filters=36,kernel_size=5,strides=(2,2)))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Third convolution layer
model.add(Conv2D(filters=48,kernel_size=5,strides=(2,2)))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Fourth Convolution Layer
model.add(Conv2D(filters=64,kernel_size=3,strides=(1,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Fifth Convolution Layer
model.add(Conv2D(filters=64,kernel_size=3,strides=(1,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))

# End of Convolution layers
# Flatten and fullyconnected layers

model.add(Flatten()) # Flattening the image here to pass to a fully connected layer

# FC1
model.add(Dense(100))
model.add(BatchNormalization())
model.add(Activation('relu'))

#FC2
model.add(Dense(50))
model.add(BatchNormalization())
model.add(Activation('relu'))

# FC3
model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('relu'))

# Adding an extra layer to predict just one logit
# FC4
model.add(Dense(1))                 # As we are predicting only the steering angle we use only one
model.add(BatchNormalization())
model.add(Activation('relu'))

# Compiling the model

model.compile(loss='mse',optimizer='adam')

model.summary()


