#Importing Required libraries to make the model work
import os
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda, Cropping2D

model = Sequential()  # Defining the model
 
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3))) # fitst input layer for normalization

model.add(Cropping2D(cropping=((70,25),(0,0)))) # cropping the unwanted portions in the above and below       

model.add(Convolution2D(24,5,5,subsample=(2,2))) # convolution layers
model.add(Activation('elu'))						# Activation function is used as elu instead for relu

model.add(Convolution2D(36,5,5,subsample=(2,2)))
model.add(Activation('elu'))

model.add(Convolution2D(48,5,5,subsample=(2,2)))
model.add(Activation('elu'))

model.add(Convolution2D(64,3,3))
model.add(Activation('elu'))

model.add(Convolution2D(64,3,3))
model.add(Activation('elu'))

model.add(Flatten())					# Flattening the image for the Fully connected layer

model.add(Dense(100))
model.add(Activation('elu'))

model.add(Dropout(0.25))

model.add(Dense(50))
model.add(Activation('elu'))


model.add(Dense(10))
model.add(Activation('elu'))

model.add(Dense(1)) # As we need only one prediction i.e., steering angle

model.compile(loss='mse',optimizer='adam')

model.summary()

samples = [] 

with open('./data/data/driving_log.csv') as csvfile: # reading the csv file
    reader = csv.reader(csvfile) 
    next(reader, None)  # taking out one iter from a iterable to skip the header info
    for line in reader:
        samples.append(line)


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

train_samples, validation_samples = train_test_split(samples,test_size=0.3)

import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def generator(data, batch_size=32):
    while 1:
        for batch in range(0,len(data),batch_size): # using a step as batch_size to get data as batches
            bc = data[batch:batch+batch_size]
            #print(bc)
            images = [] # placeholder for images
            steer = [] # placeholder for steering angle    
            for line in bc:
                for i in range(0,3): #we are taking 3 images, first one is center, second is left and third is right
                        
                        #print(line)
                        name = './data/data/IMG/'+line[i].split('/')[-1]
                        #print(line)
                        #center_image = cv2.cvtColor(cv2.imread(line[i]), cv2.COLOR_BGR2RGB) 
                        center_image = mpimg.imread(name)
                        center_angle = float(line[3]) #getting the steering angle measurement
                        images.append(center_image)
                        
                        # adding a correction for non center image using the same for tuning the model
                        correction = 0.2
                        if(i==0):
                            steer.append(center_angle)
                        elif(i==1):
                            steer.append(center_angle+correction)
                        elif(i==2):
                            steer.append(center_angle-correction)
                        
                        # Data augmentation by flipping the image and multiplying the the steering angle with -1
                        images.append(cv2.flip(center_image,1))
                        if(i==0):
                            steer.append(center_angle*-1)
                        elif(i==1):
                            steer.append((center_angle+correction)*-1)
                        elif(i==2):
                            steer.append((center_angle-correction)*-1)
                            
                        
            # converting the image data to np arrays
            X_train = np.array(images)
            y_train = np.array(steer)
            
            # Adding the yeild instead of the return makes this funtion a generator
            yield sklearn.utils.shuffle(X_train, y_train)          

train_generator = generator(train_samples, batch_size=32)			# getting the training images from the generator
validation_generator = generator(validation_samples, batch_size=32) # getting the validation images from the generator


model.fit_generator(train_generator, samples_per_epoch= len(train_samples)/32, validation_data=validation_generator,   nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)

model.save('model.h5')