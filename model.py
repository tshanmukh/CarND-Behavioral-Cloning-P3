from keras.models import Sequential, Model
from keras.layers import Lambda, Dense, Conv2D, BatchNormalization, Activation, Flatten
import csv
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import matplotlib.image as mpimg

# defining the network
# Using the Nvidia architechture as explained in the classroom that it achived best accuracy

model = Sequential() # getting the instance of a model
model.add(Lambda(lambda x : (x/255)-1 , input_shape=(160,320,3)))    #Normalization layer

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

# Reading the data
path = './dataset/' # adding a variable to path for convinience to switch between different training datas
dataset = [] # empty list to append the filenames
with open(path+'driving_log.csv') as file:
    reader = csv.reader(file)
    Header = True
    for line in reader:
        if Header:
            Header = False
        else:
            dataset.append(line)

            
            
train_samples, validation_samples = train_test_split(dataset,test_size=0.2)
# defining the generator
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
                        #name = path+'IMG/'+line[i].split('/')[-1]
                        #print(line)
                        #center_image = cv2.cvtColor(cv2.imread(line[i]), cv2.COLOR_BGR2RGB) 
                        center_image = mpimg.imread(line[i])
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

# Compiling and training the model
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator,   nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)

model.save('model.h5') # saving the model
