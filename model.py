import os
import csv
import cv2
import numpy as np
import sklearn
from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

batch_size = 32

top_cut = 70
bottom_cut = 25
        
def mirror_image(image):
    
    return cv2.flip(image, 1)
        
def brightness(image):
    
    img = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    img[:,:,2] = img[:,:,2]*(np.random.uniform(0.25,1.25))
    img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    
    return img

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
samples = samples[1:]

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name_center = 'data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.cvtColor(cv2.imread(name_center),cv2.COLOR_BGR2RGB)
                name_left = 'data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.cvtColor(cv2.imread(name_left),cv2.COLOR_BGR2RGB)
                name_right = 'data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.cvtColor(cv2.imread(name_right),cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                left_angle = float(batch_sample[3])+0.15
                right_angle = float(batch_sample[3])-0.15
                images.append(brightness(center_image))
                images.append(brightness(left_image))
                images.append(brightness(right_image))
                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)
                if abs(float(batch_sample[3]))>0.1:
                    mirror_center = mirror_image(center_image)
                    mirror_left = mirror_image(left_image)
                    mirror_right = mirror_image(right_image)
                    mirror_center_angle = float(center_angle)*-1.0
                    mirror_left_angle = float(left_angle)*-1.0
                    mirror_right_angle = float(right_angle)*-1.0
                    
                    images.append(brightness(mirror_center))
                    images.append(brightness(mirror_left))
                    images.append(brightness(mirror_right))
                    angles.append(mirror_center_angle)
                    angles.append(mirror_left_angle)
                    angles.append(mirror_right_angle)

            # trim image to only see section with road
            # images = np.asarray(images)
            # X_train = np.array(brightness_process_image(images))
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((top_cut,bottom_cut),(0,0))))
model.add(Convolution2D(24,(5,5),activation="relu",strides=(2,2)))
model.add(Dropout(.4))
model.add(Convolution2D(36,(5,5),activation="relu",strides=(2,2)))
model.add(Dropout(.4))
model.add(Convolution2D(48,(5,5),activation="relu",strides=(2,2)))
model.add(Dropout(.4))
model.add(Convolution2D(64,(3,3),activation="relu"))
model.add(Dropout(.4))
model.add(Convolution2D(64,(3,3),activation="relu"))
model.add(Dropout(.4))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#model.fit_generator(train_generator, validation_data=validation_generator, samples_per_epoch= len(train_samples), nb_val_samples=len(validation_samples), nb_epoch=3)
model.fit_generator(train_generator, 
                    validation_data=validation_generator, 
                    steps_per_epoch= len(train_samples)/batch_size, 
                    epochs=3, 
                    validation_steps=len(validation_samples)/batch_size)

model.save('model.h5')