import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from imutils import paths
import argparse,random,cv2,os,matplotlib,keras,warnings
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator,img_to_array
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from keras.optimizers import Adam,SGD
from keras.models import Sequential
from keras.layers.convolutional import MaxPooling2D,AveragePooling2D,Conv2D
from keras.layers.core import Activation,Flatten,Dense,Dropout
from keras import backend as K

imagepaths=sorted(list(paths.list_images(r"masks/")))
data=[]
labels=[]
dim=(128,128)
for imagePath in imagepaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, dim)
    image = img_to_array(image)
    data.append(image)
    if(imagePath[7:8]!='-'):
        label=imagePath[6:8]
    else:
        label = imagePath[6:7]
    labels.append(int(label))
num_classes= len(np.unique(labels))

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.25, random_state=44)
trainY = to_categorical(trainY, num_classes)
testY = to_categorical(testY, num_classes)
channels=(trainX.shape[3])


model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(128, 128,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3),activation='relu'))
model.add(Conv2D(64, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
model.add(Conv2D(64, (3, 3),activation='relu'))
model.add(Conv2D(64, (3, 3),activation='relu'))
model.add(Conv2D(128, (3, 3),activation='relu'))
model.add(Conv2D(128, (3, 3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

optimizer = Adam(lr=0.001)
# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
epochs = 10 
batch_size = 64

datagen = ImageDataGenerator(rotation_range=10, zoom_range = 0.1, width_shift_range=0.1, height_shift_range=0.1,horizontal_flip=False,   vertical_flip=False)  
datagen.fit(trainX)
history = model.fit_generator(datagen.flow(trainX,trainY, batch_size=batch_size),epochs = epochs, validation_data = (testX,testY), verbose = 2, steps_per_epoch=trainX.shape[0] // batch_size)