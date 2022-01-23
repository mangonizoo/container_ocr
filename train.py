import os
from xml.etree import ElementTree
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
#from keras.applications.densenet import DenseNet121
#from vgg import vgg_model
from keras.models import Model, load_model, Sequential
from keras.layers import Conv2D, Layer, Dense, Input, Flatten, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras import backend as K
from sklearn.utils import shuffle
from sklearn.utils import class_weight
import random

classes = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H',
           'I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
dataset_path = 'D:/Heejoo/cont_ocr/all_realchars/'

with open('D:/Heejoo/cont_ocr/listfile.txt','r',newline='\n') as f:
    train_files = f.readlines()

# path = 'D:/Hatem/drawing polygon test/all_realchars/'
# samples = []
# for i in classes:
# list = os.listdir(path+i)
# for j in range(len(list)):
# samples.append(i)
# samples = np.array(samples)
# class_weights = class_weight.compute_class_weight('balanced', np.unique(samples),samples)

train_samples = len(train_files)
batch_size = 256
label = []
train_files = shuffle(train_files)
train_h, train_w = 64,32
def data_generator(files, batch_size=16, number_of_batches=None):
    counter = 0
    #n_classes = 3

    #training parameters
    while True:
        idx_start = batch_size * counter
        idx_end = batch_size * (counter + 1)
        x_batch = []
        y_batch = []
        for file in files[idx_start:idx_end]:
            img = cv2.imread(dataset_path+file.split('\n')[0])
            img = cv2.resize(img, (train_w , train_h))
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h = hsv[:, :, 0] / 179.
            h += random.uniform(-0.5, 0.5)
            h[h > 1] = 1.0
            h[h < 0] = 0
            hsv[:, :, 0] = h * 179
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            img = img /255.0
            x_batch.append(img)
            map = np.zeros(36)
            class_name = file.split('/')[0]
            map[classes.index(class_name)] = 1
            y_batch.append(map)
        counter += 1
        x_train = np.array(x_batch)
        y_train = np.array(y_batch)
        yield x_train, y_train
        if (counter == number_of_batches):
            counter = 0

inp = Input(shape=(64,32,3))
x = Conv2D(16,3, padding = 'same', activation = 'relu')(inp)
x = MaxPooling2D(2)(x)
x = Conv2D(32,3, padding = 'same', activation = 'relu')(x)
x = MaxPooling2D(2)(x)
x = Conv2D(64,3, padding = 'same', activation = 'relu')(x)
x = MaxPooling2D(2)(x)
x = Conv2D(128,3, padding = 'same', activation = 'relu')(x)
x = Flatten()(x)
x = Dense(1024, activation= 'relu')(x)
x = Dense(512, activation= 'relu')(x)
x = Dense(256, activation= 'relu')(x)
x = Dense(128, activation= 'relu')(x)
x = Dense(64, activation= 'relu')(x)
out = Dense(36, activation= 'softmax')(x)
model = Model(inputs = inp, outputs = out)
print(model.summary())

# class_weights = [ 3.01030939, 1.00000004, 1.12886602,  2.46067425, 1.42903757,
#                   1.66857149, 3.91071443, 2.68711667,  2.15233423, 2.67073181,
#                   8.58823562, 125.14286188, 5.57961805, 31.28571547, 9.95454583,
#                   10.95000041,  1.56428577, 8.11111142,  21.36585447,  54.75000207,
#                   8.11111142,  25.02857238,   5.09302345,  25.02857238, 62.57143094,
#                   17.8775517, 43.80000166, 15.92727333, 6.34782633, 5.21428591, 1.76969704,
#                   23.67567657, 54.75000207, 38.08695796, 48.66666851, 41.71428729]

class_weights = { 0:3.01030939, 1:1.00000004, 2:1.12886602,  3:2.46067425, 4:1.42903757,
                  5:1.66857149, 6:3.91071443, 7:2.68711667,  8:2.15233423, 9:2.67073181,
                  10:8.58823562, 11:125.14286188, 12:5.57961805, 13:31.28571547, 14:9.95454583,
                  15:10.95000041,  16:1.56428577, 17:8.11111142,  18:21.36585447,  19:54.75000207,
                  20:8.11111142,  21:25.02857238, 22:5.09302345, 23:25.02857238, 24:62.57143094,
                  25:17.8775517, 26:43.80000166, 27:15.92727333, 28:6.34782633, 29:5.21428591, 30:1.76969704,
                  31:23.67567657, 32:54.75000207, 33:38.08695796, 34:48.66666851, 35:41.71428729}
#class_weights = {i: weights1[i] for i in range(len(weights1))} 

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

filepath='D:/Heejoo/cont_ocr/weights/weights-improvement-{epoch:02d}-{accuracy:.4f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=1)
callbacks_list = [checkpoint]
print(callbacks_list)

model.fit_generator(data_generator(train_files, batch_size, number_of_batches= train_samples // batch_size),
                    steps_per_epoch=(train_samples//batch_size), initial_epoch = 0,
                    #validation_data= data_generator(test_files, batch_size, number_of_batches= test_samples // batch_size),
                    #validation_steps=max(1, test_samples//batch_size),
                    epochs=500 , class_weight = class_weights,
                    callbacks= callbacks_list)