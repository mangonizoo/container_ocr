import os
from xml.etree import ElementTree
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
#from tensorflow.keras.applications.densenet import DenseNet121
#from vgg import vgg_model
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Conv2D, Layer,Input, Dense,Flatten , UpSampling2D, GlobalAveragePooling2D, GlobalMaxPooling2D,MaxPooling2D
from tensorflow.keras import backend as K
from sklearn.utils import shuffle
import random
import cv2
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import os
from keras.utils.vis_utils import plot_model


#Classification
classes = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H',
           'I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']


inp = Input(shape=(64,32,3))
x = Conv2D(32,3, padding = 'same', activation = 'relu')(inp)
x = MaxPooling2D(2)(x)
x = Conv2D(64,3, padding = 'same', activation = 'relu')(x)
x = MaxPooling2D(2)(x)
x = Conv2D(128,3, padding = 'same', activation = 'relu')(x)
x = Flatten()(x)
x = Dense(512, activation= 'relu')(x)
x = Dense(256, activation= 'relu')(x)
x = Dense(128, activation= 'relu')(x)
x = Dense(64, activation= 'relu')(x)
out = Dense(36, activation= 'softmax')(x)
model = Model(inputs = inp, outputs = out)
# model.load_weights('D:/Heejoo/cont_ocr/weights/weights-improvement-408-0.9999.h5')

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
#Binary box detection
path = 'D:/Heejoo/cont_ocr/horizontal/'
images = os.listdir(path)
for image in images:
    print(image)
    originalImage = cv2.imread(path+image)
    img_h, img_w, _ = originalImage.shape
    grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

    print('img size', img_h*img_w)
    #(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    region_type = 1
    if region_type == 1:
        size_thr = 0.0055
    else:
        size_thr = 0.002
    blur = cv2.GaussianBlur(grayImage,(5,5),0)
    (thresh, blackAndWhiteImage) = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    white_pix = np.where(blackAndWhiteImage  == 255)

    if len(white_pix[0]) > 0.5*img_h*img_w:
        blackAndWhiteImage = cv2.bitwise_not(blackAndWhiteImage)

    filter = np.ones((3, 3), dtype=np.int)
    labeled, regions = label(blackAndWhiteImage, filter)

    #labeled, regions = label(blackAndWhiteImage, filter)
    loc_new = {}
    size_mean = 0
    height_mean = 0
    width_mean = 0
    #vertical separation
    if region_type == 0:
        if img_h > 2.5*img_w :
            half_img = blackAndWhiteImage[0:img_h,int(img_w/2):img_w]/255.0
            # cv2.imshow('test',half_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            dial_h = int(img_h/50)
            a = np.zeros((dial_h, dial_h))

            a[0:dial_h, int(dial_h/2)] = 1
            half_img = binary_dilation(half_img, structure=a).astype(half_img.dtype)

            labeled2, regions2 = label(half_img, filter)
            max_h = 0
            for n in range(regions2):
                loc_half = np.where(labeled2 == n+1)
                region_h = len(np.unique(loc_half[0]))
                if region_h >= max_h:
                    max_h = region_h
                    max_reg= n+1
                print('max;',max_reg)

            for n in range(regions2):
                if n+1 != max_reg:
                    loc_half = np.where(labeled2 == n+1)
                    blackAndWhiteImage[loc_half[0],loc_half[1]+int(img_w/2)] = 0
                    # cv2.imshow('test', blackAndWhiteImage)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
            #cv2.imshow('half image',half_img)
            # cv2.imwrite('half image.png',half_img)
        labeled, regions = label(blackAndWhiteImage, filter)

        #print(regions)
        for i in range(regions):
            loc = np.where(labeled == i+1)
            box_h = (np.amax(loc[0]) - np.amin(loc[0])) // (img_h/12)
            if box_h >= 1 and img_h > 2.5*img_w:
                for k in range(int(box_h)):
                    blackAndWhiteImage[loc[0][0]+(int(img_h/12.5)*(k+1)),np.unique(loc[1])]= 0
        labeled, regions = label(blackAndWhiteImage, filter)
        for i in range(regions):
            loc = np.where(labeled == i+1)
            size = len(np.unique(loc[0]))*len(np.unique(loc[1]))
            size_mean += size
            height_mean += len(np.unique(loc[0]))
            width_mean += len(np.unique(loc[1]))
            if size < size_thr*img_h*img_w or len(np.unique(loc[0])) < (img_h/22):
                blackAndWhiteImage[loc] = 0
        size_mean /= regions
        width_mean /= regions
        labeled, regions = label(blackAndWhiteImage, filter)
        # for i in range(regions):
        # loc = np.where(labeled == i+1)
        # #size = len(np.unique(loc[0]))*len(np.unique(loc[1]))
        # if len(np.unique(loc[1])) >= 1.25*width_mean: #or len(np.unique(loc[0])) <= height_mean:
        # blackAndWhiteImage[loc[0],int((np.max(loc[1])-np.min(loc[1]))/2)] = 0
        # labeled, regions = label(blackAndWhiteImage, filter)
        for i in range(regions):
            loc = np.where(labeled == i+1)
            loc_new.update({i :np.amin(loc[0])})

        loc_sorted = {k: v for k, v in sorted(loc_new.items(), key=lambda item: item[1])}
        region= []
        box_min= []
        for key, value in loc_sorted.items():
            region.append(key)
            box_min.append(value)
        print(region)
        boxes = False
        count = 0

        for j in range(len(region)):
            loc = np.where(labeled == (region[j]+1))
            wide = np.amax(loc[1]) - np.amin(loc[1])
            high = np.amax(loc[0]) - np.amin(loc[0])
            try:
                loc2 = np.where(labeled == (region[j+1]+1))
            except:
                loc2 = np.array([0,0])
            if boxes == True:
                boxes = False
                continue #cv2.rectangle(originalImage,(np.amin(loc[1]),np.amin(loc[0])),(np.amax(loc[1]),np.amax(loc[0])),(0,0,255),1)
            elif (np.amin(loc2[1]) <= np.amax(loc[1]) and np.amax(loc2[1]) >= np.amin(loc[1])) and (np.amin(loc2[0]) <= np.amax(loc[0]) and np.amax(loc2[0]) >= np.amin(loc[0])):
                cv2.rectangle(originalImage,(min(np.amin(loc[1]),np.amin(loc2[1])),min(np.amin(loc[0]),np.amin(loc2[0]))),(max(np.amax(loc[1]),np.amax(loc2[1])),max(np.amax(loc[0]),np.amax(loc2[0]))),(0,0,255),1)
                char = model.predict(np.expand_dims(cv2.resize(originalImage[min(np.amin(loc[0]),np.amin(loc2[0])):max(np.amax(loc[0]),np.amax(loc2[0])),min(np.amin(loc[1]),np.amin(loc2[1])):max(np.amax(loc[1]),np.amax(loc2[1]))],(32,64)), axis = 0))
                cv2.putText(originalImage,classes[np.argmax(char[0])],(max(np.amax(loc[1]),np.amax(loc2[1])),max(np.amax(loc[0]),np.amax(loc2[0]))),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
                boxes = True
                count += 1
            else:
                if wide < 0.3*high:
                    cv2.rectangle(originalImage,(np.amin(loc[1])-int(wide/1.5),np.amin(loc[0])),(np.amax(loc[1])+int(wide/1.5),np.amax(loc[0])),(0,0,255),1)
                    char = model.predict(np.expand_dims(cv2.resize(originalImage[np.amin(loc[0]):np.amax(loc[0]), np.amin(loc[1])-int(wide/1.5):np.amax(loc[1])+int(wide/1.5)],(32,64)), axis = 0))
                    cv2.putText(originalImage,classes[np.argmax(char[0])],(np.amax(loc[1])+int(wide/1.5),np.amax(loc[0])),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
                else:
                    cv2.rectangle(originalImage,(np.amin(loc[1]),np.amin(loc[0])),(np.amax(loc[1]),np.amax(loc[0])),(0,0,255),1)
                    char = model.predict(np.expand_dims(cv2.resize(originalImage[np.amin(loc[0]):np.amax(loc[0]), np.amin(loc[1]):np.amax(loc[1])],(32,64)), axis = 0))
                    cv2.putText(originalImage,classes[np.argmax(char[0])],(np.amax(loc[1]),np.amax(loc[0])),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
                boxes = False
                count += 1
            # else:
            # cv2.rectangle(originalImage,(np.amin(loc[1]),np.amin(loc[0])),(np.amax(loc[1]),np.amax(loc[0])),(0,0,255),1)
        print(count)
    # misconnect = False
    #horizontal separation
    loc_new = {}
    if region_type == 1:
        if regions < 15:
            blackAndWhiteImage = cv2.bitwise_not(blackAndWhiteImage)
        labeled, regions = label(blackAndWhiteImage, filter)
        mean_width = 0
        for i in range(regions):
            loc = np.where(labeled == i+1)
            size = len(np.unique(loc[0]))*len(np.unique(loc[1]))
            if size < size_thr*img_h*img_w or size > (img_h*img_w*0.75):
                blackAndWhiteImage[loc] = 0
            if len(np.unique(loc[0])) < img_h/7 or len(np.unique(loc[0])) > img_h/1.5:  #len(np.unique(loc[1])) > img_w*0.2 or
                blackAndWhiteImage[loc] = 0
        labeled, regions = label(blackAndWhiteImage, filter)
        # for i in range(regions):
        # loc = np.where(labeled == i+1)
        # mean_width += (np.amax(loc[1]) - np.amin(loc[1]))
        mean_width /= regions
        # for i in range(regions):
        # loc = np.where(labeled == i+1)
        # if len(np.unique(loc[1]))> 2.2*mean_width: #or len(np.unique(loc[1]))< mean_width/2:            #(np.amax(loc[1]) - np.amin(loc[1])) > 2*mean_width:
        # #blackAndWhiteImage[loc] = 0
        # blackAndWhiteImage[np.unique(loc[0]), int((np.amax(loc[1]) + np.amin(loc[1]))/2)]= 0
        # labeled, regions = label(blackAndWhiteImage, filter)
        for i in range(regions):
            loc = np.where(labeled == i+1)
            loc_new.update({i :np.amin(loc[1])})

        print(loc_new)
        loc_sorted = {k: v for k, v in sorted(loc_new.items(), key=lambda item: item[1])}
        region= []
        box_min= []
        for key, value in loc_sorted.items():
            region.append(key)
            box_min.append(value)
        #print(region)
        boxes = False
        for j in range(len(region)):
            loc = np.where(labeled == (region[j]+1))
            try:
                loc2 = np.where(labeled == (region[j+1]+1))
            except:
                pass
            wide = np.amax(loc[1]) - np.amin(loc[1])
            high = np.amax(loc[0]) - np.amin(loc[0])
            if boxes == True:
                boxes = False
                continue #cv2.rectangle(originalImage,(np.amin(loc[1]),np.amin(loc[0])),(np.amax(loc[1]),np.amax(loc[0])),(0,0,255),1)
            elif (np.amin(loc2[1]) <= np.amax(loc[1]) and np.amax(loc2[1]) >= np.amin(loc[1])) and (np.amin(loc2[0]) <= np.amax(loc[0]) and np.amax(loc2[0]) >= np.amin(loc[0])):
                cv2.rectangle(originalImage,(min(np.amin(loc[1]),np.amin(loc2[1])),min(np.amin(loc[0]),np.amin(loc2[0]))),(max(np.amax(loc[1]),np.amax(loc2[1])),max(np.amax(loc[0]),np.amax(loc2[0]))),(0,0,255),1)
                char = model.predict(np.expand_dims(cv2.resize(originalImage[min(np.amin(loc[0]),np.amin(loc2[0])):max(np.amax(loc[0]),np.amax(loc2[0])),min(np.amin(loc[1]),np.amin(loc2[1])):max(np.amax(loc[1]),np.amax(loc2[1]))],(32,64)), axis = 0))
                cv2.putText(originalImage,classes[np.argmax(char[0])],(max(np.amax(loc[1]),np.amax(loc2[1])),max(np.amax(loc[0]),np.amax(loc2[0]))),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
                boxes = True
                #print("here")
            else:
                if wide < 0.25*high:
                    cv2.rectangle(originalImage,(np.amin(loc[1])-int(wide/1.5),np.amin(loc[0])),(np.amax(loc[1])+int(wide/1.5),np.amax(loc[0])),(0,0,255),1)
                    char = model.predict(np.expand_dims(cv2.resize(originalImage[np.amin(loc[0]):np.amax(loc[0]), np.amin(loc[1])-int(wide/1.5):np.amax(loc[1])+int(wide/1.5)],(32,64)), axis = 0))
                    cv2.putText(originalImage,classes[np.argmax(char[0])],(np.amax(loc[1])+int(wide/1.5),np.amax(loc[0])),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
                else:
                    cv2.rectangle(originalImage,(np.amin(loc[1]),np.amin(loc[0])),(np.amax(loc[1]),np.amax(loc[0])),(0,0,255),1)
                    char = model.predict(np.expand_dims(cv2.resize(originalImage[np.amin(loc[0]):np.amax(loc[0]), np.amin(loc[1]):np.amax(loc[1])],(32,64)) , axis = 0))
                    cv2.putText(originalImage,classes[np.argmax(char[0])],(np.amax(loc[1]),np.amax(loc[0])),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
                boxes = False
    cv2.imwrite('results_OCR/'+image,originalImage)
#     cv2.imshow('Black white image', blackAndWhiteImage)
#     cv2.imshow('Original image',originalImage)
#     cv2.imshow('Gray image', grayImage)
#     cv2.waitKey(0)
# cv2.destroyAllWindows()
#     cv2.imshow('test', blackAndWhiteImage)
#     cv2.imshow('test2',originalImage)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()