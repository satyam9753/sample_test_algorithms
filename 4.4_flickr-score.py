import numpy as np
import os
import sys
import cv2
import pandas as pd
import re
import statistics
import pickle

from PIL import Image
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, Adam
from datagen import DataGenerator as DataGenerator2
import datagen
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import copy
from sklearn.feature_extraction import image


#import keras
#from keras.models import Sequential
#from keras.layers.core import Flatten, Dense, Dropout
#from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
#from keras.layers import Activation
#from sklearn.datasets import load_sample_image
#from sklearn.feature_extraction import image



DATADIR = "/home/satyam/Desktop/personal/flickr"

def Resize_images(image):
 
    #for img in os.listdir(DATADIR): 
    #
    #    img_path = DATADIR + '/' + img
    #    image = cv2.imread(img_path)
    #    resultant_image = cv2.resize(img, dsize=(720, 1280), interpolation=cv2.INTER_CUBIC)
    x=[]
    for item in range(len(image)):
            imResize = copy.deepcopy(image[item])
            x.append(cv2.resize(imResize, (720,1080)))
            print(x[item])
    return x
            


def Dataset():

    faves_score = list()
    image=[]
    for img in os.listdir(DATADIR):
        #print(img)
        img_path = DATADIR + '/' + img
        image.append(cv2.imread(img_path))

        #print (os.listdir(DATADIR))
    for file_name in os.listdir(DATADIR):
            
        faves = re.findall(r"_0.(\d+).jpg", file_name)
        if not faves: 
          continue
        #print (faves[0])
        faves_score.append(float(faves[0]))                
    median = statistics.median(faves_score)
    print (median)
    print(img_path,image[0])
    k=Resize_images(image)
    return faves_score,k,median




#def FFS_label():
#    for i in (faves_score):
#
#    	if faves_score[i] > median:
#    		new_img_path = DATADIR + '/'+"H" + os.listdir(DATADIR)[i]                                              
#    	else:
#   		new_img_path = DATADIR + '/'+"L" + os.listdir(DATADIR)[i]


def create_patch(faves_score,imgs,median):
    for i in range(len(faves_score)):
        patches = image.extract_patches_2d(imgs[i], (224, 224), max_patches=1)[0]
        ##STORE PATCHES IN DIFFERENT DIRECTORY
        if (faves_score[i] > median):
            s="high"+str(i)
        else:
            s="low"+str(i)
        cv2.imwrite("/home/satyam/Desktop/personal/hl/"+s + '.jpg',patches)
    return patches



##### VGG-19 model to be done #####

def VGG_19():

        HEIGHT = 224
        WIDTH = 224

        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))

        def build_finetune_model(base_model, dropout, fc_layers, num_classes):
            for layer in base_model.layers:
                layer.trainable = False

            x = base_model.output
            x = Flatten()(x)
            for fc in fc_layers:
                # New FC layer, random init
                x = Dense(fc, activation='relu')(x) 
                x = Dropout(dropout)(x)

            # New softmax layer
            predictions = Dense(num_classes, activation='softmax')(x) 
            
            finetune_model = Model(inputs=base_model.input, outputs=predictions)

            return finetune_model

        class_list = ["Original","Tampered"]
        FC_LAYERS = [1024, 1024]
        NUM_EPOCHS = 10
        BATCH_SIZE = 4
        dropout = 0.5

        finetune_model = build_finetune_model(base_model, dropout=dropout, fc_layers=FC_LAYERS, num_classes=len(class_list))

        # directory = './drive/My Drive/newdata2'

        directory = '________________' ##DIRECTORY WHERE PATCHES ARE STORED

        train_generator,valid_generator,test_generator=datagen.getGenerators(directory,train_batch_size=8,valid_batch_size=1,test_batch_size=1,train_ratio=0.6,valid_ratio=0.2)

        # train_generator=DataGenerator2(directory, batch_size=BATCH_SIZE)


        adam = Adam(lr=0.00001)
        finetune_model.compile(adam, loss='binary_crossentropy', metrics=['accuracy'])

        filepath="./checkpoints/" + "MobileNetV2" + "_model_weights2.h5"
        if not os.path.exists("./checkpoints"):
            os.makedirs("./checkpoints")

        checkpoint = ModelCheckpoint(filepath, monitor="val_acc", verbose=1, save_best_only=True)
        callbacks_list = [checkpoint]

        history = finetune_model.fit_generator(train_generator, epochs=NUM_EPOCHS, workers=8,
                                               shuffle=True, callbacks=callbacks_list,validation_data=valid_generator,validation_freq=5,use_multiprocessing=True)

        # history = base_model.predict_generator(train_generator, workers=8)
        print(finetune_model.evaluate_generator(test_generator))

        # testing
        # finetune_model.predict_generator() 


if __name__=='__main__':
    fav, imgs, median = Dataset()
    a=create_patch(Dataset())
