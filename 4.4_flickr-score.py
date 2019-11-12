import numpy as np
import os
import cv2
import pandas as pd
import re2
import statistics

#import keras
#from keras.models import Sequential
#from keras.layers.core import Flatten, Dense, Dropout
#from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
#from keras.layers import Activation
#from sklearn.datasets import load_sample_image
#from sklearn.feature_extraction import image



DATADIR = "/home/satyam/Desktop/personal/flickr"

class Dataset():

    j=1
    faves_score = list()

    for img in os.listdir(DATADIR):

        img_path = DATADIR + '/' + os.listdir(DATADIR)[img]
        image = cv2.imread(img_path)

        for file_name in img_path:
            
            faves = re2.findall("flickr_"+'j' + "_(\d+).jpg", file_name)
            if not faves: 
                continue

            j++
            faves_score.append(float(faves[0]))



    median = statistics.median(faves_score)



class Resize_images():
 
    for img in os.listdir(DATADIR): 

        img_path = DATADIR + '/' + os.listdir(DATADIR)[img]
        image = cv2.imread(img_path)
        resultant_image = cv2.resize(img, dsize=(720, 1280), interpolation=cv2.INTER_CUBIC)



class FFS_label():
    for i in (faves_score):

    	if faves_score[i] > median:
    		new_img_path = DATADIR + '/'+"H" + os.listdir(DATADIR)[i]                                              


    	else:
    		new_img_path = DATADIR + '/'+"L" + os.listdir(DATADIR)[i]


class create_patch():
    
    for img in os.listdir(DATADIR):

        img_patch = cv2.imread(new_img_path)
        patches = image.extract_patches_2d(img_patch, (224, 224), max_patches = 1)


