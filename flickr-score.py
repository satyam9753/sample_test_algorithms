import numpy as np
import pandas as pd
import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation

from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image
import cv2


##### RESISIZING THE IMAGE #####

img = cv2.imread('your_image.jpg')
res = cv2.resize(img, dsize=(720, 1280), interpolation=cv2.INTER_CUBIC)

#############################################################################################

median = np.median(faves)

for i in (len):

	if FFS> median:
		quality = "high"

	else:
		quality = "low"

	df['QUALITY'] = quality    #adding extra column


##############################################################################################

img_patch = load_sample_image("image to be patched")
patches = image.extract_patches_2d(one_image, (224, 224))


##### VGG-19 style ######


model = Sequential()

model.add(Conv2D(num_features, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

model.add(Dense(num_labels, activation='relu'))
model.add(Activation('softmax'))