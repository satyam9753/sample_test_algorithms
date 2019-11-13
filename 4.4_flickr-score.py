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

        ##STORE PATCHES IN DIFFERENT DIRECTORY


class VGG_19():

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


