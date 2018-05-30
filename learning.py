# -*- coding: utf-8 -*-
"""
Created on Wed May 30 13:30:47 2018

@author: MaximdeMey
"""
#################################################################################
# training the model
#################################################################################
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense

import data_import

def create_model(input_shape):
    classifier = Sequential()

    # Step 0 downsampling
    classifier.add(AveragePooling2D(pool_size=(8, 8), input_shape=input_shape)) # to downsample to 8, 8

    # Step 1 = Convolution
    classifier.add(Conv2D(32, 5, 1, border_mode='valid', activation='relu' ))

    # Step 2 Max Pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Adding second convolutional layer
    classifier.add(Conv2D(32, 5, 1, border_mode='valid', activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Step 3 Flattening
    classifier.add(Flatten())
    # Step 4 Full Connection
    classifier.add(Dense(output_dim=4, activation='softmax'))

    # Compiling the CNN
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return classifier


def run():
    normalized_training_images, one_hot_labels, normalized_test_images = data_import.import_all_data()

    input_shape = normalized_training_images.shape[1:]

    model = create_model(input_shape)

    model.fit(normalized_training_images, one_hot_labels, epochs=10, batch_size=32)

# learning the model
classifier.fit_generator(training_set,
                        steps_per_epoch=8000,
                        epochs=25,
                        validation_data=test_set,
                        validation_steps=2000)
#################################################################################
###### Part 3 - Making new prediction
#################################################################################
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/mapmetjesinglefotos/defoto.jpg', target_size = (64,64))
test_image = image.img_to_array(test_image)

#  the prediciton requiers a fourth dimension The batch size. Even if we only have one picture we need to enter this dim.
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)

training_set.class_indices