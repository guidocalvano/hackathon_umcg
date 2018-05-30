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

classifier = Sequential()

# Step 1 = Convolution
classifier.add(Conv2D(32, 3, 3, border_mode = 'same', input_shape =(64, 64, 3), activation = 'relu' ))

# Step 2 Max Pooling
classifier.add(AveragePooling2D(pool_size = (2, 2)))

# Adding second convolutional layer
classifier.add(Conv2D(32, 3, 3, border_mode = 'same', input_shape =(64, 64, 3), activation = 'relu' ))
classifier.add(AveragePooling2D(pool_size = (2, 2)))

# Step 3 Flattening
classifier.add(Flatten())
# Step 4 Full Connection
classifier.add(Dense(output_dim = 128, activation = 'relu' ))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid' ))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#################################################################################
# Part 2 = Fitting the CNN to the images
#################################################################################
from keras.preprocessing.image import ImageDataGenerator
# dit hoeven we mogelijk buiten de zca_epsilon niet meer te doen.
train_datagen = ImageDataGenerator(featurewise_center=True,
                                             samplewise_center=False,
                                             featurewise_std_normalization=True,
                                             samplewise_std_normalization=False,
                                             zca_whitening=False,
                                             zca_epsilon=1e-06,
                                             rotation_range=0.0,
                                             width_shift_range=0.0,
                                             height_shift_range=0.0,
                                             brightness_range=None,
                                             shear_range=0.0,
                                             zoom_range=0.0,
                                             channel_shift_range=0.0,
                                             fill_mode='nearest',
                                             cval=0.0,
                                             horizontal_flip=False,
                                             vertical_flip=False,
                                             rescale=1./255,
                                             preprocessing_function=None,
                                             data_format=None,
                                             validation_split=0.0)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('train_data/images',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('test_data/images',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')
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