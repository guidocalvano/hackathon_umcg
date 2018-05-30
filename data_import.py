import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import sys

TRAINING_DATA_FILE_PATH = './data/DDSW_Hackathon_UMCG/case_1/train_data/case1_train_labels.csv'
TRAINING_IMAGE_DATA_PATH = './data/DDSW_Hackathon_UMCG/case_1/train_data/images/'

TEST_DATA_FILE_PATH = './data/DDSW_Hackathon_UMCG/case_1/test_data/case1_test_submission_waar_is_de_koffie.csv'
TEST_IMAGE_DATA_PATH = './data/DDSW_Hackathon_UMCG/case_1/test_data/images/'


corrupt_training_images = [2613]
corrupt_test_images = []

def load_training_set(file_path, image_data_path, corrupt_indices):
    input_to_label_csv = pd.read_csv(file_path)

    input_to_label_csv = input_to_label_csv.drop(corrupt_indices, axis=0)
    input_to_label_csv.index = range(input_to_label_csv.index.shape[0])
    # input_to_label_csv = input_to_label_csv.reset_index()
    # get relevant data
    labels = input_to_label_csv.iloc[:, 2]
    file_names = input_to_label_csv.iloc[:, 0]
    image_file_paths = image_data_path + file_names

    image_tensor = load_image_tensor(image_file_paths)

    # file_path_list to image_tensor

    # labels to one_hot_encoding

    # return image_tensor and labels, one hot encoded as numpy arrays

    one_hot_encoded_labels = pd.get_dummies(labels - 1)

    return image_tensor, labels

def load_test_set(file_path, image_data_path, corrupt_indices=[]):
    input_to_label_csv = pd.read_csv(file_path)

    input_to_label_csv = input_to_label_csv.drop(corrupt_indices, axis=0)
    input_to_label_csv.index = range(input_to_label_csv.index.shape[0])
    # input_to_label_csv = input_to_label_csv.reset_index()
    # get relevant data
    file_names = input_to_label_csv.iloc[:, 0]
    image_file_paths = image_data_path + file_names

    image_tensor = load_image_tensor(image_file_paths)

    return image_tensor

def load_image_tensor(image_file_paths):

    image_array_list = []

    for i in range(image_file_paths.shape[0]):
        next_image = load_img(image_file_paths[i])
        image_array = np.array(next_image)
        image_array_without_duplicate_channels = image_array[:, :, 0]
        shape_with_batch_dim = np.concatenate([np.array([1]), np.array(image_array_without_duplicate_channels.shape)])
        image_array_for_concatenation = image_array_without_duplicate_channels.reshape(shape_with_batch_dim)

        image_array_list.append(image_array_for_concatenation)

        if not (image_array_for_concatenation.shape == np.array([1, 288, 288])).all():
            print('problematic image')
            print(i)


    image_tensor = np.concatenate(image_array_list)

    return image_tensor

def normalize_data(data):
    mean = np.mean(data)
    centered_data = data - mean
    standard_deviation = np.std(centered_data) + sys.float_info.epsilon

    standardized_data = centered_data / standard_deviation

    return standardized_data, mean, standard_deviation

def normalize_test_images(data, mean, standard_deviation):
    centered_data = data - mean

    standardized_data = centered_data / standard_deviation

    return standardized_data

def import_all_data():
    training_images, one_hot_labels = load_training_set(TRAINING_DATA_FILE_PATH, TRAINING_IMAGE_DATA_PATH, corrupt_training_images)
    normalized_training_images, mean, standard_deviation = normalize_data(training_images)

    test_images = load_test_set(TEST_DATA_FILE_PATH, TEST_IMAGE_DATA_PATH, corrupt_test_images)
    normalized_test_images = normalize_test_images(test_images, mean, standard_deviation)

    return normalized_training_images, one_hot_labels, normalized_test_images




