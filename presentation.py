import pandas as pd
import numpy as np
import config

def save_predictions(input_file_path, output_file_path, one_hot_predictions):
    output_csv = pd.read_csv(input_file_path)

    labels = convert_one_hot_predictions_to_labels(one_hot_predictions)

    output_csv.class_numeric = labels

    output_csv.to_csv(output_file_path)


def convert_one_hot_predictions_to_labels(one_hot_predictions):

    labels = one_hot_predictions.argmax(1) + 1

    return labels