
import os
import pandas as pd
from utils import create_polarity_matrix
import numpy as np
import pickle


def data_load_rf(path_dir):
    """
    This function loads data and labels from directory.
    It creates a polarity matrix (i.e. flags where there is intensity changes in every file by adding a +1 at every pixel
    coordinate where this happens), normalizes the matrix, flattens it and returns flattened matrix and respective
    labels (i.e. words pronounced).
    :param path_dir: directory path
    :return: flattened matrix and respective labels
    """
    labels = []
    matrices = []

    print("---------- Start data load ----------")

    j = 1
    for dirname, _, filenames in os.walk(path_dir + 'train10/train10/'):
        print(f"Processing directory {j} over 11: " + dirname)
        total = len(filenames)
        i = 1
        for filename in filenames:
            if filename != '.DS_Store':
                print(f'Iteration {i} over {total}')
                labels.append(dirname.rsplit('/', 1)[1])
                pronounced_word = pd.read_csv(os.path.join(dirname, filename))
                width = (pronounced_word.iloc[:, 0] + 1).max()
                height = (pronounced_word.iloc[:, 1] + 1).max()
                words_matrix = np.zeros((width, height), dtype=int)
                pronounced_word_np = pronounced_word.to_numpy()
                np.apply_along_axis(create_polarity_matrix, 1, pronounced_word_np, words_matrix)
                normalized_words_matrix = words_matrix / np.amax(words_matrix)
                matrices.append(normalized_words_matrix)
                i = i + 1
        j = j + 1

    print("---------- Start flattening matrices ----------")

    flattened_matrices = []
    for matrix in matrices:
        flattened_matrices.append(matrix.flatten()[:214])

    print("---------- Store data ----------")

    with open('labels.pickle', 'wb') as f:
        pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)

    with open('flattened_matrices.pickle', 'wb') as f:
        pickle.dump(flattened_matrices, f, pickle.HIGHEST_PROTOCOL)

    return flattened_matrices, labels


def data_feed_rf():
    """
    This function loads the data we extracted with data_load_rf.
    :return: flattened_matrices, labels
    """
    # Load data
    with open('labels.pickle', 'rb') as f:
        labels = pickle.load(f)

    with open('flattened_matrices.pickle', 'rb') as f:
        flattened_matrices = pickle.load(f)

    return flattened_matrices, labels