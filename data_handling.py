
import os
import pandas as pd
import pickle
from utils import create_polarity_matrix
import numpy as np
import csv


def data_load(path_dir):
    """

    :return:
    """
    train_set = {}
    test_set = []

    for dirname, _, filenames in os.walk(path_dir):
        for filename in filenames:
            if dirname == path_dir + "test10/test10":
                test_set.append(f"{dirname}/"+filename)
            if path_dir + "train10/train10/" in dirname:
                stem_len = len(path_dir + "train10/train10/")
                class_label = dirname[stem_len:]
                if class_label in train_set.keys():
                    train_set[class_label].append(f"{dirname}/"+filename)
                else:
                    train_set[class_label] = [f"{dirname}/"+filename]

    print(f"The length of the test set is {len(test_set)}")
    print(f"The keys to access path files for the training set are {len(train_set)}")
    for k in train_set.keys():
        print(f"\t{k} has {len(train_set[k])} samples")

    return train_set, test_set


def binSample(sample, time_bins, resize_scale=0.5, size=[640, 480]):
    """

    :param sample:
    :param time_bins:
    :param resize_scale:
    :param size:
    :return:
    """
    temp = sample.copy()
    bin_size = int(sample.time.max() / time_bins) + 1
    bin_ranges = list(range(-1, sample.time.max() + bin_size, bin_size))
    temp['time_bin'] = pd.cut(temp['time'], bins=bin_ranges, labels=False)

    x_dim = size[0]
    y_dim = size[1]

    x_dim_bin = int(x_dim * resize_scale) + 1
    y_dim_bin = int(y_dim * resize_scale) + 1

    x_bin_size = int(x_dim / x_dim_bin) + 1
    x_bin_ranges = list(range(-1, x_dim, x_bin_size))

    temp['xbin'] = pd.cut(temp['x'], bins=x_bin_ranges, labels=False)
    temp['ybin'] = pd.cut(temp['y'], bins=x_bin_ranges, labels=False)

    bin_group_events = temp.groupby(by=['time_bin', 'xbin', 'ybin'])['polarity'].sum().reset_index()

    #     return bin_group_events
    layers = [np.zeros([y_dim_bin, x_dim_bin]) for x in range(time_bins)]
    for s in bin_group_events.iterrows():
        i, x, y, polarity = s[1]
        i, x, y, polarity = int(i), int(x), int(y), int(polarity)
        layers[i][y][x] += polarity

    return layers


def processTrainingSet(train_set, time_bins=10, resize=0.5):
    """

    :param train_set:
    :param time_bins:
    :param resize:
    :return:
    """
    label_dict = list(train_set.keys())
    labels = []
    train_data = []

    for i, k in enumerate(train_set.keys()):
        print(f"Building Class {i + 1}/10 with {time_bins} bins \t Resizing: {resize}")
        for s in train_set[k]:
            with open(s, newline='') as f:
                reader = csv.reader(f)
                row1 = next(reader)
                if 'x' not in row1:
                    s = pd.read_csv(s, names=['x', 'y', 'polarity', 'time'])
                else:
                    s = pd.read_csv(s)
            bucketed_sample = binSample(s, time_bins, resize)
            train_data.append(bucketed_sample)
            sample_label = [0 for x in range(len(label_dict))]
            sample_label[label_dict.index(k)] = 1
            labels.append(sample_label)

    train_data = np.array(train_data)
    height, width = train_data[0].shape[1], train_data[0].shape[2]
    train_data = train_data.reshape((len(train_data), time_bins, height, width, 1))

    return train_data, np.array(labels), label_dict


def data_load2(path_dir):
    """

    :return:
    """
    labels = []
    matrices = []

    for dirname, _, filenames in os.walk(path_dir + 'train10/'):

        for filename in filenames:
            labels.append(dirname.rsplit('/', 1)[1])
            pronounced_word = pd.read_csv(os.path.join(dirname, filename))
            width = (pronounced_word.iloc[:, 0] + 1).max()
            height = (pronounced_word.iloc[:, 1] + 1).max()
            words_matrix = np.zeros((width, height), dtype=int)
            pronounced_word_np = pronounced_word.to_numpy()
            np.apply_along_axis(create_polarity_matrix, 1, pronounced_word_np, words_matrix)
            normalized_words_matrix = words_matrix / np.amax(words_matrix)
            matrices.append(normalized_words_matrix)

    with open('labels.pickle', 'wb') as f:
        pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)

    with open('matrices.pickle', 'wb') as f:
        pickle.dump(matrices, f, pickle.HIGHEST_PROTOCOL)

    flattened_matrices = []
    for matrix in matrices:
        flattened_matrices.append(matrix.flatten()[:214])

    return flattened_matrices, labels
