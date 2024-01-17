
import os
import pandas as pd
import numpy as np
import csv
import pickle
from tensorflow.keras import Sequential, layers


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

    y_bin_size = int(y_dim / y_dim_bin) + 1
    y_bin_ranges = list(range(-1, y_dim, y_bin_size))

    temp['xbin'] = pd.cut(temp['x'], bins=x_bin_ranges, labels=False)
    temp['ybin'] = pd.cut(temp['y'], bins=y_bin_ranges, labels=False)

    bin_group_events = temp.groupby(by=['time_bin', 'xbin', 'ybin'])['polarity'].sum().reset_index()

    layers = [np.zeros([y_dim_bin, x_dim_bin]) for x in range(time_bins)]
    for s in bin_group_events.iterrows():
        i, x, y, polarity = s[1]
        i, x, y, polarity = int(i), int(x), int(y), int(polarity)
        layers[i][y][x] += polarity

    return layers


def process_training_set(train_set, time_bins=10, resize=0.5):
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

    print("---------- Store data ----------")

    with open('labels_nn.pickle', 'wb') as f:
        pickle.dump(np.array(labels), f, pickle.HIGHEST_PROTOCOL)

    with open('train_data_nn.pickle', 'wb') as f:
        pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)

    with open('label_dict_nn.pickle', 'wb') as f:
        pickle.dump(label_dict, f, pickle.HIGHEST_PROTOCOL)

    return train_data, np.array(labels), label_dict


def data_feed():
    """
    This function loads the data we extracted with process_training_set.
    :return: train_data, labels, label_dict
    """
    # Load data
    with open('labels_nn.pickle', 'rb') as f:
        labels_nn = pickle.load(f)

    with open('train_data_nn.pickle', 'rb') as f:
        train_data_nn = pickle.load(f)

    with open('label_dict_nn.pickle', 'rb') as f:
        label_dict_nn = pickle.load(f)

    return train_data_nn, labels_nn, label_dict_nn


def fft_denoise(event_df_col, alpha=0.05):
    '''

    This function takes in a dataframe and a column name, and returns a denoised version of the column.

    Inputs:
    event_df_col: column of the dataframe to be denoised
    alpha: fraction of coefficients to be zeroed out

    Output:
    denoised_data: denoised version of the column

    '''

    fft_result = np.fft.fft(event_df_col)

    # we set to zero whatever is outside the 95th percentile (we zero out the highest 5%)
    threshold = np.percentile(np.abs(fft_result), 100 * (1 - alpha))
    fft_result[np.abs(fft_result) < threshold] = 0

    denoised_data = np.fft.ifft(fft_result)

    return denoised_data.real


def process_training_set_denoised(train_set, time_bins=10, resize=0.5):
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
                    # remove noise
                    s['x'] = fft_denoise(s['x'])
                    s['y'] = fft_denoise(s['y'])
                else:
                    s = pd.read_csv(s)
                    # remove noise
                    s['x'] = fft_denoise(s['x'])
                    s['y'] = fft_denoise(s['y'])
            bucketed_sample = binSample(s, time_bins, resize)
            train_data.append(bucketed_sample)
            sample_label = [0 for x in range(len(label_dict))]
            sample_label[label_dict.index(k)] = 1
            labels.append(sample_label)

    train_data = np.array(train_data)
    height, width = train_data[0].shape[1], train_data[0].shape[2]
    train_data = train_data.reshape((len(train_data), time_bins, height, width, 1))

    print("---------- Store data ----------")

    with open('dlabels_nn.pickle', 'wb') as f:
        pickle.dump(np.array(labels), f, pickle.HIGHEST_PROTOCOL)

    with open('dtrain_data_nn.pickle', 'wb') as f:
        pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)

    with open('dlabel_dict_nn.pickle', 'wb') as f:
        pickle.dump(label_dict, f, pickle.HIGHEST_PROTOCOL)

    return train_data, np.array(labels), label_dict


def denoised_data_feed():
    """
    This function loads the data we extracted with process_training_set_denoised.
    :return: train_data, labels, label_dict
    """
    # Load data
    with open('dlabels_nn.pickle', 'rb') as f:
        labels_nn = pickle.load(f)

    with open('dtrain_data_nn.pickle', 'rb') as f:
        train_data_nn = pickle.load(f)

    with open('dlabel_dict_nn.pickle', 'rb') as f:
        label_dict_nn = pickle.load(f)

    return train_data_nn, labels_nn, label_dict_nn


def process_test_set(test_set, time_bins, resize):
    """

    :param test_set:
    :param time_bins:
    :param resize:
    :return:
    """
    test_set_imgs = []
    for i, img in enumerate(test_set):
        if i % 5 == 0:
            print(f"{i}/{len(test_set)}\t {img}")
        with open(img, newline='') as f:
            reader = csv.reader(f)
            row1 = next(reader)
            if 'x' not in row1:
                img = pd.read_csv(img, names=['x', 'y', 'polarity', 'time'])
            else:
                img = pd.read_csv(img)
            bucketed_sample = binSample(img, time_bins, resize)
            test_set_imgs.append(bucketed_sample)

    test_set_imgs = np.array(test_set_imgs)
    height, width = test_set_imgs[0].shape[1], test_set_imgs[0].shape[2]
    test_set_imgs = test_set_imgs.reshape(len(test_set), time_bins, height, width, 1)

    print("---------- Store data ----------")

    with open('test_data_nn.pickle', 'wb') as f:
        pickle.dump(test_set_imgs, f, pickle.HIGHEST_PROTOCOL)

    return test_set_imgs


def process_test_set_denoised(test_set, time_bins, resize):
    """

    :param test_set:
    :param time_bins:
    :param resize:
    :return:
    """
    test_set_imgs = []
    for i, img in enumerate(test_set):
        if i % 5 == 0:
            print(f"{i}/{len(test_set)}\t {img}")
        with open(img, newline='') as f:
            reader = csv.reader(f)
            row1 = next(reader)
            if 'x' not in row1:
                img = pd.read_csv(img, names=['x', 'y', 'polarity', 'time'])
                # remove noise
                img['x'] = fft_denoise(img['x'])
                img['y'] = fft_denoise(img['y'])
            else:
                img = pd.read_csv(img)
                # remove noise
                img['x'] = fft_denoise(img['x'])
                img['y'] = fft_denoise(img['y'])
            bucketed_sample = binSample(img, time_bins, resize)
            test_set_imgs.append(bucketed_sample)

    test_set_imgs = np.array(test_set_imgs)
    height, width = test_set_imgs[0].shape[1], test_set_imgs[0].shape[2]
    test_set_imgs = test_set_imgs.reshape(len(test_set), time_bins, height, width,1)

    print("---------- Store data ----------")

    with open('test_data_nn.pickle', 'wb') as f:
        pickle.dump(test_set_imgs, f, pickle.HIGHEST_PROTOCOL)

    return test_set_imgs


def test_data_feed():
    """
    This function loads the test data we extracted with process_test_set.
    :return: test_data
    """
    # Load data
    with open('test_data_nn.pickle', 'rb') as f:
        test_data_nn = pickle.load(f)

    return test_data_nn







