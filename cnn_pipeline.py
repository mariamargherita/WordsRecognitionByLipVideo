
from data_handling_nn import (data_load_nn, data_feed_nn, binSample, process_training_set, process_test_set,
                              test_data_feed_nn)
from utils import plot_sample_snapshot, visualizeSample, plot_history, prediction
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from cnn_model import convLSTM, full_convLSTM
import random
from datetime import datetime
from tensorflow.keras.models import load_model


print("------------- Load and preprocess data ----------------")

train_set, test_set = data_load_nn("data/")

test = train_set['Ecole'][3]
test = pd.read_csv(test)

plot_sample_snapshot(test)

ret = binSample(test, 10, 0.25)
visualizeSample(ret)

'''
# To run just once the first time we are loading the data to fill .pickle files.
TIME_BINS = 8
RESIZE = 0.6

train_data_nn, labels_nn, label_dict_nn = process_training_set(train_set, TIME_BINS, RESIZE)
train_data_nn, labels_nn, label_dict_nn = process_training_set_denoised(train_set, TIME_BINS, RESIZE)
'''
train_data_nn, labels_nn, label_dict_nn = data_feed_nn()
# train_data_nn, labels_nn, label_dict_nn = denoised_data_feed_nn()

print("------------- Perform data split ----------------")

X_train, X_val, y_train, y_val = train_test_split(train_data_nn, labels_nn, test_size=0.1, random_state=1)

print("------------- Build, train and evaluate model ----------------")

random.seed(123)

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
epochs = 50
batch_size = 32
dropout = 0.3

model = convLSTM(X_train, y_train, X_val, y_val, timestamp, dropout=dropout)
train_history = model.train(epochs, batch_size)

plot_history(train_history.history)

'''
# To run just once the first time we are loading the data to fill .pickle files.
TIME_BINS = 8
RESIZE = 0.6

test_set_imgs = process_test_set(test_set, TIME_BINS, RESIZE)
test_set_imgs = process_test_set_denoised(test_set, TIME_BINS, RESIZE)
'''

test_set_imgs = test_data_feed_nn()

print("------------- Make prediction on test set ----------------")

best_model = load_model(filepath='checkpoints/cp-best-20240112114125.model')
test_results = best_model.predict(test_set_imgs)

model_test_accuracy = prediction(test_set, test_results, label_dict_nn)

print(f"Model accuracy on test set: {model_test_accuracy}")


print("Train model with best parameters on full training set")

X_train, y_train = train_data_nn, labels_nn

random.seed(1234)

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
epochs = 10
batch_size = 32
dropout = 0.3

final_model = full_convLSTM(X_train, y_train, timestamp, dropout=dropout)
train_history = final_model.train(epochs, batch_size)

print("------------- Make prediction on test set ----------------")

test_results_full = train_history.model.predict(test_set_imgs)

model_test_accuracy_full = prediction(test_set, test_results_full, label_dict_nn)

print(f"Model accuracy on test set (model trained on full train set): {model_test_accuracy_full}")


print("------------- End of pipeline ----------------")