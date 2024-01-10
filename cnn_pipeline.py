
from data_handling_nn import (data_load_nn, data_feed_nn, binSample, process_training_set, process_test_set,
                              test_data_feed_nn)
from utils import plot_sample_snapshot, visualizeSample, plot_history, prediction
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from cnn_model import convLSTM
import random
from datetime import datetime
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

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
'''
train_data_nn, labels_nn, label_dict_nn = data_feed_nn()

print("------------- Perform data split ----------------")

X_train, X_val, y_train, y_val = train_test_split(train_data_nn, labels_nn, test_size=0.1, random_state=1)

print("------------- Build, train and evaluate model ----------------")

random.seed(123)

timestamp = datetime.now()
epochs = 50
batch_size = 32
dropout = 0.5 # 0.3, 0.5 -> 0.78 val acc

model = convLSTM(X_train, y_train, X_val, y_val, timestamp, dropout=dropout)
train_history = model.train(epochs, batch_size)

plot_history(train_history)



'''
# To run just once the first time we are loading the data to fill .pickle files.
TIME_BINS = 8
RESIZE = 0.6

test_set_imgs = process_test_set(test_set, TIME_BINS, RESIZE)
'''
test_set_imgs = test_data_feed_nn()

print("------------- Make prediction on test set ----------------")

best_model = load_model(filepath='checkpoints/cp-best.model')
test_results = best_model.predict(test_set_imgs)

model_test_accuracy = prediction(test_results, label_dict_nn)

print(f"Model accuracy on test set: {model_test_accuracy}")


print("------------- End of pipeline ----------------")