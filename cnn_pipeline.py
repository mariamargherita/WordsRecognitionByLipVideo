
from data_handling import data_load_nn, data_feed_nn, binSample, processTrainingSet
from utils import plot_sample_snapshot, visualizeSample
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree, ensemble
from sklearn.metrics import accuracy_score, confusion_matrix
from cnn_model import convLSTM
import random

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

train_data_nn, labels_nn, label_dict_nn = processTrainingSet(train_set, TIME_BINS, RESIZE)
'''
train_data_nn, labels_nn, label_dict_nn = data_feed_nn()

X_train, X_val, y_train, y_val = train_test_split(train_data_nn, labels_nn, test_size=0.1, random_state=1)

random.seed(123)

epochs = 50
batch_size = 32
dropout = 0.5

# change this to the fit thing so that the model get stored
model = convLSTM(X_train, y_train, dropout=dropout)
train_history = model.train(epochs, batch_size)

print("\n\n\nEvaluating the Model on the validation set")
model.model.evaluate(X_val, y_val)

print("End")