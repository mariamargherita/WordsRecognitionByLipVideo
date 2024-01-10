
from data_handling_nn import data_load_rf, data_feed_rf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree, ensemble
from sklearn.metrics import accuracy_score, confusion_matrix

'''
To run just once the first time we are loading the data to fill .pickle files. 
flattened_matrices, labels = data_load_rf("data/")
'''
flattened_matrices, labels = data_feed_rf()

X_train, X_val, y_train, y_val = train_test_split(flattened_matrices, labels, test_size=0.33, random_state=1)


clf = ensemble.RandomForestClassifier()

print("Train model")
clf.fit(X_train, y_train)

print("Compute predictions")
predicted = clf.predict(X_val)

print("Accuracy: ", accuracy_score(y_val, predicted))