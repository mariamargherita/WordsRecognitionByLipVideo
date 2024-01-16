
from data_handling_rf import data_load_rf, data_feed_rf
import pandas as pd
import numpy as np
from scipy.stats import randint
from sklearn.model_selection import train_test_split
from sklearn import tree, ensemble
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

'''
To run just once the first time we are loading the data to fill .pickle files. 
flattened_matrices, labels = data_load_rf("data/")
'''
flattened_matrices, labels = data_feed_rf()

X_train, X_val, y_train, y_val = train_test_split(flattened_matrices, labels, test_size=0.33, random_state=1)

print("---------------- Set up grid search ----------------")

# Create the random grid
random_grid = {'n_estimators': np.arange(10, 500, step=50),
               'max_features': randint(1,7) + ['auto', 'sqrt', None],
               'max_depth': list(np.arange(10, 100, step=10)) + [None],
               'min_samples_split': np.arange(2, 10, step=2),
               'min_samples_leaf': randint(1,4),
               'bootstrap': [True, False],
               'criterion': ['gini','entropy']}

print("---------------- Fit grid search ----------------")

rf = ensemble.RandomForestClassifier()

rf_random = RandomizedSearchCV(estimator=rf,
                               param_distributions=random_grid,
                               n_iter=100,
                               cv=3,
                               verbose=2,
                               random_state=42,
                               n_jobs=-1)

print("---------------- Fit grid search ----------------")

rf_random.fit(X_train, y_train)

print(f" The grid search best parameters are {rf_random.best_params_}")

print("---------------- Evaluate best model on validation set ----------------")

best_rf = rf_random.best_estimator_
predicted = best_rf.predict(X_val)

print("Accuracy: ", accuracy_score(y_val, predicted))