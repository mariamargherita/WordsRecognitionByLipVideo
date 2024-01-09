
from data_handling import data_load, binSample, processTrainingSet
from utils import plot_sample_snapshot, visualizeSample
import pandas as pd

train_set, test_set = data_load("data/")

test = train_set['Ecole'][3]
test = pd.read_csv(test)

plot_sample_snapshot(test)

ret = binSample(test, 10, 0.25)
visualizeSample(ret)



TIME_BINS = 4
RESIZE = 0.2

train_data, labels, label_dict = processTrainingSet(train_set, TIME_BINS, RESIZE)

#flattened_matrices, labels = data_load2("data/")

print("End")