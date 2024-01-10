
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def create_polarity_matrix(row, matrix):
    """

    :param row:
    :param matrix:
    :return:
    """
    matrix[row[0], row[1]] += 1


def plot_sample_snapshot(test):
    """

    :param test:
    :return:
    """
    x_dim = test.x.max() + 1
    y_dim = test.y.max() + 1
    print(x_dim, y_dim)

    snapshot = test.loc[test.time == 700000]
    scatter = test.groupby(by='time').count().reset_index(False)

    plt.scatter(scatter.time, scatter.x)
    plt.title("Events per timestamp of selected sample")
    plt.savefig("plots/sample_snapshot.png")
    plt.close()


def visualizeSample(sample, cmap ="binary"):
    """

    :param sample:
    :param cmap:
    :return:
    """
    plt.rcParams["figure.figsize"] = (20, 8)
    plt.title('Example Images of Event Timestamps Bucketed')
    plt.imshow(np.concatenate([l for l in sample], axis = 1), cmap = cmap)
    plt.savefig("plots/sample_visualization.png")
    plt.close()


def plot_history(history):
    """
    This function plots metrics obtained from models.
    Args:
        history: history of trained model.

    Returns: plots of training loss, validation loss, accuracy and validation accuracy.

    """
    train_loss = history['loss']
    val_loss = history['val_loss']
    train_acc = history['accuracy']
    val_acc = history['val_accuracy']

    # Loss
    plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig(f"plots/loss_vs_valloss.png")

    # Accuracy
    plt.figure()
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(f"plots/acc_vs_valacc.png")


def accuracy_fn(boolean_result, y_pred):
    """
    This function computes model accuracy.
    :param y_true: actual target
    :param y_pred: predicted target
    :return: model accuracy
    """

    correct = np.array(boolean_result).sum()
    accuracy = round((correct / len(y_pred)), 4)
    return accuracy


def prediction(test_set, test_results, label_dict):
    """

    :param test_results:
    :param label_dict:
    :return:
    """
    y_pred = [' ' for r in range(len(test_set))]
    for i, path in enumerate(test_set):
        test_sample_id = int(path.split('/')[3].split('.')[0]) # get the test sample id from the file path name
        # use the smaple id to fill in the corresponding index in the test_results_decoded list
        y_pred[test_sample_id] = label_dict[test_results[i].argmax()]

    y_true = list(pd.read_csv('solution.csv', names=['id', 'label']).label[1:])

    boolean_result = []
    for t, p in zip(y_true, y_pred):
        if t == p:
            boolean_result.append(True)
        else:
            boolean_result.append(False)

    test_accuracy = accuracy_fn(boolean_result, y_pred)

    return test_accuracy