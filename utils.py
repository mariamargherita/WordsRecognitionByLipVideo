
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