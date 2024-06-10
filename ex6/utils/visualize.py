import glob
import os
import random

import numpy as np
import zarr
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from .metric import running_mean
from .label import PredictionType


def plot_image(raw, labels, pred=None, prediction_type=PredictionType.TWO_CLASS):
    """Plot an image with the raw image, the ground truth and the prediction if any next to each other."""
    if prediction_type == PredictionType.AFFINITIES:
        labels = labels[0] + labels[1]

    fig = plt.figure(figsize=(12, 8))
    if pred is not None:
        num_plots = 3
    else:
        num_plots = 2

    fig.add_subplot(1, num_plots, 1)
    plt.imshow(np.squeeze(raw), cmap='gray')
    fig.add_subplot(1, num_plots, 2)
    plt.imshow(np.squeeze(labels), cmap='gist_earth')
    if pred is not None:
        fig.add_subplot(1, num_plots, 3)
        plt.imshow(np.squeeze(pred), cmap='gist_earth')

    plt.show()


def plot_random_image():
    """Plot a random image from the dataset."""
    fls = glob.glob(os.path.join("dsb2018", "train", "*.zarr"))
    fl = zarr.open(fls[random.randrange(len(fls))], 'r')
    raw = fl["volumes/raw"]
    labels = fl["volumes/gt_threeclass"]
    plot_image(raw, labels)


def plot_history(history, running_mean_window=9):
    """Plots the history of a training run.

    Args:
        history:
            The history to plot.
        running_mean_window:
            The window size to compute the running mean for.

    """
    loss = running_mean(history['loss'], running_mean_window)
    val_loss = running_mean(history['val_loss'], running_mean_window)
    acc = running_mean(history['binary_accuracy'], running_mean_window)
    val_acc = running_mean(history['val_binary_accuracy'], running_mean_window)
    epochs = len(loss)

    # figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # train and validation loss
    ax1.plot(range(0, epochs), loss, label='loss')
    ax1.plot(range(0, epochs), val_loss, label='val_loss')
    ax1.set_title('train and validation loss')
    ax1.legend(loc='upper right')

    # train and validation accuracy
    ax2.plot(range(0, epochs), acc, label='binary_accuracy')
    ax2.plot(range(0, epochs), val_acc, label='val_binary_accuracy')
    ax2.set_title('train and validation binary accuracy')
    ax2.legend(loc='lower right')

    plt.show()


def show_predictions(raw, gt, pred):
    """Show the raw image, the ground truth and the prediction next to each other.

    Args:
        raw:
            The raw image.
        gt:
            The ground truth.
        pred:
            The prediction.

    """
    thresh = 0.9
    max_values = np.max(pred[:, 0], axis=(1, 2))
    if np.any(max_values < thresh):
        print("Heads up: If prediction is below {} then the prediction map is shown.".format(thresh))
        print("Max predictions: {}".format(max_values))

    num_samples = pred.shape[0]
    fig, ax = plt.subplots(num_samples, 3, sharex=True, sharey=True, figsize=(12, num_samples * 4))
    for i in range(num_samples):
        ax[i, 0].imshow(raw[i, 0], aspect="auto")
        ax[i, 1].imshow(gt[i, 0], aspect="auto")
        # check for prediction threshold
        if np.sum(max_values[i]) < thresh:
            ax[i, 2].imshow(pred[i, 0], aspect="auto")
        else:
            ax[i, 2].imshow(pred[i, 0] >= thresh, aspect="auto")

    ax[0, 0].set_title("Input")
    ax[0, 1].set_title("Ground truth")
    ax[0, 2].set_title("Prediction")
    fig.tight_layout()

    plt.show()


def show_instance_segmentation(image, gt_labels, surface, instance_segmentation):
    """Show the instance segmentation results together with the input and surface next to each other.

    Args:
        image:
            The image to show.
        gt_labels:
            The ground truth labels.
        surface:
            The surface that was used for preprocessing.
        instance_segmentation:
            The instance segmentation result.

    """
    fig = plt.figure(figsize=(16, 8))

    ax = fig.add_subplot(1, 4, 1)
    ax.set_title("raw")
    plt.imshow(np.squeeze(image))

    ax = fig.add_subplot(1, 4, 2)
    ax.set_title("gt labels")
    plt.imshow(np.squeeze(1.0 - gt_labels))

    ax = fig.add_subplot(1, 4, 3)
    ax.set_title("prediction")
    plt.imshow(np.squeeze(1.0 - surface))

    ax = fig.add_subplot(1, 4, 4)
    ax.set_title("pred segmentation")
    plt.imshow(np.squeeze(instance_segmentation), cmap=get_random_colormap(), interpolation="none")

    plt.show()


def plot_receptive_field(image, fov):
    """Plot the receptive field of the network on top of the image.

    Args:
        image:
            The image to plot the receptive field on.
        fov:
            The field of view of the network.

    """
    image_s = np.squeeze(image)
    _ = plt.figure(figsize=(8, 8))
    plt.imshow(image_s, cmap='gray')
    xmin = image_s.shape[1] / 2 - fov / 2
    xmax = image_s.shape[1] / 2 + fov / 2
    ymin = image_s.shape[1] / 2 - fov / 2
    ymax = image_s.shape[1] / 2 + fov / 2
    plt.hlines(ymin, xmin, xmax, color="magenta", lw=3)
    plt.hlines(ymax, xmin, xmax, color="magenta", lw=3)
    plt.vlines(xmin, ymin, ymax, color="magenta", lw=3)
    plt.vlines(xmax, ymin, ymax, color="magenta", lw=3)
    plt.show()


def get_random_colormap():
    """Get a random colormap.

    Returns:
        A random colormap.

    """

    N = 256
    vals = np.ones((N, 4))
    vals[0, 0] = 0
    vals[0, 1] = 0
    vals[0, 2] = 0
    for n in range(1, N):
        vals[n, 0] = np.random.rand()
        vals[n, 1] = np.random.rand()
        vals[n, 2] = np.random.rand()
    rand_cmap = ListedColormap(vals)

    return rand_cmap
