from __future__ import division, print_function

from enum import Enum

import mahotas
import numpy as np
import scipy.ndimage
import scipy.special
import torch
from skimage import measure

from .mean_shift import MeanShift


def watershed(surface, markers, fg, show_output=False):
    """Execute mahotas watershed algorithm.

    :param surface:
        The image to run watershed on.
    :param markers:
        initial markers (must be a labeled image, i.e., one where 0 represents
        the background and higher integers represent different regions)
    :param fg:
        foreground image
    :param show_output:
        whether to show output or not.
    :return:
        returns watershed of foreground image as integer (unint16)
    """
    # compute watershed
    ws = mahotas.cwatershed(surface, markers)

    # write watershed directly
    if show_output:
        print("watershed output: %s %s %f %f", ws.shape, ws.dtype, ws.max(), ws.min())

    # overlay fg and write
    wsFG = ws * fg
    if show_output:
        print("watershed (foreground only): %s %s %f %f", wsFG.shape, wsFG.dtype, wsFG.max(), wsFG.min())
    wsFGUI = wsFG.astype(np.uint16)

    return wsFGUI


def cluster_nearest_centroid(pred, centroids):
    """Cluster nearest centroids together.

    :param pred:
        The prediction vector
    :param centroids:
        The centroid point
    :return:
        The clustered centroids as float segmentation
    """
    # make channels last for broadcasted operations
    pred = np.moveaxis(pred, 0, -1)
    # get num centroids
    num_cens = centroids.shape[0]
    distances = np.zeros(list(pred.shape[:-1]) + [num_cens])
    for i in range(0, num_cens):
        distances[:, :, i] = np.linalg.norm(pred - centroids[i,], ord=2, axis=-1)
    segmentation = np.argmin(distances, axis=2) + 1
    return segmentation.astype(np.float32)


def _run_watershed(fg, seeds, surface, show_output=False):
    if np.count_nonzero(seeds) == 0 and show_output:
        print("no seed points found for watershed!")
    markers, cnt = scipy.ndimage.label(seeds)

    if show_output:
        print("num markers %s", cnt)

    # compute watershed
    labelling = watershed(surface, markers, fg)

    return labelling


def label(prediction, prediction_type, fg_thresh=0.9, seed_thresh=0.9, show_info=False):
    """Post-processing routine for various instance segmentation prediction types.

    :param prediction:
        The prediction of a neural network
    :param prediction_type:
        prediction type
    :param fg_thresh:
        foreground threshold (what predicted value counts as foreground. Usually smaller than seed threshold.)
    :param seed_thresh:
        seed threshold (when a predicted value is a seed)
    :param show_info:
        whether to show additional information
    :return:
        The instance segmentation and the surface that was used for post-processing.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if show_info:
        print("labelling")

    if prediction_type == PredictionType.TWO_CLASS:
        fg = 1.0 * (prediction[0] > fg_thresh)
        surface = 1.0 - prediction[0]

        seeds = (1 * (prediction[0] > seed_thresh)).astype(np.uint8)
        instance_segmentation = _run_watershed(fg, seeds, surface)
    elif prediction_type == PredictionType.AFFINITIES:
        # combine  components of affinities vector
        surface = 0.5 * (prediction[0] + prediction[1])
        # background pixel have affinity zero with everything
        # (including other bg pixel)
        fg = 1.0 * (surface > fg_thresh)
        surface = 1.0 - surface

        seeds = (1 * (prediction > seed_thresh)).astype(np.uint8)
        seeds = (seeds[0] + seeds[1])
        seeds = (seeds > 1).astype(np.uint8)
        instance_segmentation = _run_watershed(fg, seeds, surface)
    elif prediction_type == PredictionType.THREE_CLASS:
        # prediction[0] = bg
        # prediction[1] = inside
        # prediction[2] = boundary
        prediction = scipy.special.softmax(prediction, axis=0)
        fg = 1.0 * ((1.0 - prediction[0, ...]) > fg_thresh)
        surface = 1.0 - prediction[1, ...]
        seeds = (1 * (prediction[1, ...] > seed_thresh)).astype(np.uint8)
        instance_segmentation = _run_watershed(fg, seeds, surface)
    elif prediction_type == PredictionType.SDT:
        # distance transform in negative inside an instance
        # so negative values correspond to fg
        if fg_thresh > 0:
            print("fg threshold should be zero/negative")
        fg = 1.0 * (prediction < fg_thresh)
        fg = fg.astype(np.uint8)

        surface = prediction
        if seed_thresh > 0:
            print("surface/seed threshold should be negative")
        seeds = (1 * (surface < seed_thresh)).astype(np.uint8)
        instance_segmentation = _run_watershed(fg, seeds, surface)
    elif prediction_type == PredictionType.METRIC_LEARNING:
        fg = 1.0 * (prediction[0] > fg_thresh)
        emb = prediction[1:]
        emb *= fg
        ms = MeanShift(x=torch.Tensor(emb).to(device), bandwidth=2., chan=3, n_seeds=2000)
        C_ov = ms.forward()
        C_ov = torch.round(C_ov * 100) / 100
        C_ov = torch.unique(C_ov, dim=0)  # num_centroids x chan
        instance_segmentation = cluster_nearest_centroid(emb, C_ov.cpu().numpy()).astype(np.int32)
        instance_segmentation = instance_segmentation.astype(np.int32) * fg.astype(np.int32)
        instance_segmentation = measure.label(instance_segmentation, background=0)
        uni, counts = np.unique(instance_segmentation, return_counts=True)
        instance_segmentation[instance_segmentation == uni[counts < 50]] = 0
        surface = prediction
    else:
        raise ValueError("unknown prediction type")

    return instance_segmentation, surface


class PredictionType(Enum):
    """Enum class for different prediction types."""
    TWO_CLASS = 1
    THREE_CLASS = 2
    SDT = 3
    AFFINITIES = 4
    METRIC_LEARNING = 5


class UpsampleMode(Enum):
    """Enum class for different upsampling modes."""
    NEAREST = 1
    TRANSPOSED_CONV = 2
