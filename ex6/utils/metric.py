import numpy as np
import zarr
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import relabel_sequential


def evaluate(predicted_segmentation, gt_file, show_info=False):
    """Evaluate prediction given a ground truth file.

    Args:
        predicted_segmentation:
            The instance segmentation.
        gt_file:
            The zarr file to get the ground truth from.
        show_info:
            whether to show additional information.

    Returns:
        ap, precision, recall, tp, fp, fn

    """
    if show_info:
        print("evaluating %s", gt_file)

    # read result file
    if show_info:
        print("prediction min %f, max %f, shape %s", np.min(predicted_segmentation), np.max(predicted_segmentation),
              predicted_segmentation.shape)
    pred_labels = np.squeeze(predicted_segmentation)

    if show_info:
        print("prediction shape %s", pred_labels.shape)

    # read ground truth data
    gt = zarr.open(gt_file, 'r')
    gt_labels = np.array(gt['volumes/gt_labels'])

    if show_info:
        print("gt min %f, max %f, shape %s", np.min(gt_labels), np.max(gt_labels), gt_labels.shape)
    gt_labels = np.squeeze(gt_labels)

    if show_info:
        print("gt shape %s", gt_labels.shape)

    return _evaluate_linear_sum_assignment(gt_labels, pred_labels, show_info)


def _evaluate_linear_sum_assignment(gt_labels, pred_labels, show_info=False):
    pred_labels_rel, _, _ = relabel_sequential(pred_labels)
    gt_labels_rel, _, _ = relabel_sequential(gt_labels)

    overlay = np.array([pred_labels_rel.flatten(),
                        gt_labels_rel.flatten()])
    if show_info:
        print("overlay shape relabeled %s", overlay.shape)

    # get overlaying cells and the size of the overlap
    overlay_labels, overlay_labels_counts = np.unique(
        overlay, return_counts=True, axis=1)
    overlay_labels = np.transpose(overlay_labels)

    # get gt cell ids and the size of the corresponding cell
    gt_labels_list, gt_counts = np.unique(gt_labels_rel, return_counts=True)
    gt_labels_count_dict = {}

    if show_info:
        print("%s %s", gt_labels_list, gt_counts)

    for (l, c) in zip(gt_labels_list, gt_counts):
        gt_labels_count_dict[l] = c

    # get pred cell ids
    pred_labels_list, pred_counts = np.unique(pred_labels_rel,
                                              return_counts=True)

    if show_info:
        print("%s %s", pred_labels_list, pred_counts)

    pred_labels_count_dict = {}
    for (l, c) in zip(pred_labels_list, pred_counts):
        pred_labels_count_dict[l] = c

    num_pred_labels = int(np.max(pred_labels_rel))
    num_gt_labels = int(np.max(gt_labels_rel))
    num_matches = min(num_gt_labels, num_pred_labels)
    iouMat = np.zeros((num_gt_labels + 1, num_pred_labels + 1),
                      dtype=np.float32)

    for (u, v), c in zip(overlay_labels, overlay_labels_counts):
        iou = c / (gt_labels_count_dict[v] + pred_labels_count_dict[u] - c)
        iouMat[v, u] = iou

    # remove background
    iouMat = iouMat[1:, 1:]

    th = 0.5
    if num_matches > 0 and np.max(iouMat) > th:
        costs = -(iouMat > th).astype(float) - iouMat / (2 * num_matches)
        gt_ind, pred_ind = linear_sum_assignment(costs)
        assert num_matches == len(gt_ind) == len(pred_ind)
        match_ok = iouMat[gt_ind, pred_ind] > th
        tp = np.count_nonzero(match_ok)
    else:
        tp = 0
    fp = num_pred_labels - tp
    fn = num_gt_labels - tp
    ap = tp / max(1, tp + fn + fp)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)

    return ap, precision, recall, tp, fp, fn


def running_mean(x, n):
    """Computes the running mean of a vector.

    Args:
        x:
            The vector.
        n:
            The window size to compute the mean for.

    Returns:
        The running mean.

    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)
