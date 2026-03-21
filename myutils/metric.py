
from skimage import measure

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc,average_precision_score,accuracy_score
# REMARK: CODE WAS TAKEN FROM https://github.com/eliahuhorwitz/3D-ADS/blob/main/utils/au_pro_util.py

"""
Code based on the official MVTec 3D-AD evaluation code found at
https://www.mydrive.ch/shares/45924/9ce7a138c69bbd4c8d648b72151f839d/download/428846918-1643297332/evaluation_code.tar.xz
Utility functions that compute a PRO curve and its definite integral, given
pairs of anomaly and ground truth maps.
The PRO curve can also be integrated up to a constant integration limit.
"""
from bisect import bisect

import numpy as np
from scipy.ndimage.measurements import label

from myutils.cfm_metrics_utils import calculate_au_pro
class GroundTruthComponent:
    """
    Stores sorted anomaly scores of a single ground truth component.
    Used to efficiently compute the region overlap for many increasing thresholds.
    """

    def __init__(self, anomaly_scores):
        """
        Initialize the module.
        Args:
            anomaly_scores: List of all anomaly scores within the ground truth
                            component as numpy array.
        """
        # Keep a sorted list of all anomaly scores within the component.
        self.anomaly_scores = anomaly_scores.copy()
        self.anomaly_scores.sort()

        # Pointer to the anomaly score where the current threshold divides the component into OK / NOK pixels.
        self.index = 0

        # The last evaluated threshold.
        self.last_threshold = None

    def compute_overlap(self, threshold):
        """
        Compute the region overlap for a specific threshold.
        Thresholds must be passed in increasing order.
        Args:
            threshold: Threshold to compute the region overlap.
        Returns:
            Region overlap for the specified threshold.
        """
        if self.last_threshold is not None:
            assert self.last_threshold <= threshold

        # Increase the index until it points to an anomaly score that is just above the specified threshold.
        while self.index < len(self.anomaly_scores) and self.anomaly_scores[self.index] <= threshold:
            self.index += 1

        # Compute the fraction of component pixels that are correctly segmented as anomalous.
        return 1.0 - self.index / len(self.anomaly_scores)


def trapezoid(x, y, x_max=None):
    """
    This function calculates the definit integral of a curve given by x- and corresponding y-values.
    In contrast to, e.g., 'numpy.trapz()', this function allows to define an upper bound to the integration range by
    setting a value x_max.
    Points that do not have a finite x or y value will be ignored with a warning.
    Args:
        x:     Samples from the domain of the function to integrate need to be sorted in ascending order. May contain
               the same value multiple times. In that case, the order of the corresponding y values will affect the
               integration with the trapezoidal rule.
        y:     Values of the function corresponding to x values.
        x_max: Upper limit of the integration. The y value at max_x will be determined by interpolating between its
               neighbors. Must not lie outside of the range of x.
    Returns:
        Area under the curve.
    """

    x = np.array(x)
    y = np.array(y)
    finite_mask = np.logical_and(np.isfinite(x), np.isfinite(y))
    if not finite_mask.all():
        print(
            "WARNING: Not all x and y values passed to trapezoid are finite. Will continue with only the finite values."
        )
    x = x[finite_mask]
    y = y[finite_mask]

    # Introduce a correction term if max_x is not an element of x.
    correction = 0.0
    if x_max is not None:
        if x_max not in x:
            # Get the insertion index that would keep x sorted after np.insert(x, ins, x_max).
            ins = bisect(x, x_max)
            # x_max must be between the minimum and the maximum, so the insertion_point cannot be zero or len(x).
            assert 0 < ins < len(x)

            # Calculate the correction term which is the integral between the last x[ins-1] and x_max. Since we do not
            # know the exact value of y at x_max, we interpolate between y[ins] and y[ins-1].
            y_interp = y[ins - 1] + ((y[ins] - y[ins - 1]) * (x_max - x[ins - 1]) / (x[ins] - x[ins - 1]))
            correction = 0.5 * (y_interp + y[ins - 1]) * (x_max - x[ins - 1])

        # Cut off at x_max.
        mask = x <= x_max
        x = x[mask]
        y = y[mask]

    # Return area under the curve using the trapezoidal rule.
    return np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])) + correction


def collect_anomaly_scores(anomaly_maps, ground_truth_maps):
    """
    Extract anomaly scores for each ground truth connected component as well as anomaly scores for each potential false
    positive pixel from anomaly maps.

    Args:
        anomaly_maps: List of anomaly maps (2D numpy arrays) that contain a real-valued anomaly score at each pixel.
        ground_truth_maps: List of ground truth maps (2D numpy arrays) that contain binary-valued ground truth labels
                           for each pixel. 0 indicates that a pixel is anomaly-free. 1 indicates that a pixel contains
                           an anomaly.
    Returns:
        ground_truth_components: A list of all ground truth connected components that appear in the dataset.
            For each component, a sorted list of its anomaly scores is stored.
        anomaly_scores_ok_pixels: A sorted list of anomaly scores of all anomaly-free pixels of the dataset.
            This list can be used to quickly select thresholds that fix a certain false positive rate.
    """
    # Make sure an anomaly map is present for each ground truth map.
    assert len(anomaly_maps) == len(ground_truth_maps)

    # Initialize ground truth components and scores of potential fp pixels.
    ground_truth_components = []
    anomaly_scores_ok_pixels = np.zeros(len(ground_truth_maps) * ground_truth_maps[0].size)

    # Structuring element for computing connected components.
    structure = np.ones((3, 3), dtype=int)

    # Collect anomaly scores within each ground truth region and for all potential fp pixels.
    ok_index = 0
    for gt_map, prediction in zip(ground_truth_maps, anomaly_maps):
        # Compute the connected components in the ground truth map.
        labeled, n_components = label(gt_map, structure)

        # Store all potential fp scores.
        num_ok_pixels = len(prediction[labeled == 0])
        anomaly_scores_ok_pixels[ok_index : ok_index + num_ok_pixels] = prediction[labeled == 0].copy()
        ok_index += num_ok_pixels

        # Fetch anomaly scores within each GT component.
        for k in range(n_components):
            component_scores = prediction[labeled == (k + 1)]
            ground_truth_components.append(GroundTruthComponent(component_scores))

    # Sort all potential false positive scores.
    anomaly_scores_ok_pixels = np.resize(anomaly_scores_ok_pixels, ok_index)
    anomaly_scores_ok_pixels.sort()

    return ground_truth_components, anomaly_scores_ok_pixels


def compute_pro(anomaly_maps, ground_truth_maps, num_thresholds):
    """
    Compute the PRO curve at equidistant interpolation points for a set of anomaly maps with corresponding ground
    truth maps. The number of interpolation points can be set manually.

    Args:
        anomaly_maps: List of anomaly maps (2D numpy arrays) that contain a real-valued anomaly score at each pixel.
        ground_truth_maps: List of ground truth maps (2D numpy arrays) that contain binary-valued ground truth labels
            for each pixel. 0 indicates that a pixel is anomaly-free. 1 indicates that a pixel contains an anomaly.
        num_thresholds: Number of thresholds to compute the PRO curve.

    Returns:
        fprs: List of false positive rates.
        pros: List of correspoding PRO values.
    """
    # Fetch sorted anomaly scores.
    ground_truth_components, anomaly_scores_ok_pixels = collect_anomaly_scores(anomaly_maps, ground_truth_maps)
    # Select equidistant thresholds.
    threshold_positions = np.linspace(0, len(anomaly_scores_ok_pixels) - 1, num=num_thresholds, dtype=int)

    fprs = [1.0]
    pros = [1.0]
    for pos in threshold_positions:
        threshold = anomaly_scores_ok_pixels[pos]

        # Compute the false positive rate for this threshold.
        fpr = 1.0 - (pos + 1) / len(anomaly_scores_ok_pixels)

        # Compute the PRO value for this threshold.
        pro = 0.0
        for component in ground_truth_components:
            pro += component.compute_overlap(threshold)
        pro /= len(ground_truth_components)

        fprs.append(fpr)
        pros.append(pro)

    # Return (FPR/PRO) pairs in increasing FPR order.
    fprs = fprs[::-1]
    pros = pros[::-1]

    return fprs, pros

#
# def calculate_au_pro(gts, predictions, integration_limit=0.3, num_thresholds=100):
#     """
#     Compute the area under the PRO curve for a set of ground truth images and corresponding anomaly images.
#     Args:
#         gts:         List of tensors that contain the ground truth images for a single dataset object.
#         predictions: List of tensors containing anomaly images for each ground truth image.
#         integration_limit:    Integration limit to use when computing the area under the PRO curve.
#         num_thresholds:       Number of thresholds to use to sample the area under the PRO curve.
#     Returns:
#         au_pro:    Area under the PRO curve computed up to the given integration limit.
#         pro_curve: PRO curve values for localization (fpr,pro).
#     """
#     # Compute the PRO curve.
#     pro_curve = compute_pro(anomaly_maps=predictions, ground_truth_maps=gts, num_thresholds=num_thresholds)
#
#     # Compute the area under the PRO curve.
#     au_pro = trapezoid(pro_curve[0], pro_curve[1], x_max=integration_limit)
#     au_pro /= integration_limit
#
#     # Return the evaluation metrics.
#     return au_pro, pro_curve

def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc
def squeeze_all_dim(x):
    del_list = []
    for i, s in enumerate(x.shape):
        if s == 1:
            del_list.append(i)
    for layer in del_list[::-1]:
        x = x.squeeze(layer)
    return x
import torch
from torchmetrics.functional import auroc, average_precision, precision_recall_curve

def calculate_max_f1_torch(gt: torch.Tensor, scores: torch.Tensor):
    precision, recall, thresholds = precision_recall_curve(scores, gt, task="binary")
    f1s = 2 * precision * recall / (precision + recall + 1e-8)
    max_idx = torch.argmax(f1s)
    max_f1 = f1s[max_idx].item()

    recall_sorted, indices = torch.sort(recall)
    precision_sorted = precision[indices]
    aupr = torch.trapz(precision_sorted, recall_sorted).item()
    prec_at_max = precision[max_idx].item()
    rec_at_max = recall[max_idx].item()
    thresh = thresholds[max_idx].item() if max_idx < len(thresholds) else 1.0
    return max_f1, aupr, prec_at_max, rec_at_max, thresh


def cal_metric(gt_sp, gt_mask, pr_sp, pr_mask, shape=None):
    """
    gt_sp, pr_sp: tensors, shape (N,)
    gt_mask, pr_mask: tensors, shape (N, H, W)
    shape: (H, W)
    """

    # 如果 calculate_au_pro 你有 Torch 版本，直接替换下面调用
    pro = calculate_au_pro(gt_mask.cpu().numpy().reshape(-1, *shape), pr_mask.cpu().numpy().reshape(-1, *shape))

    # 展平
    gt_mask = gt_mask.flatten().long()
    pr_mask = pr_mask.flatten()

    spauc = auroc(pr_sp, gt_sp, task="binary").item()
    spap = average_precision(pr_sp, gt_sp, task="binary").item()
    spmaxf1, spaupr, _, _, _ = calculate_max_f1_torch(gt_sp, pr_sp)

    pixelauc = auroc(pr_mask, gt_mask, task="binary").item()
    pixelap = average_precision(pr_mask, gt_mask, task="binary").item()
    pixelmaxf1, pixelaupr, _, _, _ = calculate_max_f1_torch(gt_mask, pr_mask)

    return (spauc,  spap,spmaxf1, pixelauc, pixelmaxf1, pro[0][0],pixelap,)
