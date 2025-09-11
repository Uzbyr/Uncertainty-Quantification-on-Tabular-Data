import numpy as np
from numpy._typing import NDArray

def confidence_score(probs: NDArray):
    return np.max(-probs, axis=1)

def margin_score(probs: NDArray):
    sorted_probs = np.sort(probs, axis=1)
    return sorted_probs[:, -2] - sorted_probs[:, -1]

def entropy_score(probs: NDArray, eps = 1e-9):
    return -np.sum(probs * np.log(probs + eps), axis=1)

def nnl_score(probs: NDArray, true_labels: NDArray, eps = 1e-9):
    return -np.log(probs[np.arange(probs.shape[0]), true_labels] + eps)

def ri_score(probs: NDArray, eps = 1e-9):
    return -np.sum(np.log(probs + eps), axis=1)


def lac_conformal_score(probs: NDArray, true_labels: NDArray):
    """
    Compute the LAC conformal score for a batch of softmax score vectors and true labels.

    Parameters:
    - probs: 2D numpy array of shape (n_samples, num_classes), softmax probs for each sample
    - true_labels: 1D numpy array of shape (n_samples,), true class labels for each sample

    Returns:
    - conformal_scores: 1D numpy array of shape (n_samples,), LAC conformal probs for each sample
    """
    conformal_scores = 1 - probs[np.arange(probs.shape[0]), true_labels]
    return conformal_scores

def aps_conformal_score(probs: NDArray, true_labels: NDArray):
    """
    Compute the APS conformal score for a batch of softmax score vectors and true labels.

    Parameters:
    - probs: 2D numpy array of shape (n_samples, num_classes), softmax probs for each sample
    - true_labels: 1D numpy array of shape (n_samples,), true class labels for each sample

    Returns:
    - conformal_scores: 1D numpy array of shape (n_samples,), APS conformal probs for each sample
    """
    # Create a mask for each sample: probs >= true_score
    true_scores = probs[np.arange(probs.shape[0]), true_labels]
    mask = probs >= true_scores[:, np.newaxis]
    # Sum along the class axis
    conformal_scores = np.sum(probs * mask, axis=1)

    return conformal_scores

def compute_quantile(scores: NDArray, n: int, alpha = 0.1):
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    return np.quantile(scores, q_level, method="higher")

def lac_prediction_set(calibration_probs: NDArray, probs: NDArray, calibration_labels: NDArray, alpha = 0.1):
    n = calibration_labels.shape[0]
    cal_scores = 1 - calibration_probs[np.arange(calibration_probs.shape[0]), calibration_labels]
    # Get the score quantile

    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    qhat = np.quantile(cal_scores, q_level, method='higher')

    prediction_sets = probs >= (1 - qhat)
    return prediction_sets

def aps_prediction_set(calibration_probs: NDArray, probs: NDArray, calibration_labels: NDArray, alpha = 0.1):
    # Get scores. calib_X.shape[0] == calib_Y.shape[0] == n
    n = calibration_labels.shape[0]
    cal_order = calibration_probs.argsort(1)[:,::-1]
    # cal_sum = calibration_probs[np.arange(n)[:, None], cal_order].cumsum(axis=1)
    cal_sum = np.take_along_axis(calibration_probs, cal_order, axis=1).cumsum(axis=1)
    cal_scores = np.take_along_axis(cal_sum, cal_order.argsort(axis=1), axis=1)[range(n), calibration_labels]

    # Get the score quantile
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    qhat = np.quantile(cal_scores, q_level, method='higher')

    # Deploy (output=list of length n, each element is tensor of classes)
    test_order = probs.argsort(1)[:,::-1]
    test_sum = np.take_along_axis(probs,test_order,axis=1).cumsum(axis=1)
    prediction_sets = np.take_along_axis(test_sum <= qhat, test_order.argsort(axis=1), axis=1)
    return prediction_sets

def raps_prediction_set(calibration_probs: NDArray, test_probs: NDArray, calibration_labels: NDArray, alpha = 0.1, lam_reg=0.01, k_reg = 5, disallow_zero_sets = False, rand = True):
    # Set RAPS regularization parameters (larger lam_reg and smaller k_reg leads to smaller sets)
    # Set disallow_zero_sets to False in order to see the coverage upper bound hold
    # Set rand to True in order to see the coverage upper bound hold

    probs = np.concatenate([calibration_probs, test_probs], axis=0)
    k_reg = min(k_reg, probs.shape[1] - 1)
    reg_vec = np.array(k_reg * [0,] + (probs.shape[1] - k_reg) * [lam_reg,])[None, :]

    n = calibration_labels.shape[0]
    cal_order = calibration_probs.argsort(axis=1)[:,::-1]
    cal_sort = np.take_along_axis(calibration_probs, cal_order, axis=1)
    cal_sort_reg = cal_sort + reg_vec
    cal_true_labels = np.where(cal_order == calibration_labels[:,None])[1]
    cal_scores = cal_sort_reg.cumsum(axis=1)[np.arange(n), cal_true_labels] - np.random.rand(n) * cal_sort_reg[np.arange(n), cal_true_labels]

    # Get the score quantile
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    qhat = np.quantile(cal_scores, q_level, method='higher')

    n_test = test_probs.shape[0]
    test_order = test_probs.argsort(1)[:,::-1]
    test_sort = np.take_along_axis(test_probs, test_order, axis=1)
    test_sort_reg = test_sort + reg_vec
    test_srt_reg_cumsum = test_sort_reg.cumsum(axis=1)
    indicators = (test_srt_reg_cumsum - np.random.rand(n_test, 1) * test_sort_reg) <= qhat if rand else test_srt_reg_cumsum - test_sort_reg <= qhat

    if disallow_zero_sets: indicators[:,0] = True
    prediction_sets = np.take_along_axis(indicators, test_order.argsort(axis=1), axis=1)
    return prediction_sets

def cqr_prediction_set(calibration_upper: NDArray, calibration_lower: NDArray, calibration_labels: NDArray, test_upper: NDArray, test_lower: NDArray, alpha = 0.1):
    # Get scores. cal_upper.shape[0] == cal_lower.shape[0] == n
    n = calibration_labels.shape[0]
    cal_scores = np.maximum(calibration_labels - calibration_upper, calibration_lower - calibration_labels)

    # Get the score quantile
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    qhat = np.quantile(cal_scores, q_level, method='higher')

    # Deploy (output=lower and upper adjusted quantiles)
    prediction_sets = [test_lower - qhat, test_upper + qhat]
    return prediction_sets

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def set_size(pred_set):
    return np.mean([np.sum(ps) for ps in pred_set])

def coverage_rate(y_true, pred_sets):
    return pred_sets[np.arange(pred_sets.shape[0]), y_true].mean()

# Example usage:
labels = np.array([0, 2, 1, 1, 2, 0, 0, 1, 2])
probs = np.random.uniform(0, 1, (labels.shape[0], labels.max() + 1))
probs /= probs.sum(axis=1, keepdims=True)
print(labels)
print(probs)

# print(lac_conformal_score(probs, labels))
# print(aps_conformal_score(probs, labels))
# print(entropy_score(probs))
