"""
Conformal Prediction and Uncertainty Quantification Library

This module provides various uncertainty scoring methods and conformal prediction
algorithms for classification tasks. It includes implementations of LAC, APS, RAPS,
and CQR conformal prediction methods, along with various uncertainty metrics.
"""

import numpy as np
from numpy._typing import NDArray
from typing import Tuple, List, Union, Optional


def margin_score(probs: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Calculate the margin score between the top two predicted probabilities.

    The margin score measures the difference between the second-highest and highest
    predicted probabilities. Lower (more negative) values indicate higher confidence,
    as there's a larger gap between the top prediction and the runner-up.

    Parameters
    ----------
    probs : NDArray[np.float64]
        2D array of shape (n_samples, n_classes) containing predicted probabilities
        for each class. Each row should sum to 1.

    Returns
    -------
    NDArray[np.float64]
        1D array of shape (n_samples,) containing margin scores.
        Negative values indicate confident predictions.

    Examples
    --------
    >>> probs = np.array([[0.7, 0.2, 0.1], [0.4, 0.35, 0.25]])
    >>> margin_score(probs)
    array([-0.5, -0.05])
    """
    sorted_probs = np.sort(probs, axis=1)
    return sorted_probs[:, -2] - sorted_probs[:, -1]


def entropy_score(probs: NDArray[np.float64], eps: float = 1e-9) -> NDArray[np.float64]:
    """
    Calculate the entropy-based uncertainty score.

    Entropy measures the uncertainty in the probability distribution. Higher entropy
    indicates more uncertainty (predictions spread across multiple classes).

    Parameters
    ----------
    probs : NDArray[np.float64]
        2D array of shape (n_samples, n_classes) containing predicted probabilities
        for each class. Each row should sum to 1.
    eps : float, optional
        Small constant added to probabilities to avoid log(0). Default is 1e-9.

    Returns
    -------
    NDArray[np.float64]
        1D array of shape (n_samples,) containing entropy scores.
        Higher values indicate higher uncertainty.

    Examples
    --------
    >>> probs = np.array([[0.9, 0.05, 0.05], [0.33, 0.33, 0.34]])
    >>> entropy_score(probs)
    array([0.295, 1.098])  # Approximate values
    """
    return -np.sum(probs * np.log(probs + eps), axis=1)


def nnl_score(probs: NDArray[np.float64], true_labels: NDArray[np.int32],
              eps: float = 1e-9) -> NDArray[np.float64]:
    """
    Calculate the Negative Log-Likelihood (NNL) score.

    NNL measures how well the model's predicted probabilities align with the true labels.
    Lower scores indicate better calibration for the true class.

    Parameters
    ----------
    probs : NDArray[np.float64]
        2D array of shape (n_samples, n_classes) containing predicted probabilities
        for each class. Each row should sum to 1.
    true_labels : NDArray[np.int32]
        1D array of shape (n_samples,) containing true class labels (integers).
    eps : float, optional
        Small constant added to probabilities to avoid log(0). Default is 1e-9.

    Returns
    -------
    NDArray[np.float64]
        1D array of shape (n_samples,) containing NNL scores.
        Lower values indicate better predictions for the true class.

    Examples
    --------
    >>> probs = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])
    >>> true_labels = np.array([0, 1])
    >>> nnl_score(probs, true_labels)
    array([0.223, 0.357])  # Approximate values
    """
    return -np.log(probs[np.arange(probs.shape[0]), true_labels] + eps)


def ri_score(probs: NDArray[np.float64], eps: float = 1e-9) -> NDArray[np.float64]:
    """
    Calculate the Reverse Information (RI) score.

    RI score sums the negative log probabilities across all classes, measuring
    the total information content of the prediction.

    Parameters
    ----------
    probs : NDArray[np.float64]
        2D array of shape (n_samples, n_classes) containing predicted probabilities
        for each class. Each row should sum to 1.
    eps : float, optional
        Small constant added to probabilities to avoid log(0). Default is 1e-9.

    Returns
    -------
    NDArray[np.float64]
        1D array of shape (n_samples,) containing RI scores.
        Higher values indicate more uncertain predictions.

    Examples
    --------
    >>> probs = np.array([[0.8, 0.1, 0.1], [0.33, 0.33, 0.34]])
    >>> ri_score(probs)
    array([4.828, 3.329])  # Approximate values
    """
    return -np.sum(np.log(probs + eps), axis=1)


def lac_conformal_score(probs: NDArray[np.float64],
                        true_labels: NDArray[np.int32]) -> NDArray[np.float64]:
    """
    Compute the Least Ambiguous set-valued Classifier (LAC) conformal score.

    LAC uses the complement of the true class probability as the nonconformity score.
    Higher scores indicate less conformity (worse predictions).

    Parameters
    ----------
    probs : NDArray[np.float64]
        2D array of shape (n_samples, n_classes) containing softmax probabilities
        for each sample. Each row should sum to 1.
    true_labels : NDArray[np.int32]
        1D array of shape (n_samples,) containing true class labels (integers).

    Returns
    -------
    NDArray[np.float64]
        1D array of shape (n_samples,) containing LAC conformal scores.
        Values range from 0 to 1, where higher values indicate worse predictions.

    Examples
    --------
    >>> probs = np.array([[0.8, 0.15, 0.05], [0.3, 0.6, 0.1]])
    >>> true_labels = np.array([0, 1])
    >>> lac_conformal_score(probs, true_labels)
    array([0.2, 0.4])
    """
    conformal_scores = 1 - probs[np.arange(probs.shape[0]), true_labels]
    return conformal_scores


def aps_conformal_score(probs: NDArray[np.float64],
                        true_labels: NDArray[np.int32]) -> NDArray[np.float64]:
    """
    Compute the Adaptive Prediction Sets (APS) conformal score.

    APS creates prediction sets that adapt to the difficulty of each example by
    summing probabilities of all classes with scores at least as high as the true class.

    Parameters
    ----------
    probs : NDArray[np.float64]
        2D array of shape (n_samples, n_classes) containing softmax probabilities
        for each sample. Each row should sum to 1.
    true_labels : NDArray[np.int32]
        1D array of shape (n_samples,) containing true class labels (integers).

    Returns
    -------
    NDArray[np.float64]
        1D array of shape (n_samples,) containing APS conformal scores.
        Higher values indicate that more classes have high probabilities.

    Examples
    --------
    >>> probs = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
    >>> true_labels = np.array([1, 0])
    >>> aps_conformal_score(probs, true_labels)
    array([0.8, 0.8])
    """
    # Create a mask for each sample: probs >= true_score
    true_scores = probs[np.arange(probs.shape[0]), true_labels]
    mask = probs >= true_scores[:, np.newaxis]
    # Sum along the class axis
    conformal_scores = np.sum(probs * mask, axis=1)

    return conformal_scores


def compute_quantile(scores: NDArray[np.float64], n: int,
                    alpha: float = 0.1) -> float:
    """
    Compute the conformal prediction quantile threshold.

    Calculates the (1-α) quantile of the calibration scores with finite-sample
    correction for valid coverage guarantee.

    Parameters
    ----------
    scores : NDArray[np.float64]
        1D array of calibration nonconformity scores.
    n : int
        Number of calibration samples.
    alpha : float, optional
        Desired miscoverage rate (e.g., 0.1 for 90% coverage). Default is 0.1.

    Returns
    -------
    float
        Quantile threshold for constructing prediction sets.

    Notes
    -----
    Uses the finite-sample correction: ceil((n+1)(1-α))/n to ensure valid coverage.
    """
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    return np.quantile(scores, q_level, method="higher")


def lac_prediction_set(calibration_probs: NDArray[np.float64],
                       probs: NDArray[np.float64],
                       calibration_labels: NDArray[np.int32],
                       alpha: float = 0.1) -> NDArray[np.bool_]:
    """
    Construct LAC (Least Ambiguous set-valued Classifier) prediction sets.

    LAC creates prediction sets by including all classes with probability above
    a threshold calibrated to achieve the desired coverage.

    Parameters
    ----------
    calibration_probs : NDArray[np.float64]
        2D array of shape (n_calibration, n_classes) containing calibration set
        softmax probabilities.
    probs : NDArray[np.float64]
        2D array of shape (n_test, n_classes) containing test set softmax probabilities.
    calibration_labels : NDArray[np.int32]
        1D array of shape (n_calibration,) containing calibration set true labels.
    alpha : float, optional
        Desired miscoverage rate (e.g., 0.1 for 90% coverage). Default is 0.1.

    Returns
    -------
    NDArray[np.bool_]
        2D boolean array of shape (n_test, n_classes) where True indicates
        class inclusion in the prediction set.

    Examples
    --------
    >>> cal_probs = np.array([[0.7, 0.2, 0.1], [0.5, 0.3, 0.2]])
    >>> test_probs = np.array([[0.6, 0.3, 0.1]])
    >>> cal_labels = np.array([0, 1])
    >>> pred_sets = lac_prediction_set(cal_probs, test_probs, cal_labels, alpha=0.1)
    """
    n = calibration_labels.shape[0]
    cal_scores = 1 - calibration_probs[np.arange(calibration_probs.shape[0]), calibration_labels]
    # Get the score quantile

    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    qhat = np.quantile(cal_scores, q_level, method='higher')

    prediction_sets = probs >= (1 - qhat)
    return prediction_sets


def aps_prediction_set(calibration_probs: NDArray[np.float64],
                      probs: NDArray[np.float64],
                      calibration_labels: NDArray[np.int32],
                      alpha: float = 0.1) -> NDArray[np.bool_]:
    """
    Construct APS (Adaptive Prediction Sets) for conformal prediction.

    APS creates prediction sets that adapt to the difficulty of each example,
    typically resulting in smaller sets for easy examples and larger sets for
    difficult ones.

    Parameters
    ----------
    calibration_probs : NDArray[np.float64]
        2D array of shape (n_calibration, n_classes) containing calibration set
        softmax probabilities.
    probs : NDArray[np.float64]
        2D array of shape (n_test, n_classes) containing test set softmax probabilities.
    calibration_labels : NDArray[np.int32]
        1D array of shape (n_calibration,) containing calibration set true labels.
    alpha : float, optional
        Desired miscoverage rate (e.g., 0.1 for 90% coverage). Default is 0.1.

    Returns
    -------
    NDArray[np.bool_]
        2D boolean array of shape (n_test, n_classes) where True indicates
        class inclusion in the prediction set.
    """
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


def raps_prediction_set(calibration_probs: NDArray[np.float64],
                        test_probs: NDArray[np.float64],
                        calibration_labels: NDArray[np.int32],
                        alpha: float = 0.1,
                        lam_reg: float = 0.01,
                        k_reg: int = 5,
                        disallow_zero_sets: bool = False,
                        rand: bool = True) -> NDArray[np.bool_]:
    """
    Construct RAPS (Regularized Adaptive Prediction Sets) for conformal prediction.

    RAPS extends APS by adding regularization to favor including high-probability
    classes first, resulting in more stable and interpretable prediction sets.

    Parameters
    ----------
    calibration_probs : NDArray[np.float64]
        2D array of shape (n_calibration, n_classes) containing calibration set
        softmax probabilities.
    test_probs : NDArray[np.float64]
        2D array of shape (n_test, n_classes) containing test set softmax probabilities.
    calibration_labels : NDArray[np.int32]
        1D array of shape (n_calibration,) containing calibration set true labels.
    alpha : float, optional
        Desired miscoverage rate (e.g., 0.1 for 90% coverage). Default is 0.1.
    lam_reg : float, optional
        Regularization parameter. Larger values lead to smaller sets. Default is 0.01.
    k_reg : int, optional
        Number of top classes to not regularize. Smaller values lead to smaller sets.
        Default is 5.
    disallow_zero_sets : bool, optional
        If True, always include at least the top predicted class. Default is False.
    rand : bool, optional
        If True, add randomization for exact coverage guarantee. Default is True.

    Returns
    -------
    NDArray[np.bool_]
        2D boolean array of shape (n_test, n_classes) where True indicates
        class inclusion in the prediction set.

    Notes
    -----
    - Larger lam_reg and smaller k_reg lead to smaller prediction sets
    - Set disallow_zero_sets=False to see the coverage upper bound hold
    - Set rand=True to see the coverage upper bound hold
    """
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


def cqr_prediction_set(calibration_upper: NDArray[np.float64],
                       calibration_lower: NDArray[np.float64],
                       calibration_labels: NDArray[np.float64],
                       test_upper: NDArray[np.float64],
                       test_lower: NDArray[np.float64],
                       alpha: float = 0.1) -> List[NDArray[np.float64]]:
    """
    Construct CQR (Conformalized Quantile Regression) prediction intervals.

    CQR creates prediction intervals for regression tasks by conformally calibrating
    the outputs of a quantile regression model.

    Parameters
    ----------
    calibration_upper : NDArray[np.float64]
        1D array of shape (n_calibration,) containing upper quantile predictions
        for calibration set.
    calibration_lower : NDArray[np.float64]
        1D array of shape (n_calibration,) containing lower quantile predictions
        for calibration set.
    calibration_labels : NDArray[np.float64]
        1D array of shape (n_calibration,) containing true values for calibration set.
    test_upper : NDArray[np.float64]
        1D array of shape (n_test,) containing upper quantile predictions for test set.
    test_lower : NDArray[np.float64]
        1D array of shape (n_test,) containing lower quantile predictions for test set.
    alpha : float, optional
        Desired miscoverage rate (e.g., 0.1 for 90% coverage). Default is 0.1.

    Returns
    -------
    List[NDArray[np.float64]]
        List containing two arrays:
        - Lower bounds of prediction intervals (shape: n_test,)
        - Upper bounds of prediction intervals (shape: n_test,)

    Examples
    --------
    >>> cal_upper = np.array([2.5, 3.0, 2.8])
    >>> cal_lower = np.array([1.5, 2.0, 1.8])
    >>> cal_labels = np.array([2.0, 2.5, 2.2])
    >>> test_upper = np.array([2.7, 3.2])
    >>> test_lower = np.array([1.7, 2.2])
    >>> intervals = cqr_prediction_set(cal_upper, cal_lower, cal_labels,
    ...                                test_upper, test_lower, alpha=0.1)
    """
    # Get scores. cal_upper.shape[0] == cal_lower.shape[0] == n
    n = calibration_labels.shape[0]
    cal_scores = np.maximum(calibration_labels - calibration_upper, calibration_lower - calibration_labels)

    # Get the score quantile
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    qhat = np.quantile(cal_scores, q_level, method='higher')

    # Deploy (output=lower and upper adjusted quantiles)
    prediction_sets = [test_lower - qhat, test_upper + qhat]
    return prediction_sets


def accuracy(y_true: NDArray[np.int32], y_pred: NDArray[np.int32]) -> float:
    """
    Calculate classification accuracy.

    Parameters
    ----------
    y_true : NDArray[np.int32]
        1D array of true class labels.
    y_pred : NDArray[np.int32]
        1D array of predicted class labels.

    Returns
    -------
    float
        Accuracy score between 0 and 1.

    Examples
    --------
    >>> y_true = np.array([0, 1, 2, 1])
    >>> y_pred = np.array([0, 1, 1, 1])
    >>> accuracy(y_true, y_pred)
    0.75
    """
    return np.mean(y_true == y_pred)


def set_size(pred_sets: NDArray[np.bool_]) -> float:
    """
    Calculate the average size of prediction sets.

    Parameters
    ----------
    pred_sets : NDArray[np.bool_]
        2D boolean array of shape (n_samples, n_classes) where True indicates
        class inclusion in the prediction set.

    Returns
    -------
    float
        Average number of classes included in prediction sets.

    Examples
    --------
    >>> pred_sets = np.array([[True, False, True], [True, True, False]])
    >>> set_size(pred_sets)
    2.0
    """
    return np.mean([np.sum(ps) for ps in pred_sets])


def coverage_rate(y_true: NDArray[np.int32], pred_sets: NDArray[np.bool_]) -> float:
    """
    Calculate the empirical coverage rate of prediction sets.

    Coverage rate is the fraction of test samples where the true label
    is included in the prediction set.

    Parameters
    ----------
    y_true : NDArray[np.int32]
        1D array of shape (n_samples,) containing true class labels.
    pred_sets : NDArray[np.bool_]
        2D boolean array of shape (n_samples, n_classes) where True indicates
        class inclusion in the prediction set.

    Returns
    -------
    float
        Empirical coverage rate between 0 and 1.

    Examples
    --------
    >>> y_true = np.array([0, 1, 2])
    >>> pred_sets = np.array([[True, False, False],
    ...                       [True, True, False],
    ...                       [False, False, False]])
    >>> coverage_rate(y_true, pred_sets)
    0.667  # 2 out of 3 samples have true label in prediction set
    """
    return pred_sets[np.arange(pred_sets.shape[0]), y_true].mean()


# Example usage (commented out):
"""
if __name__ == "__main__":
    # Generate example data
    labels = np.array([0, 2, 1, 1, 2, 0, 0, 1, 2])
    probs = np.random.uniform(0, 1, (labels.shape[0], labels.max() + 1))
    probs /= probs.sum(axis=1, keepdims=True)

    print("Labels:", labels)
    print("Probabilities:", probs)

    # Test uncertainty scores
    print("LAC scores:", lac_conformal_score(probs, labels))
    print("APS scores:", aps_conformal_score(probs, labels))
    print("Entropy scores:", entropy_score(probs))

    # Split data for conformal prediction
    n_cal = 5
    cal_probs, test_probs = probs[:n_cal], probs[n_cal:]
    cal_labels, test_labels = labels[:n_cal], labels[n_cal:]

    # Generate prediction sets
    lac_sets = lac_prediction_set(cal_probs, test_probs, cal_labels, alpha=0.1)
    aps_sets = aps_prediction_set(cal_probs, test_probs, cal_labels, alpha=0.1)

    print("LAC prediction sets:", lac_sets)
    print("APS prediction sets:", aps_sets)
    print("LAC coverage:", coverage_rate(test_labels, lac_sets))
    print("APS coverage:", coverage_rate(test_labels, aps_sets))
"""
