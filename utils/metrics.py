import numpy as np
from sklearn.metrics import confusion_matrix as sk_confusion_matrix


def unweighted_avg_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """UAR: mean per-class recall. Standard primary metric for imbalanced SER datasets."""
    classes = np.unique(y_true)
    recalls = []
    for c in classes:
        mask = y_true == c
        if mask.sum() == 0:
            continue
        recalls.append((y_pred[mask] == c).mean())
    return float(np.mean(recalls))


def weighted_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def per_frame_accuracy(
    all_logits: list,
    all_labels: list,
    max_frames: int = None,
) -> np.ndarray:
    """
    Compute accuracy at each frame index t across the test set.

    Args:
        all_logits : list of (T_i, C) numpy arrays, one per clip
        all_labels : list of int scalars, one per clip
        max_frames : if set, truncate/pad curve to this length

    Returns:
        curve : 1-D array of length max_frames (or max clip length),
                where curve[t] = fraction of clips correctly classified at frame t.
                Clips shorter than t are excluded from the denominator at that t.
    """
    if max_frames is None:
        max_frames = max(lg.shape[0] for lg in all_logits)

    correct_at = np.zeros(max_frames, dtype=np.float64)
    count_at = np.zeros(max_frames, dtype=np.float64)

    for logits, label in zip(all_logits, all_labels):
        T = min(logits.shape[0], max_frames)
        preds = logits[:T].argmax(axis=1)
        correct_at[:T] += (preds == label)
        count_at[:T] += 1

    with np.errstate(invalid="ignore"):
        curve = np.where(count_at > 0, correct_at / count_at, np.nan)
    return curve


def first_correct_frame(logits: np.ndarray, label: int):
    """
    Return the index of the earliest frame at which argmax(logits[t]) == label.
    Returns None if the model never predicts correctly.
    """
    preds = logits.argmax(axis=1)
    indices = np.where(preds == label)[0]
    return int(indices[0]) if len(indices) > 0 else None


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels=None):
    return sk_confusion_matrix(y_true, y_pred, labels=labels)
