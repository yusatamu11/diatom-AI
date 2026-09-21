import numpy as np

from utils.confusion_matrix import (
    match_confusion_matrix,
    normalize_confusion_matrix,
)


def test_confusion_matrix_records_cross_class_fp_and_fn():
    category_id_to_index = {1: 0, 2: 1}
    ious = np.array(
        [
            [0.90, 0.10],
            [0.20, 0.80],
            [0.10, 0.10],
        ],
        dtype=np.float32,
    )
    matrix = match_confusion_matrix(
        ious,
        pred_labels=np.array([2, 2, 1]),
        gt_labels=np.array([1, 2]),
        category_id_to_index=category_id_to_index,
        match_iou_threshold=0.5,
    )

    assert matrix.tolist() == [
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],
    ]


def test_confusion_matrix_records_unmatched_ground_truth():
    matrix = match_confusion_matrix(
        np.zeros((0, 1), dtype=np.float32),
        pred_labels=np.array([], dtype=np.int64),
        gt_labels=np.array([1]),
        category_id_to_index={1: 0},
        match_iou_threshold=0.5,
    )

    assert matrix.tolist() == [[0, 1], [0, 0]]


def test_normalization_is_by_actual_class_row():
    normalized = normalize_confusion_matrix(
        np.array([[3, 1], [0, 0]], dtype=np.int64)
    )

    assert np.allclose(normalized[0], [0.75, 0.25])
    assert np.allclose(normalized[1], [0.0, 0.0])
