"""Detection/segmentation confusion matrices with background FP/FN bins."""

import csv
import os

import numpy as np
import torch

from utils.threshold_metrics import _box_iou_matrix, _mask_iou_matrix


def _match_instances(
    iou_matrix,
    pred_labels,
    gt_labels,
    category_id_to_index,
):
    """Create one image-level matrix using class-agnostic greedy IoU matching."""
    class_count = len(category_id_to_index)
    background_index = class_count
    matrix = np.zeros((class_count + 1, class_count + 1), dtype=np.int64)
    iou_matrix = np.asarray(iou_matrix, dtype=np.float32)
    pred_labels = np.asarray(pred_labels, dtype=np.int64)
    gt_labels = np.asarray(gt_labels, dtype=np.int64)

    matched_predictions = set()
    matched_ground_truths = set()
    if iou_matrix.size:
        prediction_indices, gt_indices = np.nonzero(iou_matrix >= 0.0)
        candidate_pairs = sorted(
            zip(
                iou_matrix[prediction_indices, gt_indices],
                prediction_indices,
                gt_indices,
            ),
            key=lambda item: (-float(item[0]), int(item[1]), int(item[2])),
        )
        for iou, prediction_index, gt_index in candidate_pairs:
            if iou < 0.0:
                continue
            if prediction_index in matched_predictions:
                continue
            if gt_index in matched_ground_truths:
                continue
            matched_predictions.add(int(prediction_index))
            matched_ground_truths.add(int(gt_index))
            actual_index = category_id_to_index[int(gt_labels[gt_index])]
            predicted_index = category_id_to_index[int(pred_labels[prediction_index])]
            matrix[actual_index, predicted_index] += 1

    for gt_index, category_id in enumerate(gt_labels):
        if gt_index not in matched_ground_truths:
            matrix[category_id_to_index[int(category_id)], background_index] += 1
    for prediction_index, category_id in enumerate(pred_labels):
        if prediction_index not in matched_predictions:
            matrix[background_index, category_id_to_index[int(category_id)]] += 1
    return matrix


def match_confusion_matrix(
    iou_matrix,
    pred_labels,
    gt_labels,
    category_id_to_index,
    match_iou_threshold,
):
    """Match predictions to GT and include unmatched instances as background."""
    thresholded_ious = np.asarray(iou_matrix, dtype=np.float32).copy()
    thresholded_ious[thresholded_ious < match_iou_threshold] = -1.0
    return _match_instances(
        thresholded_ious,
        pred_labels,
        gt_labels,
        category_id_to_index,
    )


def normalize_confusion_matrix(matrix):
    """Normalize each actual-class row while preserving empty rows as zero."""
    matrix = np.asarray(matrix, dtype=np.float64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_sums > 0,
    )


@torch.no_grad()
def evaluate_confusion_matrices(
    model,
    data_loader,
    device,
    category_names,
    score_threshold,
    match_iou_threshold=0.5,
    mask_threshold=0.5,
):
    """Evaluate bbox and mask confusion matrices in one inference pass."""
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be between 0 and 1")
    if not 0.0 <= match_iou_threshold <= 1.0:
        raise ValueError("match_iou_threshold must be between 0 and 1")
    if not 0.0 <= mask_threshold <= 1.0:
        raise ValueError("mask_threshold must be between 0 and 1")

    category_ids = sorted(int(value) for value in category_names)
    category_id_to_index = {
        category_id: index for index, category_id in enumerate(category_ids)
    }
    size = len(category_ids) + 1
    matrices = {
        "bbox": np.zeros((size, size), dtype=np.int64),
        "segm": np.zeros((size, size), dtype=np.int64),
    }

    model.eval()
    for images, targets in data_loader:
        outputs = model([image.to(device) for image in images])
        for target, output in zip(targets, outputs):
            scores = output["scores"].detach().cpu().numpy()
            keep_predictions = scores >= score_threshold
            pred_labels = output["labels"].detach().cpu().numpy()[keep_predictions]
            pred_boxes = output["boxes"].detach().cpu().numpy()[keep_predictions]
            pred_masks = (
                output["masks"].detach().cpu().numpy()[keep_predictions, 0]
                >= mask_threshold
            )

            keep_ground_truths = target["iscrowd"].detach().cpu().numpy() == 0
            gt_labels = target["labels"].detach().cpu().numpy()[keep_ground_truths]
            gt_boxes = target["boxes"].detach().cpu().numpy()[keep_ground_truths]
            gt_masks = (
                target["masks"].detach().cpu().numpy()[keep_ground_truths] > 0
            )

            bbox_ious = _box_iou_matrix(pred_boxes, gt_boxes)
            segm_ious = _mask_iou_matrix(pred_masks, gt_masks)
            matrices["bbox"] += match_confusion_matrix(
                bbox_ious,
                pred_labels,
                gt_labels,
                category_id_to_index,
                match_iou_threshold,
            )
            matrices["segm"] += match_confusion_matrix(
                segm_ious,
                pred_labels,
                gt_labels,
                category_id_to_index,
                match_iou_threshold,
            )

    labels = [str(category_names[category_id]) for category_id in category_ids]
    labels.append("background")
    return {
        "score_threshold": float(score_threshold),
        "match_iou_threshold": float(match_iou_threshold),
        "mask_threshold": float(mask_threshold),
        "labels": labels,
        "bbox": matrices["bbox"],
        "segm": matrices["segm"],
    }


def _write_matrix_csv(path, matrix, labels, normalized):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual\\predicted", *labels])
        for label, row in zip(labels, matrix):
            if normalized:
                values = [float(value) for value in row]
            else:
                values = [int(value) for value in row]
            writer.writerow([label, *values])
    return os.path.abspath(path)


def _annotation_color(value, maximum, normalized):
    if normalized:
        relative = float(value)
    else:
        relative = 0.0 if maximum <= 0 else float(value) / float(maximum)
    return "white" if relative >= 0.55 else "#172033"


def _plot_matrix(
    matrix,
    labels,
    evaluation_type,
    normalized,
    score_threshold,
    match_iou_threshold,
    output_dir,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.asarray(matrix)
    class_count = len(labels)
    figure_size = max(9.0, class_count * 0.72)
    fig, axis = plt.subplots(
        figsize=(figure_size + 1.6, figure_size),
        constrained_layout=True,
    )
    image = axis.imshow(
        matrix,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0 if normalized else None,
        interpolation="nearest",
    )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Fraction of actual class" if normalized else "Count")

    ticks = np.arange(class_count)
    axis.set_xticks(ticks, labels=labels, rotation=48, ha="right")
    axis.set_yticks(ticks, labels=labels)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    kind = "row-normalized" if normalized else "counts"
    axis.set_title(
        f"{evaluation_type.upper()} confusion matrix ({kind})\n"
        f"score ≥ {score_threshold:.2f}, IoU ≥ {match_iou_threshold:.2f}"
    )

    maximum = float(np.max(matrix)) if matrix.size else 0.0
    for row in range(class_count):
        for column in range(class_count):
            value = matrix[row, column]
            text = f"{value:.2f}" if normalized else str(int(value))
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=7.5,
                color=_annotation_color(value, maximum, normalized),
            )

    suffix = "normalized" if normalized else "counts"
    stem = os.path.join(
        output_dir,
        f"confusion_matrix_{evaluation_type}_{suffix}",
    )
    paths = {}
    for extension in ("png", "pdf", "svg"):
        path = f"{stem}.{extension}"
        kwargs = {"dpi": 300} if extension == "png" else {}
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths[extension] = os.path.abspath(path)
    plt.close(fig)
    return paths


def save_confusion_matrix_outputs(results, output_dir):
    """Save raw and row-normalized bbox/segm matrices as figures and CSV."""
    os.makedirs(output_dir, exist_ok=True)
    labels = results["labels"]
    outputs = {}
    for evaluation_type in ("bbox", "segm"):
        counts = np.asarray(results[evaluation_type], dtype=np.int64)
        normalized = normalize_confusion_matrix(counts)
        count_csv = _write_matrix_csv(
            os.path.join(
                output_dir,
                f"confusion_matrix_{evaluation_type}_counts.csv",
            ),
            counts,
            labels,
            normalized=False,
        )
        normalized_csv = _write_matrix_csv(
            os.path.join(
                output_dir,
                f"confusion_matrix_{evaluation_type}_normalized.csv",
            ),
            normalized,
            labels,
            normalized=True,
        )
        outputs[evaluation_type] = {
            "counts": {
                "csv": count_csv,
                **_plot_matrix(
                    counts,
                    labels,
                    evaluation_type,
                    False,
                    results["score_threshold"],
                    results["match_iou_threshold"],
                    output_dir,
                ),
            },
            "normalized": {
                "csv": normalized_csv,
                **_plot_matrix(
                    normalized,
                    labels,
                    evaluation_type,
                    True,
                    results["score_threshold"],
                    results["match_iou_threshold"],
                    output_dir,
                ),
            },
        }
    return outputs
