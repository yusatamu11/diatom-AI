"""Fixed-score instance metrics and validation threshold selection."""

import csv
import json
import os

import numpy as np
import torch
from pycocotools import mask as mask_utils


CSV_HEADER = [
    "evaluation_type",
    "score_threshold",
    "scope",
    "category_id",
    "class_name",
    "gt_count",
    "prediction_count",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
]


def make_score_thresholds(start=0.05, stop=0.95, step=0.05):
    """Return an inclusive, numerically stable score-threshold grid."""
    if not 0.0 <= start <= 1.0:
        raise ValueError("threshold_start must be between 0 and 1")
    if not 0.0 <= stop <= 1.0:
        raise ValueError("threshold_stop must be between 0 and 1")
    if stop < start:
        raise ValueError("threshold_stop must be greater than or equal to start")
    if step <= 0.0:
        raise ValueError("threshold_step must be greater than 0")

    count = int(np.floor((stop - start) / step + 1e-9)) + 1
    thresholds = [round(start + index * step, 10) for index in range(count)]
    if thresholds[-1] < stop - 1e-9:
        thresholds.append(round(stop, 10))
    return thresholds


def _box_iou_matrix(pred_boxes, gt_boxes):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)

    pred_boxes = np.asarray(pred_boxes, dtype=np.float32)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
    top_left = np.maximum(pred_boxes[:, None, :2], gt_boxes[None, :, :2])
    bottom_right = np.minimum(pred_boxes[:, None, 2:], gt_boxes[None, :, 2:])
    intersection_size = np.clip(bottom_right - top_left, 0.0, None)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]

    pred_size = np.clip(pred_boxes[:, 2:] - pred_boxes[:, :2], 0.0, None)
    gt_size = np.clip(gt_boxes[:, 2:] - gt_boxes[:, :2], 0.0, None)
    pred_area = pred_size[:, 0] * pred_size[:, 1]
    gt_area = gt_size[:, 0] * gt_size[:, 1]
    union = pred_area[:, None] + gt_area[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > 0,
    )


def _encode_masks(masks):
    return [
        mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        for mask in masks
    ]


def _mask_iou_matrix(pred_masks, gt_masks):
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)
    return np.asarray(
        mask_utils.iou(
            _encode_masks(pred_masks),
            _encode_masks(gt_masks),
            [0] * len(gt_masks),
        ),
        dtype=np.float32,
    )


def _match_counts(iou_matrix, scores, score_threshold, match_iou_threshold):
    """Greedily match score-sorted predictions to unused ground truths."""
    scores = np.asarray(scores, dtype=np.float32)
    selected = np.flatnonzero(scores >= score_threshold)
    if len(selected):
        order = selected[np.argsort(-scores[selected], kind="stable")]
    else:
        order = selected

    matched_gt = set()
    true_positives = 0
    gt_count = iou_matrix.shape[1]
    for prediction_index in order:
        if gt_count == 0:
            break
        available = [index for index in range(gt_count) if index not in matched_gt]
        if not available:
            break
        available_ious = iou_matrix[prediction_index, available]
        best_offset = int(np.argmax(available_ious))
        best_gt = available[best_offset]
        if available_ious[best_offset] >= match_iou_threshold:
            matched_gt.add(best_gt)
            true_positives += 1

    prediction_count = len(order)
    return {
        "gt_count": int(gt_count),
        "prediction_count": int(prediction_count),
        "tp": int(true_positives),
        "fp": int(prediction_count - true_positives),
        "fn": int(gt_count - true_positives),
    }


def _empty_counts():
    return {
        "gt_count": 0,
        "prediction_count": 0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }


def _add_counts(total, addition):
    for key in total:
        total[key] += addition[key]


def _metrics_from_counts(counts):
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    precision_denominator = tp + fp
    recall_denominator = tp + fn

    precision = tp / precision_denominator if precision_denominator else 0.0
    recall = tp / recall_denominator if recall_denominator else None
    if recall is None:
        f1 = None
    elif precision + recall:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        **counts,
        "precision": float(precision),
        "recall": None if recall is None else float(recall),
        "f1": None if f1 is None else float(f1),
    }


def _macro_metrics(per_class):
    included = [metrics for metrics in per_class.values() if metrics["gt_count"] > 0]
    if not included:
        return {"precision": None, "recall": None, "f1": None}
    return {
        metric: float(np.mean([values[metric] for values in included]))
        for metric in ("precision", "recall", "f1")
    }


def _finalize(raw_stats, thresholds, category_names):
    finalized = {"bbox": [], "segm": []}
    for evaluation_type in finalized:
        for threshold in thresholds:
            per_class = {}
            micro_counts = _empty_counts()
            for category_id, class_name in category_names.items():
                counts = raw_stats[evaluation_type][threshold][category_id]
                metrics = _metrics_from_counts(counts)
                metrics["class_name"] = class_name
                per_class[str(category_id)] = metrics
                _add_counts(micro_counts, counts)

            finalized[evaluation_type].append(
                {
                    "score_threshold": threshold,
                    "per_class": per_class,
                    "micro": _metrics_from_counts(micro_counts),
                    "macro": _macro_metrics(per_class),
                }
            )
    return finalized


def evaluate_threshold_metrics(
    model,
    data_loader,
    device,
    category_names,
    score_thresholds,
    match_iou_threshold=0.5,
    mask_threshold=0.5,
):
    """Calculate bbox and mask TP/FP/FN over one or more score thresholds."""
    if not 0.0 <= match_iou_threshold <= 1.0:
        raise ValueError("match_iou_threshold must be between 0 and 1")
    if not 0.0 <= mask_threshold <= 1.0:
        raise ValueError("mask_threshold must be between 0 and 1")

    thresholds = [float(value) for value in score_thresholds]
    if not thresholds:
        raise ValueError("At least one score threshold is required")
    if any(value < 0.0 or value > 1.0 for value in thresholds):
        raise ValueError("Score thresholds must be between 0 and 1")

    raw_stats = {
        evaluation_type: {
            threshold: {
                category_id: _empty_counts() for category_id in category_names
            }
            for threshold in thresholds
        }
        for evaluation_type in ("bbox", "segm")
    }

    model.eval()
    with torch.no_grad():
        for images, targets in data_loader:
            outputs = model([image.to(device) for image in images])
            for target, output in zip(targets, outputs):
                scores = output["scores"].detach().cpu().numpy()
                pred_labels = output["labels"].detach().cpu().numpy()
                pred_boxes = output["boxes"].detach().cpu().numpy()
                pred_masks = (
                    output["masks"].detach().cpu().numpy()[:, 0]
                    >= mask_threshold
                )

                gt_labels = target["labels"].detach().cpu().numpy()
                gt_boxes = target["boxes"].detach().cpu().numpy()
                gt_masks = target["masks"].detach().cpu().numpy() > 0

                for category_id in category_names:
                    pred_indices = np.flatnonzero(pred_labels == category_id)
                    gt_indices = np.flatnonzero(gt_labels == category_id)
                    class_scores = scores[pred_indices]
                    bbox_ious = _box_iou_matrix(
                        pred_boxes[pred_indices],
                        gt_boxes[gt_indices],
                    )
                    segm_ious = _mask_iou_matrix(
                        pred_masks[pred_indices],
                        gt_masks[gt_indices],
                    )

                    for threshold in thresholds:
                        bbox_counts = _match_counts(
                            bbox_ious,
                            class_scores,
                            threshold,
                            match_iou_threshold,
                        )
                        segm_counts = _match_counts(
                            segm_ious,
                            class_scores,
                            threshold,
                            match_iou_threshold,
                        )
                        _add_counts(
                            raw_stats["bbox"][threshold][category_id],
                            bbox_counts,
                        )
                        _add_counts(
                            raw_stats["segm"][threshold][category_id],
                            segm_counts,
                        )

    return {
        "match_iou_threshold": float(match_iou_threshold),
        "mask_threshold": float(mask_threshold),
        "thresholds": thresholds,
        **_finalize(raw_stats, thresholds, category_names),
    }


def _best_entry(entries, selection_metric):
    scope, metric = selection_metric.split("_", 1)

    def ranking(entry):
        value = entry[scope][metric]
        recall = entry[scope]["recall"]
        return (
            float("-inf") if value is None else value,
            float("-inf") if recall is None else recall,
            -entry["score_threshold"],
        )

    return max(entries, key=ranking)


def _selected_summary(entry):
    return {
        "score_threshold": entry["score_threshold"],
        "micro": entry["micro"],
        "macro": entry["macro"],
        "per_class": entry["per_class"],
    }


def save_threshold_metrics(
    results,
    output_dir,
    mode,
    selection_metric="macro_f1",
):
    """Save detailed CSV plus selected/fixed-threshold JSON summary."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "threshold_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_HEADER)
        for evaluation_type in ("bbox", "segm"):
            for entry in results[evaluation_type]:
                threshold = entry["score_threshold"]
                for category_id, metrics in entry["per_class"].items():
                    writer.writerow(
                        [
                            evaluation_type,
                            threshold,
                            "class",
                            category_id,
                            metrics["class_name"],
                            metrics["gt_count"],
                            metrics["prediction_count"],
                            metrics["tp"],
                            metrics["fp"],
                            metrics["fn"],
                            metrics["precision"],
                            metrics["recall"],
                            metrics["f1"],
                        ]
                    )
                micro = entry["micro"]
                writer.writerow(
                    [
                        evaluation_type,
                        threshold,
                        "micro",
                        "",
                        "overall micro",
                        micro["gt_count"],
                        micro["prediction_count"],
                        micro["tp"],
                        micro["fp"],
                        micro["fn"],
                        micro["precision"],
                        micro["recall"],
                        micro["f1"],
                    ]
                )
                macro = entry["macro"]
                writer.writerow(
                    [
                        evaluation_type,
                        threshold,
                        "macro",
                        "",
                        "overall macro",
                        "",
                        "",
                        "",
                        "",
                        "",
                        macro["precision"],
                        macro["recall"],
                        macro["f1"],
                    ]
                )

    if mode == "sweep":
        selected = {
            evaluation_type: _selected_summary(
                _best_entry(results[evaluation_type], selection_metric)
            )
            for evaluation_type in ("bbox", "segm")
        }
    elif mode == "fixed":
        selected = {
            evaluation_type: _selected_summary(results[evaluation_type][0])
            for evaluation_type in ("bbox", "segm")
        }
    else:
        raise ValueError(f"Unsupported threshold mode: {mode}")

    summary = {
        "mode": mode,
        "selection_metric": selection_metric if mode == "sweep" else None,
        "match_iou_threshold": results["match_iou_threshold"],
        "mask_threshold": results["mask_threshold"],
        "selected": selected,
    }
    summary_path = os.path.join(output_dir, "threshold_summary.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return csv_path, summary_path, summary


def print_threshold_summary(summary):
    """Print selected/fixed overall metrics in a compact form."""
    label = "Best validation" if summary["mode"] == "sweep" else "Fixed test"
    print(f"{label} threshold metrics (IoU={summary['match_iou_threshold']:.2f}):")
    for evaluation_type in ("bbox", "segm"):
        selected = summary["selected"][evaluation_type]
        micro = selected["micro"]
        macro = selected["macro"]
        print(
            f"  {evaluation_type}: score threshold={selected['score_threshold']:.2f}, "
            f"micro P/R/F1={micro['precision']:.3f}/"
            f"{micro['recall']:.3f}/{micro['f1']:.3f}, "
            f"macro F1={macro['f1']:.3f}"
        )
