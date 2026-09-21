"""Save publication-ready precision-recall curves from accumulated COCOeval."""

import csv
import os

import numpy as np


def _mean_valid(values, axis=None):
    """Average COCO precision values while ignoring the -1 sentinel."""
    values = np.asarray(values, dtype=float)
    valid = values > -1
    counts = valid.sum(axis=axis)
    totals = np.where(valid, values, 0.0).sum(axis=axis)
    result = np.full(np.shape(totals), np.nan, dtype=float)
    np.divide(totals, counts, out=result, where=counts > 0)
    return result


def _iou_index(iou_thresholds, target):
    matches = np.flatnonzero(np.isclose(iou_thresholds, target))
    if len(matches) != 1:
        raise ValueError(f"COCO IoU threshold not found: {target}")
    return int(matches[0])


def _valid_precision(values):
    """Replace COCO's unavailable-value sentinel with NaN."""
    values = np.asarray(values, dtype=float)
    return np.where(values > -1, values, np.nan)


def extract_coco_pr_curves(coco_eval, category_names):
    """Extract COCO's 101-point PR curves for area=all and maxDets=100."""
    area_index = coco_eval.params.areaRngLbl.index("all")
    max_dets_index = coco_eval.params.maxDets.index(100)
    precision = np.asarray(coco_eval.eval["precision"], dtype=float)[
        :, :, :, area_index, max_dets_index
    ]
    recall_thresholds = np.asarray(coco_eval.params.recThrs, dtype=float)
    iou_thresholds = np.asarray(coco_eval.params.iouThrs, dtype=float)
    category_ids = [int(value) for value in coco_eval.params.catIds]

    iou50_index = _iou_index(iou_thresholds, 0.50)
    iou75_index = _iou_index(iou_thresholds, 0.75)

    def curve(values, axis):
        return np.asarray(_mean_valid(values, axis=axis), dtype=float)

    overall = {
        "mean": curve(precision, axis=(0, 2)),
        "iou50": curve(precision[iou50_index], axis=1),
        "iou75": curve(precision[iou75_index], axis=1),
    }
    per_class = {}
    for category_index, category_id in enumerate(category_ids):
        class_precision = precision[:, :, category_index]
        per_class[category_id] = {
            "class_name": str(
                category_names.get(category_id, f"class_{category_id}")
            ),
            "mean": curve(class_precision, axis=0),
            "iou50": _valid_precision(class_precision[iou50_index]),
            "iou75": _valid_precision(class_precision[iou75_index]),
        }

    return {
        "recall": recall_thresholds,
        "overall": overall,
        "per_class": per_class,
    }


def _average_precision(curve):
    values = np.asarray(curve, dtype=float)
    values = values[np.isfinite(values)]
    return None if values.size == 0 else float(values.mean())


def _save_figure(fig, output_stem):
    paths = {}
    for extension in ("png", "pdf", "svg"):
        path = f"{output_stem}.{extension}"
        save_kwargs = {"dpi": 300} if extension == "png" else {}
        fig.savefig(path, bbox_inches="tight", **save_kwargs)
        paths[extension] = os.path.abspath(path)
    return paths


def _plot_overall(curves, evaluation_type, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recall = curves["recall"]
    styles = (
        ("mean", "IoU 0.50:0.95", "#111111", "-"),
        ("iou50", "IoU 0.50", "#0072B2", "--"),
        ("iou75", "IoU 0.75", "#D55E00", "-."),
    )
    fig, axis = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    for key, label, color, linestyle in styles:
        values = curves["overall"][key]
        ap = _average_precision(values)
        ap_label = "n/a" if ap is None else f"{ap:.3f}"
        axis.plot(
            recall,
            values,
            color=color,
            linestyle=linestyle,
            linewidth=2.2,
            label=f"{label} (AP={ap_label})",
        )
    axis.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.02),
        xlabel="Recall",
        ylabel="Precision",
        title=f"COCO {evaluation_type} precision–recall",
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="lower left", frameon=True)
    stem = os.path.join(output_dir, f"coco_pr_{evaluation_type}_overall")
    paths = _save_figure(fig, stem)
    plt.close(fig)
    return paths


def _plot_per_class(curves, evaluation_type, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from utils.visualize import CVAT_CLASS_COLORS
    except ImportError:
        CVAT_CLASS_COLORS = {}

    recall = curves["recall"]
    fallback_colors = plt.get_cmap("tab20").colors
    fig, axis = plt.subplots(figsize=(10.5, 6.8))
    for index, (category_id, values) in enumerate(curves["per_class"].items()):
        class_name = values["class_name"]
        precision = values["mean"]
        ap = _average_precision(precision)
        ap_label = "n/a" if ap is None else f"{ap:.3f}"
        axis.plot(
            recall,
            precision,
            linewidth=1.8,
            color=CVAT_CLASS_COLORS.get(
                class_name,
                fallback_colors[index % len(fallback_colors)],
            ),
            label=f"{class_name} (AP={ap_label})",
        )
    axis.set(
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.02),
        xlabel="Recall",
        ylabel="Precision",
        title=f"COCO {evaluation_type} class-wise PR — IoU 0.50:0.95",
    )
    axis.grid(alpha=0.25)
    axis.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=8.5,
    )
    fig.subplots_adjust(right=0.72)
    stem = os.path.join(output_dir, f"coco_pr_{evaluation_type}_per_class")
    paths = _save_figure(fig, stem)
    plt.close(fig)
    return paths


def _write_curves_csv(curves, evaluation_type, output_dir):
    path = os.path.join(output_dir, f"coco_pr_{evaluation_type}_values.csv")
    labels = {
        "mean": "0.50:0.95",
        "iou50": "0.50",
        "iou75": "0.75",
    }
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "evaluation_type",
                "scope",
                "category_id",
                "class_name",
                "iou",
                "recall",
                "precision",
            ]
        )
        for key, iou_label in labels.items():
            for recall, precision in zip(
                curves["recall"], curves["overall"][key]
            ):
                writer.writerow(
                    [
                        evaluation_type,
                        "overall",
                        "",
                        "all classes",
                        iou_label,
                        float(recall),
                        "" if not np.isfinite(precision) else float(precision),
                    ]
                )
        for category_id, values in curves["per_class"].items():
            for key, iou_label in labels.items():
                for recall, precision in zip(curves["recall"], values[key]):
                    writer.writerow(
                        [
                            evaluation_type,
                            "class",
                            category_id,
                            values["class_name"],
                            iou_label,
                            float(recall),
                            "" if not np.isfinite(precision) else float(precision),
                        ]
                    )
    return os.path.abspath(path)


def save_coco_pr_outputs(coco_eval, category_names, evaluation_type, output_dir):
    """Save overall/class-wise COCO PR figures and the underlying values."""
    if coco_eval is None:
        return None
    if evaluation_type not in {"bbox", "segm"}:
        raise ValueError(f"Unsupported COCO evaluation type: {evaluation_type}")
    os.makedirs(output_dir, exist_ok=True)
    curves = extract_coco_pr_curves(coco_eval, category_names)
    return {
        "overall": _plot_overall(curves, evaluation_type, output_dir),
        "per_class": _plot_per_class(curves, evaluation_type, output_dir),
        "csv": _write_curves_csv(curves, evaluation_type, output_dir),
    }
