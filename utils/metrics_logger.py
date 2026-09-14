


"""
metrics_logger.py

Training/validation metricsをCSVに保存するためのユーティリティ関数群。

train.py側では、各epochで得られたtrain loss, validation loss,
COCO bbox metrics, COCO segmentation metrics, checkpoint pathをこのファイルの
関数に渡すだけで、metrics.csvとして記録できる。

このファイルにCSV処理を分離することで、train.pyを学習処理の本体に集中させる。
"""

import csv
import os

import numpy as np


# metrics.csv に出力する列名。
# COCOevalでよく見る AP / AP50 / AP75 / size別AP をbboxとsegmの両方で保存する。
CSV_HEADER = [
    "epoch",
    "train_loss",
    "val_loss",
    "bbox_AP",
    "bbox_AP50",
    "bbox_AP75",
    "bbox_AP_small",
    "bbox_AP_medium",
    "bbox_AP_large",
    "segm_AP",
    "segm_AP50",
    "segm_AP75",
    "segm_AP_small",
    "segm_AP_medium",
    "segm_AP_large",
    "checkpoint",
]

CLASS_CSV_HEADER = [
    "epoch",
    "category_id",
    "class_name",
    "gt_count",
    "bbox_prediction_count",
    "bbox_AP",
    "bbox_AP50",
    "bbox_AP75",
    "bbox_AR100",
    "segm_prediction_count",
    "segm_AP",
    "segm_AP50",
    "segm_AP75",
    "segm_AR100",
]


# 出力ディレクトリ内に metrics.csv を作成し、ヘッダー行だけを書き込む。
# 学習開始時に1回だけ呼び出す。
def init_metrics_csv(output_dir):
    metrics_csv_path = os.path.join(output_dir, "metrics.csv")

    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

    return metrics_csv_path


def init_class_metrics_csv(output_dir):
    """Create the epoch-by-class validation metrics CSV."""
    class_metrics_csv_path = os.path.join(output_dir, "class_metrics.csv")
    with open(class_metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CLASS_CSV_HEADER)
    return class_metrics_csv_path


# metrics が None の場合は空欄を返す。
# 例えば、検出結果が0個でCOCO評価ができなかったepochでもCSV保存で落ちないようにする。
def get_metric(metrics, key):
    if metrics is None:
        return ""
    return metrics[key]


def _display_metric(value):
    """Return an empty CSV cell for unavailable COCO metrics."""
    return "" if value is None else value


def append_class_metrics_csv(
    class_metrics_csv_path,
    epoch,
    bbox_metrics,
    segm_metrics,
):
    """Append one row per foreground class for a validation epoch."""
    bbox_by_class = (bbox_metrics or {}).get("per_class", {})
    segm_by_class = (segm_metrics or {}).get("per_class", {})
    category_ids = sorted(set(bbox_by_class) | set(segm_by_class))

    with open(class_metrics_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for category_id in category_ids:
            bbox = bbox_by_class.get(category_id, {})
            segm = segm_by_class.get(category_id, {})
            category = segm or bbox
            writer.writerow(
                [
                    epoch,
                    category_id,
                    category.get("class_name", f"class_{category_id}"),
                    max(bbox.get("gt_count", 0), segm.get("gt_count", 0)),
                    bbox.get("prediction_count", 0),
                    _display_metric(bbox.get("AP")),
                    _display_metric(bbox.get("AP50")),
                    _display_metric(bbox.get("AP75")),
                    _display_metric(bbox.get("AR100")),
                    segm.get("prediction_count", 0),
                    _display_metric(segm.get("AP")),
                    _display_metric(segm.get("AP50")),
                    _display_metric(segm.get("AP75")),
                    _display_metric(segm.get("AR100")),
                ]
            )


def save_class_ap_plot(output_dir, epoch, bbox_metrics, segm_metrics):
    """Save a horizontal per-class bbox/segmentation AP comparison plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bbox_by_class = (bbox_metrics or {}).get("per_class", {})
    segm_by_class = (segm_metrics or {}).get("per_class", {})
    category_ids = sorted(set(bbox_by_class) | set(segm_by_class))
    if not category_ids:
        return None

    class_names = [
        (segm_by_class.get(category_id) or bbox_by_class[category_id])["class_name"]
        for category_id in category_ids
    ]

    def values(metrics_by_class):
        return [
            (
                np.nan
                if metrics_by_class.get(category_id, {}).get("AP") is None
                else metrics_by_class[category_id]["AP"]
            )
            for category_id in category_ids
        ]

    y_positions = np.arange(len(category_ids), dtype=float)
    figure_height = max(5.0, 0.48 * len(category_ids) + 1.5)
    fig, axis = plt.subplots(figsize=(9.0, figure_height), constrained_layout=True)
    axis.barh(
        y_positions - 0.18,
        values(bbox_by_class),
        height=0.34,
        label="BBox AP@[0.50:0.95]",
        color="#5B8FF9",
    )
    axis.barh(
        y_positions + 0.18,
        values(segm_by_class),
        height=0.34,
        label="Segmentation AP@[0.50:0.95]",
        color="#E868A2",
    )
    axis.set_yticks(y_positions, labels=class_names)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Average precision")
    axis.set_title(f"Validation class-wise AP — epoch {epoch}")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(loc="lower right")

    output_path = os.path.join(output_dir, f"class_ap_epoch_{epoch:03d}.png")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


# 1 epoch分の学習・評価結果を metrics.csv に1行追記する。
# bbox_metrics と segm_metrics は evaluate_coco_bbox / evaluate_coco_segm が返す辞書を想定する。
def append_metrics_csv(
    metrics_csv_path,
    epoch,
    train_loss,
    val_loss,
    bbox_metrics,
    segm_metrics,
    checkpoint_path,
):
    # "a" モードで開くことで、既存のCSVに行を追加する。
    # 各epoch終了時にこの関数を呼ぶと、epochごとの推移が保存される。
    with open(metrics_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                epoch,
                train_loss,
                val_loss if val_loss is not None else "",
                get_metric(bbox_metrics, "AP"),
                get_metric(bbox_metrics, "AP50"),
                get_metric(bbox_metrics, "AP75"),
                get_metric(bbox_metrics, "AP_small"),
                get_metric(bbox_metrics, "AP_medium"),
                get_metric(bbox_metrics, "AP_large"),
                get_metric(segm_metrics, "AP"),
                get_metric(segm_metrics, "AP50"),
                get_metric(segm_metrics, "AP75"),
                get_metric(segm_metrics, "AP_small"),
                get_metric(segm_metrics, "AP_medium"),
                get_metric(segm_metrics, "AP_large"),
                checkpoint_path,
            ]
        )
