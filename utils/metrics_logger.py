


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


# 出力ディレクトリ内に metrics.csv を作成し、ヘッダー行だけを書き込む。
# 学習開始時に1回だけ呼び出す。
def init_metrics_csv(output_dir):
    metrics_csv_path = os.path.join(output_dir, "metrics.csv")

    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

    return metrics_csv_path


# metrics が None の場合は空欄を返す。
# 例えば、検出結果が0個でCOCO評価ができなかったepochでもCSV保存で落ちないようにする。
def get_metric(metrics, key):
    if metrics is None:
        return ""
    return metrics[key]


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