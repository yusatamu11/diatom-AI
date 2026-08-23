"""現在の11クラスモデルについて、クラス別PR曲線とF1曲線を計算する。"""

import csv
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config
from pycocotools.coco import COCO
from scipy.ndimage import gaussian_filter1d


# =========================
# 1. 設定
# =========================
CONFIG_FILE = "/content/mmdetection/configs/diatom/mask_rCNN_diatom_pre-train.py"
CHECKPOINT_FILE = (
    "/content/drive/MyDrive/Colab Notebooks/MMdetection/Mask_R-CNN/"
    "work_dirs_26-01-13/epoch_100.pth"
)
DEVICE = "cuda:0"  # CPUで実行する場合は "cpu"
SPLIT = "val"
IOU_THRESH = 0.5
CONF_THRESHOLDS = np.linspace(0.0, 1.0, 100)

# 現在のモデルにおける内部ラベル0〜10の順番。
CURRENT_CLASS_NAMES = (
    "A.subarctica",       # 0
    "cyclostephanoids",   # 1
    "circle",             # 2
    "A.ambigua",          # 3
    "A.glanulate",        # 4
    "buble",              # 5
    "band",               # 6
    "Bacillariophyceae",  # 7
    "Fragilariophycea",   # 8
    "debri",              # 9
    "plant fragment",     # 10
)

# PR曲線・F1曲線へ表示し、平均指標にも使う5クラス。
IMPORTANT_CLASS_NAMES = (
    "A.subarctica",
    "cyclostephanoids",
    "A.ambigua",
    "Fragilariophycea",
    "plant fragment",
)

CLASS_COLORS = {
    "A.subarctica": "#F3A6B8",
    "cyclostephanoids": "#A98AD9",
    "A.ambigua": "#76C9D8",
    "Fragilariophycea": "#F1DD63",
    "plant fragment": "#8FA36B",
}

SMOOTH_SIGMA_PR = 15
SMOOTH_SIGMA_F1 = 3

OUT_DIR = (
    "/content/drive/MyDrive/Colab Notebooks/MMdetection/Mask_R-CNN/"
    "analysis_plots_current_classes"
)
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# 2. ユーティリティ
# =========================
def xyxy_iou(box1, box2):
    """xyxy形式の2つのbbox間のIoUを計算する。"""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    intersection_width = max(0.0, x_right - x_left)
    intersection_height = max(0.0, y_bottom - y_top)
    intersection = intersection_width * intersection_height

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def match_predictions_to_gt(predictions, ground_truths, iou_threshold=0.5):
    """スコア順の予測を未対応GTへ貪欲に一対一対応させる。"""
    sorted_predictions = sorted(
        predictions,
        key=lambda prediction: prediction["score"],
        reverse=True,
    )
    matched_gt = [False] * len(ground_truths)
    prediction_matches = []

    for prediction in sorted_predictions:
        best_iou = -1.0
        best_gt_index = -1
        for gt_index, ground_truth in enumerate(ground_truths):
            if matched_gt[gt_index]:
                continue
            iou = xyxy_iou(prediction["bbox"], ground_truth["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if best_gt_index >= 0 and best_iou >= iou_threshold:
            matched_gt[best_gt_index] = True
            prediction_matches.append(True)
        else:
            prediction_matches.append(False)

    return prediction_matches, len(ground_truths), sorted_predictions


def compute_pr_curve(
    predictions_by_image,
    ground_truths_by_image,
    image_ids,
    iou_threshold=0.5,
):
    """全画像の予測を使い、precision・recall・APを計算する。"""
    scored_predictions = []
    total_gt = 0

    # GTがない画像上の予測もFPとして評価するため、全画像を走査する。
    for image_id in image_ids:
        predictions = predictions_by_image.get(image_id, [])
        ground_truths = ground_truths_by_image.get(image_id, [])
        matches, num_gt, sorted_predictions = match_predictions_to_gt(
            predictions,
            ground_truths,
            iou_threshold=iou_threshold,
        )
        total_gt += num_gt
        for prediction, is_true_positive in zip(sorted_predictions, matches):
            scored_predictions.append(
                (prediction["score"], is_true_positive)
            )

    scored_predictions.sort(key=lambda item: item[0], reverse=True)
    if total_gt == 0:
        return np.array([]), np.array([]), 0.0, scored_predictions, 0

    true_positives = np.asarray(
        [1.0 if item[1] else 0.0 for item in scored_predictions],
        dtype=np.float32,
    )
    false_positives = 1.0 - true_positives
    cumulative_tp = np.cumsum(true_positives)
    cumulative_fp = np.cumsum(false_positives)

    precision = cumulative_tp / np.maximum(
        cumulative_tp + cumulative_fp,
        1e-9,
    )
    recall = cumulative_tp / total_gt

    recall_envelope = np.concatenate(([0.0], recall, [1.0]))
    precision_envelope = np.concatenate(([1.0], precision, [0.0]))
    for index in range(len(precision_envelope) - 2, -1, -1):
        precision_envelope[index] = max(
            precision_envelope[index],
            precision_envelope[index + 1],
        )

    changed_indices = np.where(
        recall_envelope[1:] != recall_envelope[:-1]
    )[0]
    ap = np.sum(
        (
            recall_envelope[changed_indices + 1]
            - recall_envelope[changed_indices]
        )
        * precision_envelope[changed_indices + 1]
    )
    return precision, recall, ap, scored_predictions, total_gt


def compute_f1_vs_conf(scored_predictions, total_gt, confidence_thresholds):
    """confidence閾値ごとのprecision・recall・F1を計算する。"""
    if total_gt == 0:
        zeros = np.zeros_like(confidence_thresholds, dtype=np.float32)
        return zeros, zeros, zeros

    scores = np.asarray(
        [item[0] for item in scored_predictions],
        dtype=np.float32,
    )
    is_true_positive = np.asarray(
        [1.0 if item[1] else 0.0 for item in scored_predictions],
        dtype=np.float32,
    )

    precisions = []
    recalls = []
    f1_scores = []
    for threshold in confidence_thresholds:
        keep = scores >= threshold
        if keep.sum() == 0:
            precision = recall = f1_score = 0.0
        else:
            true_positive = is_true_positive[keep].sum()
            false_positive = keep.sum() - true_positive
            false_negative = total_gt - true_positive
            precision = true_positive / (
                true_positive + false_positive + 1e-9
            )
            recall = true_positive / (
                true_positive + false_negative + 1e-9
            )
            f1_score = (
                2 * precision * recall / (precision + recall + 1e-9)
            )
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    return (
        np.asarray(precisions),
        np.asarray(recalls),
        np.asarray(f1_scores),
    )


def compute_detection_confusion_matrix(
    predictions_by_class,
    ground_truths_by_class,
    image_ids,
    num_classes,
    confidence_threshold,
    iou_threshold,
):
    """全クラスを一括照合し、Backgroundを含む混同行列を作る。"""
    background_index = num_classes
    matrix = np.zeros(
        (num_classes + 1, num_classes + 1),
        dtype=np.int64,
    )

    for image_id in image_ids:
        ground_truths = []
        predictions = []
        for class_id in range(num_classes):
            ground_truths.extend(
                ground_truths_by_class[class_id].get(image_id, [])
            )
            predictions.extend(
                prediction
                for prediction in predictions_by_class[class_id].get(
                    image_id,
                    [],
                )
                if prediction["score"] >= confidence_threshold
            )

        predictions.sort(
            key=lambda prediction: prediction["score"],
            reverse=True,
        )
        matched_gt = [False] * len(ground_truths)

        for prediction in predictions:
            best_iou = -1.0
            best_gt_index = -1
            for gt_index, ground_truth in enumerate(ground_truths):
                if matched_gt[gt_index]:
                    continue
                iou = xyxy_iou(
                    prediction["bbox"],
                    ground_truth["bbox"],
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = gt_index

            predicted_label = prediction["label"]
            if best_gt_index >= 0 and best_iou >= iou_threshold:
                matched_gt[best_gt_index] = True
                true_label = ground_truths[best_gt_index]["label"]
                matrix[true_label, predicted_label] += 1
            else:
                # 対応GTがない予測はfalse positive。
                matrix[background_index, predicted_label] += 1

        for gt_index, ground_truth in enumerate(ground_truths):
            if not matched_gt[gt_index]:
                # 対応予測がないGTはfalse negative。
                matrix[ground_truth["label"], background_index] += 1

    return matrix


def normalize_confusion_matrix(matrix):
    """各正解クラスの合計が1になるよう混同行列を行方向に正規化する。"""
    row_totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals > 0,
    )


def save_confusion_matrix_csv(matrix, labels, output_path, normalized=False):
    """混同行列のカウントまたは正規化値をCSVへ保存する。"""
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["ground_truth\\predicted", *labels])
        for label, row in zip(labels, matrix):
            if normalized:
                writer.writerow([label, *[float(value) for value in row]])
            else:
                writer.writerow([label, *[int(value) for value in row]])


def plot_confusion_matrix(
    matrix,
    labels,
    output_stem,
    title,
    normalized=False,
):
    """混同行列をセル値付きヒートマップとしてPNG・SVGへ保存する。"""
    fig, axis = plt.subplots(figsize=(14, 12))
    image = axis.imshow(matrix, cmap="Blues", vmin=0)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Proportion" if normalized else "Count")

    axis.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted class",
        ylabel="Ground-truth class",
        title=title,
    )
    plt.setp(
        axis.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    maximum = float(np.max(matrix)) if matrix.size else 0.0
    threshold = maximum / 2
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            text = f"{value:.1%}" if normalized else str(int(value))
            axis.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if value > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(f"{output_stem}.png", dpi=300)
    fig.savefig(f"{output_stem}.svg", format="svg")
    plt.close(fig)


def unwrap_dataset_config(dataset_config):
    """RepeatDatasetなどのラッパーから実データセット設定を取り出す。"""
    while "dataset" in dataset_config:
        dataset_config = dataset_config.dataset
    return dataset_config


def build_category_mapping(coco, configured_classes):
    """COCO category_idを設定ファイルのクラス順に内部ラベルへ変換する。"""
    categories = {
        category["name"]: category["id"]
        for category in coco.loadCats(coco.getCatIds())
    }
    missing = [
        class_name
        for class_name in configured_classes
        if class_name not in categories
    ]
    if missing:
        raise ValueError(
            "COCO categoriesに設定クラスがありません: "
            + ", ".join(missing)
        )
    return {
        categories[class_name]: label
        for label, class_name in enumerate(configured_classes)
    }


def interpolate_and_smooth_pr(recall, precision, recall_grid):
    """PR曲線を共通recallグリッドへ補間して平滑化する。"""
    if len(recall) == 0:
        return np.zeros_like(recall_grid)

    recall_extended = np.concatenate(
        ([0.0], recall, [recall[-1] + 1e-5, 1.0])
    )
    precision_extended = np.concatenate(
        ([precision[0]], precision, [0.0, 0.0])
    )
    sort_indices = np.argsort(recall_extended)
    recall_extended = recall_extended[sort_indices]
    precision_extended = precision_extended[sort_indices]

    precision_interpolated = np.interp(
        recall_grid,
        recall_extended,
        precision_extended,
    )
    precision_smoothed = gaussian_filter1d(
        precision_interpolated,
        sigma=SMOOTH_SIGMA_PR,
    )
    precision_smoothed[0] = precision_interpolated[0]
    precision_smoothed[-1] = 0.0
    return np.clip(precision_smoothed, 0.0, 1.0)


# =========================
# 3. COCO GT読み込み
# =========================
cfg = Config.fromfile(CONFIG_FILE)
if SPLIT == "val":
    dataset_cfg = unwrap_dataset_config(cfg.val_dataloader.dataset)
else:
    dataset_cfg = unwrap_dataset_config(cfg.test_dataloader.dataset)

annotation_file = dataset_cfg.ann_file
data_root = dataset_cfg.data_root
image_prefix = dataset_cfg.data_prefix["img"]
annotation_path = (
    os.path.join(data_root, annotation_file)
    if not os.path.isabs(annotation_file)
    else annotation_file
)

coco = COCO(annotation_path)
configured_classes = tuple(dataset_cfg.metainfo["classes"])
num_classes = len(configured_classes)
if num_classes != len(CURRENT_CLASS_NAMES):
    raise ValueError(
        f"現在のクラス数は{len(CURRENT_CLASS_NAMES)}を想定していますが、"
        f"設定ファイルは{num_classes}クラスです: {configured_classes}"
    )

# 設定内のカテゴリ名が数字でも、結果表示には現在の意味名を使う。
class_names = CURRENT_CLASS_NAMES
category_to_label = build_category_mapping(coco, configured_classes)
important_class_ids = [
    class_names.index(class_name)
    for class_name in IMPORTANT_CLASS_NAMES
]

print(f"num_classes used for evaluation = {num_classes}")
print("configured classes =", configured_classes)
print("display class names =", class_names)
print("important classes =", IMPORTANT_CLASS_NAMES)

image_ids = coco.getImgIds()
image_infos = coco.loadImgs(image_ids)
ground_truth_by_class = {
    class_id: defaultdict(list)
    for class_id in range(num_classes)
}

for image_info in image_infos:
    image_id = image_info["id"]
    annotation_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=None)
    for annotation in coco.loadAnns(annotation_ids):
        category_id = annotation["category_id"]
        if category_id not in category_to_label:
            continue
        label = category_to_label[category_id]
        x, y, width, height = annotation["bbox"]
        ground_truth_by_class[label][image_id].append({
            "bbox": [x, y, x + width, y + height],
            "label": label,
        })


# =========================
# 4. 推論
# =========================
model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
predictions_by_class = {
    class_id: defaultdict(list)
    for class_id in range(num_classes)
}

for index, image_info in enumerate(image_infos, start=1):
    image_id = image_info["id"]
    image_path = os.path.join(
        data_root,
        image_prefix,
        image_info["file_name"],
    )
    result = inference_detector(model, image_path)
    prediction_instances = result.pred_instances
    boxes = prediction_instances.bboxes.detach().cpu().numpy()
    scores = prediction_instances.scores.detach().cpu().numpy()
    labels = prediction_instances.labels.detach().cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        label = int(label)
        if 0 <= label < num_classes:
            predictions_by_class[label][image_id].append({
                "bbox": box.tolist(),
                "score": float(score),
                "label": label,
            })

    if index % 50 == 0 or index == len(image_infos):
        print(f"Inference {index}/{len(image_infos)}")


# =========================
# 5. クラス別PR・F1計算
# =========================
class_results = {}
for class_id in range(num_classes):
    precision, recall, ap, scored_predictions, total_gt = compute_pr_curve(
        predictions_by_class[class_id],
        ground_truth_by_class[class_id],
        image_ids,
        iou_threshold=IOU_THRESH,
    )
    precision_at_threshold, recall_at_threshold, f1_at_threshold = (
        compute_f1_vs_conf(
            scored_predictions,
            total_gt,
            CONF_THRESHOLDS,
        )
    )
    class_results[class_id] = {
        "precision_curve": precision,
        "recall_curve": recall,
        "ap": ap,
        "scored_predictions": scored_predictions,
        "total_gt": total_gt,
        "precision_vs_threshold": precision_at_threshold,
        "recall_vs_threshold": recall_at_threshold,
        "f1_vs_threshold": f1_at_threshold,
    }


# =========================
# 6. 平均指標と最適confidence
# =========================
mean_f1 = np.mean(
    np.stack([
        class_results[class_id]["f1_vs_threshold"]
        for class_id in important_class_ids
    ]),
    axis=0,
)
mean_ap = float(np.mean([
    class_results[class_id]["ap"]
    for class_id in important_class_ids
]))
best_index = int(np.argmax(mean_f1))
best_confidence = float(CONF_THRESHOLDS[best_index])
best_mean_f1 = float(mean_f1[best_index])

print(
    f"\nBest confidence (mean over selected classes) = "
    f"{best_confidence:.3f}"
)
print(f"Best mean F1 = {best_mean_f1:.4f}")
print(f"Mean AP = {mean_ap:.4f}\n")

for class_id in important_class_ids:
    result = class_results[class_id]
    class_best_index = int(np.argmax(result["f1_vs_threshold"]))
    print(
        f"Class [{class_names[class_id]}]: "
        f"AP={result['ap']:.4f}, "
        f"best_conf={CONF_THRESHOLDS[class_best_index]:.3f}, "
        f"best_F1={result['f1_vs_threshold'][class_best_index]:.4f}, "
        f"num_gt={result['total_gt']}"
    )


# =========================
# 7. Confusion matrix
# =========================
confusion_labels = [*class_names, "Background"]
confusion_matrix_counts = compute_detection_confusion_matrix(
    predictions_by_class,
    ground_truth_by_class,
    image_ids,
    num_classes,
    confidence_threshold=best_confidence,
    iou_threshold=IOU_THRESH,
)
confusion_matrix_normalized = normalize_confusion_matrix(
    confusion_matrix_counts
)

save_confusion_matrix_csv(
    confusion_matrix_counts,
    confusion_labels,
    os.path.join(OUT_DIR, "C_confusion_matrix_counts.csv"),
)
save_confusion_matrix_csv(
    confusion_matrix_normalized,
    confusion_labels,
    os.path.join(OUT_DIR, "D_confusion_matrix_normalized.csv"),
    normalized=True,
)
plot_confusion_matrix(
    confusion_matrix_counts,
    confusion_labels,
    os.path.join(OUT_DIR, "C_confusion_matrix_counts"),
    (
        f"Detection Confusion Matrix "
        f"(confidence={best_confidence:.2f}, IoU={IOU_THRESH:.2f})"
    ),
)
plot_confusion_matrix(
    confusion_matrix_normalized,
    confusion_labels,
    os.path.join(OUT_DIR, "D_confusion_matrix_normalized"),
    (
        f"Normalized Detection Confusion Matrix "
        f"(confidence={best_confidence:.2f}, IoU={IOU_THRESH:.2f})"
    ),
    normalized=True,
)


# =========================
# 8. F1 vs Confidence
# =========================
plt.figure(figsize=(10, 7))
for class_id in important_class_ids:
    class_name = class_names[class_id]
    f1_values = class_results[class_id]["f1_vs_threshold"]
    f1_smoothed = gaussian_filter1d(
        f1_values,
        sigma=SMOOTH_SIGMA_F1,
    )
    plt.plot(
        CONF_THRESHOLDS,
        np.clip(f1_smoothed, 0.0, 1.0),
        linewidth=2.5,
        color=CLASS_COLORS[class_name],
        alpha=0.9,
        label=class_name,
    )

plt.axvline(
    best_confidence,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label=f"Best mean confidence={best_confidence:.2f}",
)
near_best = mean_f1 >= best_mean_f1 - 0.02
if near_best.any():
    minimum_index = np.where(near_best)[0][0]
    maximum_index = np.where(near_best)[0][-1]
    plt.axvspan(
        CONF_THRESHOLDS[minimum_index],
        CONF_THRESHOLDS[maximum_index],
        color="#B9DDF2",
        alpha=0.25,
    )

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Confidence", fontsize=14)
plt.ylabel("F1 Score", fontsize=14)
plt.title("F1 Score vs Confidence", fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10, loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig(
    os.path.join(OUT_DIR, "A_f1_vs_confidence.png"),
    dpi=300,
)
plt.savefig(
    os.path.join(OUT_DIR, "A_f1_vs_confidence.svg"),
    format="svg",
)
plt.close()


# =========================
# 9. Precision-Recall
# =========================
recall_grid = np.linspace(0, 1, 500)
plt.figure(figsize=(10, 7))
for class_id in important_class_ids:
    class_name = class_names[class_id]
    precision = class_results[class_id]["precision_curve"]
    recall = class_results[class_id]["recall_curve"]
    if len(precision) == 0:
        continue
    precision_smoothed = interpolate_and_smooth_pr(
        recall,
        precision,
        recall_grid,
    )
    plt.plot(
        recall_grid,
        precision_smoothed,
        linewidth=2.5,
        color=CLASS_COLORS[class_name],
        alpha=0.9,
        label=f"{class_name} (AP={class_results[class_id]['ap']:.3f})",
    )

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Recall", fontsize=14)
plt.ylabel("Precision", fontsize=14)
plt.title("Precision-Recall Curve", fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10, loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig(
    os.path.join(OUT_DIR, "B_precision_recall.png"),
    dpi=300,
)
plt.savefig(
    os.path.join(OUT_DIR, "B_precision_recall.svg"),
    format="svg",
)
plt.close()


# =========================
# 10. 結果保存
# =========================
summary = {
    "config_file": CONFIG_FILE,
    "checkpoint_file": CHECKPOINT_FILE,
    "split": SPLIT,
    "iou_threshold": IOU_THRESH,
    "configured_classes": configured_classes,
    "current_class_names": class_names,
    "evaluated_class_names": IMPORTANT_CLASS_NAMES,
    "best_confidence_mean_selected": best_confidence,
    "best_f1_mean_selected": best_mean_f1,
    "mean_ap_selected": mean_ap,
    "confusion_matrix_confidence": best_confidence,
    "confusion_matrix_labels": confusion_labels,
    "confusion_matrix_counts": confusion_matrix_counts.tolist(),
    "class_results": {},
}

for class_id in range(num_classes):
    result = class_results[class_id]
    class_best_index = int(np.argmax(result["f1_vs_threshold"]))
    summary["class_results"][class_names[class_id]] = {
        "class_id": class_id,
        "included_in_selected_mean": class_id in important_class_ids,
        "ap": float(result["ap"]),
        "best_confidence": float(
            CONF_THRESHOLDS[class_best_index]
        ),
        "best_f1": float(
            result["f1_vs_threshold"][class_best_index]
        ),
        "num_gt": int(result["total_gt"]),
    }

with open(
    os.path.join(OUT_DIR, "summary.json"),
    "w",
    encoding="utf-8",
) as file:
    json.dump(summary, file, ensure_ascii=False, indent=2)

print(f"\nSaved to: {OUT_DIR}")
