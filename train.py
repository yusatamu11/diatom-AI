import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils
from models.maskrcnn import get_model
from utils.checkpoint import make_training_checkpoint
from utils.dataset import CocoDiatomDataset
from utils.metrics_logger import (
    append_class_metrics_csv,
    append_metrics_csv,
    init_class_metrics_csv,
    init_metrics_csv,
    save_class_ap_plot,
    save_class_metrics_table,
)



def collate_fn(batch):
    return tuple(zip(*batch))


def _mean_valid(values):
    """Average COCO values while ignoring the -1 missing-value sentinel."""
    values = np.asarray(values, dtype=float)
    values = values[values > -1]
    if values.size == 0:
        return None
    return float(values.mean())


def _iou_index(coco_eval, threshold):
    matches = np.flatnonzero(np.isclose(coco_eval.params.iouThrs, threshold))
    if len(matches) != 1:
        raise ValueError(f"COCO IoU threshold not found: {threshold}")
    return int(matches[0])


def extract_coco_metrics(coco_eval, coco_gt, coco_results):
    """Return overall and per-category metrics from an accumulated COCOeval."""
    area_index = coco_eval.params.areaRngLbl.index("all")
    max_dets_index = coco_eval.params.maxDets.index(100)
    iou50_index = _iou_index(coco_eval, 0.50)
    iou75_index = _iou_index(coco_eval, 0.75)
    precision = coco_eval.eval["precision"]
    recall = coco_eval.eval["recall"]

    prediction_counts = {}
    for result in coco_results:
        category_id = int(result["category_id"])
        prediction_counts[category_id] = prediction_counts.get(category_id, 0) + 1

    per_class = {}
    for category_index, category_id in enumerate(coco_eval.params.catIds):
        category_id = int(category_id)
        category = coco_gt.cats[category_id]
        per_class[category_id] = {
            "class_name": str(category["name"]),
            "gt_count": len(
                coco_gt.getAnnIds(
                    imgIds=coco_eval.params.imgIds,
                    catIds=[category_id],
                    iscrowd=False,
                )
            ),
            "prediction_count": prediction_counts.get(category_id, 0),
            "AP": _mean_valid(
                precision[:, :, category_index, area_index, max_dets_index]
            ),
            "AP50": _mean_valid(
                precision[
                    iou50_index, :, category_index, area_index, max_dets_index
                ]
            ),
            "AP75": _mean_valid(
                precision[
                    iou75_index, :, category_index, area_index, max_dets_index
                ]
            ),
            "AR100": _mean_valid(
                recall[:, category_index, area_index, max_dets_index]
            ),
        }

    return {
        "AP": float(coco_eval.stats[0]),
        "AP50": float(coco_eval.stats[1]),
        "AP75": float(coco_eval.stats[2]),
        "AP_small": float(coco_eval.stats[3]),
        "AP_medium": float(coco_eval.stats[4]),
        "AP_large": float(coco_eval.stats[5]),
        "per_class": per_class,
    }


def print_class_metrics(bbox_metrics, segm_metrics):
    """Print one compact table combining bbox and segmentation results."""
    bbox_by_class = (bbox_metrics or {}).get("per_class", {})
    segm_by_class = (segm_metrics or {}).get("per_class", {})
    category_ids = sorted(set(bbox_by_class) | set(segm_by_class))
    if not category_ids:
        return

    rows = []
    for category_id in category_ids:
        bbox = bbox_by_class.get(category_id, {})
        segm = segm_by_class.get(category_id, {})
        category = segm or bbox
        rows.append(
            [
                category.get("class_name", f"class_{category_id}"),
                str(max(bbox.get("gt_count", 0), segm.get("gt_count", 0))),
                str(max(
                    bbox.get("prediction_count", 0),
                    segm.get("prediction_count", 0),
                )),
                _format_console_metric(bbox.get("AP")),
                _format_console_metric(segm.get("AP")),
                _format_console_metric(segm.get("AP50")),
                _format_console_metric(segm.get("AP75")),
                _format_console_metric(segm.get("AR100")),
            ]
        )

    headers = [
        "class", "GT", "pred", "bbox AP", "segm AP", "AP50", "AP75", "AR100"
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def render(row):
        cells = [f"{row[0]:<{widths[0]}}"]
        cells.extend(
            f"{value:>{widths[index]}}" for index, value in enumerate(row[1:], 1)
        )
        return "| " + " | ".join(cells) + " |"

    print("\nValidation class-wise metrics")
    print(separator)
    print(render(headers))
    print(separator)
    for row in rows:
        print(render(row))
    print(separator)


def _format_console_metric(value):
    return "n/a" if value is None else f"{value:.3f}"

# validationのlossを計算する関数．model.eval()では計算できないので，model.train()にして計算する．
@torch.no_grad()
def evaluate_loss(model, data_loader, device):
    was_training = model.training
    model.train()

    total_loss = 0.0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [
            {k: v.to(device) for k, v in t.items()}
            for t in targets
        ]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        total_loss += losses.item()

    avg_loss = total_loss / len(data_loader)

    if not was_training:
        model.eval()

    return avg_loss

# bboxの評価を行う関数．COCOevalを用いてmAPを計算する．
@torch.no_grad()
def evaluate_coco_bbox(model, data_loader, device, score_thresh=0.0):
    model.eval()

    coco_gt = data_loader.dataset.coco
    coco_results = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]

        outputs = model(images)

        for target, output in zip(targets, outputs):
            image_id = int(target["image_id"].item())

            boxes = output["boxes"].detach().cpu()
            labels = output["labels"].detach().cpu()
            scores = output["scores"].detach().cpu()

            keep = scores >= score_thresh
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]

            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = box.tolist()
                w = x2 - x1
                h = y2 - y1

                if w <= 0 or h <= 0:
                    continue

                coco_results.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, w, h],
                        "score": float(score),
                    }
                )

    if len(coco_results) == 0:
        print("No validation detections.")
        return None

    coco_dt = coco_gt.loadRes(coco_results)

    # mAP 計算
    coco_eval = COCOeval(
        coco_gt,
        coco_dt,
        iouType="bbox",
    )
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return extract_coco_metrics(coco_eval, coco_gt, coco_results)

# segmentationの評価を行う関数．COCOevalを用いてmAPを計算する．
@torch.no_grad()
def evaluate_coco_segm(
    model,
    data_loader,
    device,
    score_thresh=0.0,
    mask_thresh=0.5,
):
    model.eval()

    coco_gt = data_loader.dataset.coco
    coco_results = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]

        outputs = model(images)

        for target, output in zip(targets, outputs):
            image_id = int(target["image_id"].item())

            labels = output["labels"].detach().cpu()
            scores = output["scores"].detach().cpu()
            masks = output["masks"].detach().cpu()

            keep = scores >= score_thresh
            labels = labels[keep]
            scores = scores[keep]
            masks = masks[keep]

            for mask, label, score in zip(masks, labels, scores):
                if mask.ndim == 3:
                    mask = mask[0]

                binary_mask = (mask >= mask_thresh).numpy().astype(np.uint8)

                if binary_mask.sum() == 0:
                    continue

                rle = mask_utils.encode(
                    np.asfortranarray(binary_mask)
                )
                rle["counts"] = rle["counts"].decode("utf-8")

                coco_results.append(
                    {
                        "image_id": image_id,
                        "category_id": int(label),
                        "segmentation": rle,
                        "score": float(score),
                    }
                )

    if len(coco_results) == 0:
        print("No validation mask detections.")
        return None

    coco_dt = coco_gt.loadRes(coco_results)

    coco_eval = COCOeval(
        coco_gt,
        coco_dt,
        iouType="segm",
    )
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return extract_coco_metrics(coco_eval, coco_gt, coco_results)

def get_args():
    parser = argparse.ArgumentParser()#インスタンス(オブジェクト)を作成
    
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Training image directory",
    )
        
    parser.add_argument(
        "--ann_file",
        type=str,
        required=True,
        help="COCO annotation file for training",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=0.005,
    )
    
    parser.add_argument(
        "--val_image_dir",
        type=str,
        default=None,
        help="Validation image directory",
    )

    parser.add_argument(
        "--val_ann_file",
        type=str,
        default=None,
        help="COCO annotation file for validation",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs",
        help="Directory to save model checkpoints",
    )
    
    parser.add_argument(
        "--eval_mask_thresh",
        type=float,
        default=0.5,
        help="Mask threshold used for COCO segmentation evaluation",
    )
    
    return parser.parse_args()


def main():
    args = get_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    metrics_csv_path = init_metrics_csv(args.output_dir)
    class_metrics_csv_path = init_class_metrics_csv(args.output_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Dataset
    train_dataset = CocoDiatomDataset(
        args.image_dir,
        args.ann_file,
    )
    num_classes = train_dataset.num_classes
    print(f"Model classes (including background): {num_classes}")
    print(f"Foreground categories: {train_dataset.category_names}")

    # DataLoader for training
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
    )
    
    val_loader = None

    if args.val_image_dir is not None and args.val_ann_file is not None:
        val_dataset = CocoDiatomDataset(
            args.val_image_dir,
            args.val_ann_file,
        )
        if val_dataset.category_names != train_dataset.category_names:
            raise ValueError(
                "Training and validation category definitions do not match. "
                f"train={train_dataset.category_names}, "
                f"validation={val_dataset.category_names}"
            )

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn,
        )
    
    # Model
    model = get_model(num_classes)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005,
    )
    
    
    best_segm_ap = float("-inf")

    # Training
    for epoch in range(args.epochs):
        model.train()

        val_loss = None
        bbox_metrics = None
        segm_metrics = None

        epoch_loss = 0.0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [
                {k: v.to(device) for k, v in t.items()}
                for t in targets
            ]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{args.epochs}], train loss: {avg_loss:.4f}")

        if val_loader is not None:
            val_loss = evaluate_loss(
                model,
                val_loader,
                device,
            )
            print(f"Validation loss: {val_loss:.4f}")

            print("Running COCO bbox evaluation...")
            bbox_metrics = evaluate_coco_bbox(
                model,
                val_loader,
                device,
                score_thresh=0.0,
            )

            if bbox_metrics is not None:
                print(
                    "Validation bbox metrics: "
                    f"AP={bbox_metrics['AP']:.4f}, "
                    f"AP50={bbox_metrics['AP50']:.4f}, "
                    f"AP75={bbox_metrics['AP75']:.4f}"
                )

            print("Running COCO segm evaluation...")
            segm_metrics = evaluate_coco_segm(
                model,
                val_loader,
                device,
                score_thresh=0.0,
                mask_thresh=args.eval_mask_thresh,
            )

            if segm_metrics is not None:
                print(
                    "Validation segm metrics: "
                    f"AP={segm_metrics['AP']:.4f}, "
                    f"AP50={segm_metrics['AP50']:.4f}, "
                    f"AP75={segm_metrics['AP75']:.4f}"
                )

            print_class_metrics(bbox_metrics, segm_metrics)

        save_path = os.path.join(
            args.output_dir,
            f"maskrcnn_epoch_{epoch + 1}.pth"
        )

        checkpoint = make_training_checkpoint(
            model=model,
            category_names=train_dataset.category_names,
            epoch=epoch + 1,
        )
        torch.save(checkpoint, save_path)
        print(f"Saved: {save_path}")

        if segm_metrics is not None and segm_metrics["AP"] > best_segm_ap:
            best_segm_ap = segm_metrics["AP"]
            best_model_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(checkpoint, best_model_path)
            print(
                f"New best model: {best_model_path} "
                f"(segm AP={best_segm_ap:.4f})"
            )

        append_metrics_csv(
            metrics_csv_path=metrics_csv_path,
            epoch=epoch + 1,
            train_loss=avg_loss,
            val_loss=val_loss,
            bbox_metrics=bbox_metrics,
            segm_metrics=segm_metrics,
            checkpoint_path=save_path,
        )
        print(f"Metrics saved to: {metrics_csv_path}")

        if val_loader is not None:
            append_class_metrics_csv(
                class_metrics_csv_path=class_metrics_csv_path,
                epoch=epoch + 1,
                bbox_metrics=bbox_metrics,
                segm_metrics=segm_metrics,
            )
            plot_path = save_class_ap_plot(
                output_dir=args.output_dir,
                epoch=epoch + 1,
                bbox_metrics=bbox_metrics,
                segm_metrics=segm_metrics,
            )
            table_path = save_class_metrics_table(
                output_dir=args.output_dir,
                epoch=epoch + 1,
                bbox_metrics=bbox_metrics,
                segm_metrics=segm_metrics,
            )
            print(f"Class metrics saved to: {class_metrics_csv_path}")
            if plot_path is not None:
                print(f"Class AP plot saved to: {plot_path}")
            if table_path is not None:
                print(f"Class metrics table saved to: {table_path}")


if __name__ == "__main__":
    main()
