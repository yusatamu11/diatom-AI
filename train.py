import os
import argparse
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils
from models.maskrcnn import get_model
from utils.checkpoint import make_training_checkpoint
from utils.dataset import (
    BasicDiatomAugmentation,
    CocoDiatomDataset,
    make_class_balanced_sample_weights,
)
from utils.metrics_logger import (
    append_class_metrics_csv,
    append_metrics_csv,
    init_class_metrics_csv,
    init_metrics_csv,
    save_class_ap_plot,
    save_class_metrics_table,
)


LOSS_NAMES = (
    "loss_classifier",
    "loss_box_reg",
    "loss_mask",
    "loss_objectness",
    "loss_rpn_box_reg",
)


def collate_fn(batch):
    return tuple(zip(*batch))


def seed_everything(seed):
    """Seed training and data-loader randomness for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """Seed NumPy and Python inside each data-loader worker."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


def update_early_stopping(
    current,
    reference_best,
    epochs_without_improvement,
    min_delta,
):
    """Update early-stopping state using a minimum meaningful improvement."""
    if reference_best is None or current > reference_best + min_delta:
        return current, 0, True
    return reference_best, epochs_without_improvement + 1, False


def average_loss_sums(loss_sums, image_count):
    """Convert image-weighted loss sums into per-image epoch averages."""
    if image_count <= 0:
        return {}
    return {name: value / image_count for name, value in loss_sums.items()}


# validationのlossを計算する関数．model.eval()では計算できないので，model.train()にして計算する．
@torch.no_grad()
def evaluate_loss(model, data_loader, device):
    was_training = model.training
    model.train()

    total_loss_sum = 0.0
    component_loss_sums = {}
    image_count = 0

    for images, targets in data_loader:
        batch_size = len(images)
        images = [img.to(device) for img in images]
        targets = [
            {k: v.to(device) for k, v in t.items()}
            for t in targets
        ]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        total_loss_sum += losses.item() * batch_size
        for name in LOSS_NAMES:
            if name in loss_dict:
                component_loss_sums[name] = component_loss_sums.get(name, 0.0) + (
                    loss_dict[name].item() * batch_size
                )
        image_count += batch_size

    avg_loss = total_loss_sum / image_count
    avg_component_losses = average_loss_sums(component_loss_sums, image_count)

    if not was_training:
        model.eval()

    return avg_loss, avg_component_losses

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
        "--augmentation",
        choices=["none", "basic"],
        default="none",
        help="Training-only synchronized image/mask augmentation",
    )

    parser.add_argument(
        "--balanced_sampling",
        action="store_true",
        help="Sample images containing rare classes more frequently",
    )

    parser.add_argument(
        "--balanced_sampling_max_weight",
        type=float,
        default=5.0,
        help="Maximum image sampling weight for rare classes",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for model initialization, sampling, and augmentation",
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

    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=15,
        help=(
            "Stop after this many validation epochs without a meaningful segm "
            "AP improvement; set to 0 to disable"
        ),
    )

    parser.add_argument(
        "--early_stopping_min_epochs",
        type=int,
        default=20,
        help="Always train for at least this many epochs before early stopping",
    )

    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.001,
        help="Minimum segm AP increase counted as an early-stopping improvement",
    )
    
    return parser.parse_args()


def main():
    args = get_args()

    if args.early_stopping_patience < 0:
        raise ValueError("--early_stopping_patience must be 0 or greater")
    if args.early_stopping_min_epochs < 1:
        raise ValueError("--early_stopping_min_epochs must be 1 or greater")
    if args.early_stopping_min_delta < 0:
        raise ValueError("--early_stopping_min_delta must be 0 or greater")
    if args.balanced_sampling_max_weight < 1.0:
        raise ValueError("--balanced_sampling_max_weight must be 1.0 or greater")

    seed_everything(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)

    config_path = os.path.join(args.output_dir, "training_config.json")
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"Training configuration saved to: {config_path}")

    metrics_csv_path = init_metrics_csv(args.output_dir)
    class_metrics_csv_path = init_class_metrics_csv(args.output_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Dataset
    train_transform = (
        BasicDiatomAugmentation() if args.augmentation == "basic" else None
    )
    train_dataset = CocoDiatomDataset(
        args.image_dir,
        args.ann_file,
        transform=train_transform,
    )
    num_classes = train_dataset.num_classes
    print(f"Model classes (including background): {num_classes}")
    print(f"Foreground categories: {train_dataset.category_names}")

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_sampler = None
    if args.balanced_sampling:
        (
            sample_weights,
            annotation_counts,
            category_weights,
        ) = make_class_balanced_sample_weights(
            train_dataset,
            max_weight=args.balanced_sampling_max_weight,
        )
        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(args.seed + 1)
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
            generator=sampler_generator,
        )
        print("Class-balanced sampling enabled:")
        for category_id, class_name in train_dataset.category_names.items():
            print(
                f"  {class_name}: {annotation_counts[category_id]} annotations, "
                f"weight={category_weights[category_id]:.2f}"
            )
        print(
            "  image weight range: "
            f"{sample_weights.min().item():.2f}-"
            f"{sample_weights.max().item():.2f}"
        )

    print(f"Training augmentation: {args.augmentation}")
    print(f"Random seed: {args.seed}")

    # DataLoader for training
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=2,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=train_generator,
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
    early_stopping_best = None
    epochs_without_improvement = 0
    early_stopping_enabled = (
        val_loader is not None and args.early_stopping_patience > 0
    )
    if early_stopping_enabled:
        print(
            "Early stopping enabled: "
            f"metric=segm AP, min_epochs={args.early_stopping_min_epochs}, "
            f"patience={args.early_stopping_patience}, "
            f"min_delta={args.early_stopping_min_delta}"
        )
    elif args.early_stopping_patience == 0:
        print("Early stopping disabled (--early_stopping_patience=0).")
    else:
        print("Early stopping disabled because no validation dataset was supplied.")

    # Training
    for epoch in range(args.epochs):
        model.train()

        val_loss = None
        val_loss_components = None
        bbox_metrics = None
        segm_metrics = None

        epoch_loss_sum = 0.0
        train_component_loss_sums = {}
        train_image_count = 0

        for images, targets in train_loader:
            batch_size = len(images)
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

            epoch_loss_sum += losses.item() * batch_size
            for name in LOSS_NAMES:
                if name in loss_dict:
                    train_component_loss_sums[name] = train_component_loss_sums.get(
                        name, 0.0
                    ) + (
                        loss_dict[name].item() * batch_size
                    )
            train_image_count += batch_size
            
        avg_loss = epoch_loss_sum / train_image_count
        train_loss_components = average_loss_sums(
            train_component_loss_sums,
            train_image_count,
        )
        print(f"Epoch [{epoch + 1}/{args.epochs}], train loss: {avg_loss:.4f}")

        if val_loader is not None:
            val_loss, val_loss_components = evaluate_loss(
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

        stop_training = False
        if early_stopping_enabled and segm_metrics is not None:
            early_stopping_best, epochs_without_improvement, improved = (
                update_early_stopping(
                    current=segm_metrics["AP"],
                    reference_best=early_stopping_best,
                    epochs_without_improvement=epochs_without_improvement,
                    min_delta=args.early_stopping_min_delta,
                )
            )
            if epoch + 1 <= args.early_stopping_min_epochs:
                epochs_without_improvement = 0
            elif improved:
                print(
                    "Early stopping: meaningful improvement "
                    f"(reference segm AP={early_stopping_best:.4f})."
                )
            else:
                print(
                    "Early stopping: no meaningful improvement for "
                    f"{epochs_without_improvement}/"
                    f"{args.early_stopping_patience} epoch(s)."
                )
                stop_training = (
                    epochs_without_improvement >= args.early_stopping_patience
                )

        append_metrics_csv(
            metrics_csv_path=metrics_csv_path,
            epoch=epoch + 1,
            train_loss=avg_loss,
            val_loss=val_loss,
            train_loss_components=train_loss_components,
            val_loss_components=val_loss_components,
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

        if stop_training:
            print(
                f"Early stopping at epoch {epoch + 1}. "
                f"Best segm AP={best_segm_ap:.4f}; "
                "best weights are saved in best_model.pth."
            )
            break


if __name__ == "__main__":
    main()
