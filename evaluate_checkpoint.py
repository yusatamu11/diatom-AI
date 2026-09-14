"""Evaluate a saved project Mask R-CNN checkpoint on a COCO validation set."""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from models.maskrcnn import get_model
from train import collate_fn, evaluate_coco_bbox, evaluate_coco_segm, print_class_metrics
from utils.checkpoint import load_training_checkpoint
from utils.dataset import CocoDiatomDataset
from utils.metrics_logger import (
    append_class_metrics_csv,
    init_class_metrics_csv,
    save_class_ap_plot,
)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--ann_file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default=None, help="For example cuda:0 or cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_mask_thresh", type=float, default=0.5)
    return parser.parse_args()


def resolve_device(requested):
    if requested is not None:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def json_ready(metrics):
    """Convert integer-keyed per-class mappings into ordinary JSON objects."""
    if metrics is None:
        return None
    result = dict(metrics)
    result["per_class"] = {
        str(category_id): values
        for category_id, values in metrics["per_class"].items()
    }
    return result


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    dataset = CocoDiatomDataset(args.image_dir, args.ann_file)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )
    state_dict, num_classes, category_names, metadata = load_training_checkpoint(
        args.checkpoint,
        map_location=device,
    )
    if num_classes != dataset.num_classes:
        raise ValueError(
            "Checkpoint and validation dataset class counts do not match: "
            f"checkpoint={num_classes}, dataset={dataset.num_classes}"
        )
    if category_names and category_names != dataset.category_names:
        raise ValueError(
            "Checkpoint and validation category definitions do not match: "
            f"checkpoint={category_names}, dataset={dataset.category_names}"
        )

    model = get_model(num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    epoch = int(metadata.get("epoch") or 0)

    print("Running COCO bbox evaluation...")
    bbox_metrics = evaluate_coco_bbox(model, data_loader, device, score_thresh=0.0)
    print_class_metrics(bbox_metrics, "bbox")

    print("Running COCO segm evaluation...")
    segm_metrics = evaluate_coco_segm(
        model,
        data_loader,
        device,
        score_thresh=0.0,
        mask_thresh=args.eval_mask_thresh,
    )
    print_class_metrics(segm_metrics, "segm")

    class_metrics_path = init_class_metrics_csv(args.output_dir)
    append_class_metrics_csv(
        class_metrics_path,
        epoch,
        bbox_metrics,
        segm_metrics,
    )
    plot_path = save_class_ap_plot(
        args.output_dir,
        epoch,
        bbox_metrics,
        segm_metrics,
    )
    summary_path = os.path.join(args.output_dir, "evaluation_metrics.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "checkpoint": os.path.abspath(args.checkpoint),
                "epoch": epoch,
                "bbox": json_ready(bbox_metrics),
                "segm": json_ready(segm_metrics),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved: {class_metrics_path}")
    print(f"Saved: {summary_path}")
    if plot_path is not None:
        print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
