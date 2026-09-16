"""Evaluate a saved project Mask R-CNN checkpoint on a COCO dataset."""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from models.maskrcnn import get_model
from train import (
    collate_fn,
    evaluate_coco_bbox,
    evaluate_coco_segm,
    print_class_metrics,
)
from utils.checkpoint import load_training_checkpoint
from utils.dataset import CocoDiatomDataset
from utils.metrics_logger import (
    append_class_metrics_csv,
    init_class_metrics_csv,
    save_class_ap_plot,
    save_class_metrics_table,
)
from utils.threshold_metrics import (
    evaluate_threshold_metrics,
    make_score_thresholds,
    print_threshold_summary,
    save_threshold_metrics,
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
    parser.add_argument(
        "--threshold_mode",
        choices=["none", "sweep", "fixed"],
        default="none",
        help="Sweep validation thresholds or evaluate one fixed test threshold",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=None,
        help="Confidence threshold required by --threshold_mode fixed",
    )
    parser.add_argument("--threshold_start", type=float, default=0.05)
    parser.add_argument("--threshold_stop", type=float, default=0.95)
    parser.add_argument("--threshold_step", type=float, default=0.05)
    parser.add_argument("--match_iou_thresh", type=float, default=0.5)
    parser.add_argument(
        "--threshold_selection_metric",
        choices=["macro_f1", "micro_f1"],
        default="macro_f1",
        help="Validation-only score-threshold selection criterion",
    )
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
    if args.threshold_mode == "fixed" and args.score_thresh is None:
        raise ValueError("--score_thresh is required for --threshold_mode fixed")
    if args.score_thresh is not None and not 0.0 <= args.score_thresh <= 1.0:
        raise ValueError("--score_thresh must be between 0 and 1")

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

    print("Running COCO segm evaluation...")
    segm_metrics = evaluate_coco_segm(
        model,
        data_loader,
        device,
        score_thresh=0.0,
        mask_thresh=args.eval_mask_thresh,
    )
    print_class_metrics(bbox_metrics, segm_metrics)

    threshold_outputs = None
    if args.threshold_mode != "none":
        if args.threshold_mode == "sweep":
            score_thresholds = make_score_thresholds(
                args.threshold_start,
                args.threshold_stop,
                args.threshold_step,
            )
            print(
                "Running validation threshold sweep: "
                f"{score_thresholds[0]:.2f}-{score_thresholds[-1]:.2f}"
            )
        else:
            score_thresholds = [args.score_thresh]
            print(f"Running fixed test threshold evaluation: {args.score_thresh:.2f}")

        threshold_results = evaluate_threshold_metrics(
            model,
            data_loader,
            device,
            dataset.category_names,
            score_thresholds,
            match_iou_threshold=args.match_iou_thresh,
            mask_threshold=args.eval_mask_thresh,
        )
        threshold_csv_path, threshold_summary_path, threshold_summary = (
            save_threshold_metrics(
                threshold_results,
                args.output_dir,
                args.threshold_mode,
                selection_metric=args.threshold_selection_metric,
            )
        )
        print_threshold_summary(threshold_summary)
        threshold_outputs = {
            "mode": args.threshold_mode,
            "csv": os.path.abspath(threshold_csv_path),
            "summary": os.path.abspath(threshold_summary_path),
        }

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
    table_path = save_class_metrics_table(
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
                "threshold_evaluation": threshold_outputs,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved: {class_metrics_path}")
    print(f"Saved: {summary_path}")
    if plot_path is not None:
        print(f"Saved: {plot_path}")
    if table_path is not None:
        print(f"Saved: {table_path}")
    if threshold_outputs is not None:
        print(f"Saved: {threshold_outputs['csv']}")
        print(f"Saved: {threshold_outputs['summary']}")


if __name__ == "__main__":
    main()
