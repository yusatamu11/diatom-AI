import os
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils
from models.maskrcnn import get_model
from utils.dataset import CocoDiatomDataset
from utils.metrics_logger import init_metrics_csv, append_metrics_csv


NUM_CLASSES = 20



def collate_fn(batch):
    return tuple(zip(*batch))

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

    metrics = {
        "AP": float(coco_eval.stats[0]),
        "AP50": float(coco_eval.stats[1]),
        "AP75": float(coco_eval.stats[2]),
        "AP_small": float(coco_eval.stats[3]),
        "AP_medium": float(coco_eval.stats[4]),
        "AP_large": float(coco_eval.stats[5]),
    }

    return metrics

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

    metrics = {
        "AP": float(coco_eval.stats[0]),
        "AP50": float(coco_eval.stats[1]),
        "AP75": float(coco_eval.stats[2]),
        "AP_small": float(coco_eval.stats[3]),
        "AP_medium": float(coco_eval.stats[4]),
        "AP_large": float(coco_eval.stats[5]),
    }

    return metrics

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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Dataset
    train_dataset = CocoDiatomDataset(
        args.image_dir,
        args.ann_file,
    )

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

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn,
        )
    
    # Model
    model = get_model(NUM_CLASSES)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0005,
    )
    
    
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

        save_path = os.path.join(
            args.output_dir,
            f"maskrcnn_epoch_{epoch + 1}.pth"
        )

        torch.save(
            model.state_dict(),
            save_path,
        )
        print(f"Saved: {save_path}")

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


if __name__ == "__main__":
    main()