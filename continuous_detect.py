"""
continuous_detect.py

Batch inference script for Mask R-CNN.

This script performs inference on all images in a directory,
optionally saves visualization images,
and archives prediction results.
"""

import argparse
import os

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from models.maskrcnn import get_model
from utils.checkpoint import load_training_checkpoint
from utils.visualize import save_visualization
from utils.archive import archive_directory
from utils.compact_masks import (
    MASK_FORMAT,
    encode_cropped_binary_masks,
    restore_full_binary_masks,
)

from pathlib import Path #detect.pyと違う．一気に画像を取得可能


class InferenceImageDataset(Dataset):
    """Load inference images in DataLoader workers."""

    def __init__(self, image_paths):
        self.image_paths = list(image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        with Image.open(image_path) as image_file:
            image_tensor = F.to_tensor(image_file.convert("RGB"))
        return image_path, image_tensor


def collate_image_batch(batch):
    """Keep differently sized detection images as lists rather than stacking."""
    image_paths, image_tensors = zip(*batch)
    return list(image_paths), list(image_tensors)

def get_args():
    parser = argparse.ArgumentParser(
        description="Run inference on all images in a directory"
    )
    
    parser.add_argument(
        "--image_dir", #detect.py と違う．
        type=str,
        required=True,
        help="Path to input image directory",
    )
    
    parser.add_argument(
    "--weights",
    type=str,
    required=True,
    help="Path to trained model weights",
)

    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference",
        help="Directory to save prediction results",
    )

    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.5,
        help="Score threshold for detections",
    )

    parser.add_argument(
        "--mask_thresh",
        type=float,
        default=0.5,
        help="Threshold used to store compact binary masks",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of images inferred together (default: 1)",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Parallel image-loading workers (default: 0)",
    )

    parser.add_argument(
        "--save_image",
        action="store_true",
        help="Save visualization images",
    )

    parser.add_argument(
        "--show_masks",
        action="store_true",
        help="Overlay masks on visualization",
    )
    
    parser.add_argument(
        "--archive",
        type=str,
        choices=["none", "zip", "tar", "tar.zst"],
        default="none",
        help="Archive prediction results",
    )
    
    return parser.parse_args()

def main():
    args = get_args()
    if args.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if args.num_workers < 0:
        raise ValueError("num_workers must be 0 or greater")
    
        
    os.makedirs(args.output_dir, exist_ok=True)# exists_ok=Trueで既に存在していてもエラーにならない
    
    image_dir = Path(args.image_dir)

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]:
        image_paths.extend(image_dir.glob(ext))#append だとリストそのものを追加してしまうが，extendだとリストを展開して１つずつ追加できる

    image_paths = sorted(image_paths)
    print(f"Found {len(image_paths)} images.")
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(f"Batch size: {args.batch_size}")
    print(f"Image loading workers: {args.num_workers}")

    data_loader = DataLoader(
        InferenceImageDataset(image_paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_image_batch,
    )

    state_dict, num_classes, class_names, checkpoint_metadata = (
        load_training_checkpoint(args.weights, map_location=device)
    )
    print(f"Model classes (including background): {num_classes}")
    if class_names:
        print(f"Foreground categories: {class_names}")
    else:
        print("Class names are unavailable in this legacy checkpoint.")
    model = get_model(num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    # Apply the requested threshold before the mask head so low-confidence
    # detections do not incur mask-generation and storage costs.
    model.roi_heads.score_thresh = args.score_thresh
    
    
    processed_count = 0
    for batch_paths, cpu_image_tensors in data_loader:
        image_tensors = [
            image_tensor.to(
                device,
                non_blocking=device.type == "cuda",
            )
            for image_tensor in cpu_image_tensors
        ]

        with torch.inference_mode():
            outputs = model(image_tensors)

        for batch_index, (
            image_path,
            cpu_image_tensor,
            output,
        ) in enumerate(zip(batch_paths, cpu_image_tensors, outputs), start=1):
            image_number = processed_count + batch_index
            print(
                f"[{image_number}/{len(image_paths)}] Processing: "
                f"{image_path.name}"
            )

            scores = output["scores"]
            keep = scores >= args.score_thresh

            boxes = output["boxes"][keep]
            labels = output["labels"][keep]
            masks = output["masks"][keep]
            scores = scores[keep]

            mask_crops, mask_origins_xy = encode_cropped_binary_masks(
                masks, threshold=args.mask_thresh
            )

            output_path = Path(args.output_dir) / f"{image_path.stem}.pt"
            result = {
                "boxes": boxes.detach().cpu(),
                "labels": labels.detach().cpu(),
                "scores": scores.detach().cpu(),
                "format": "diatom-ai-tile-prediction-v2",
                "mask_format": MASK_FORMAT,
                "mask_threshold": args.mask_thresh,
                "mask_crops": mask_crops,
                "mask_origins_xy": mask_origins_xy,
                "mask_canvas_size_hw": torch.tensor(
                    cpu_image_tensor.shape[-2:], dtype=torch.int32
                ),
                "image_path": str(image_path),
                "class_names": class_names,
                "checkpoint_format": checkpoint_metadata["format"],
                "checkpoint_epoch": checkpoint_metadata["epoch"],
            }

            torch.save(result, output_path)
            print(f"Saved: {output_path}")

            if args.save_image:
                save_visualization(
                    image=F.to_pil_image(cpu_image_tensor),
                    boxes=boxes.detach().cpu(),
                    labels=labels.detach().cpu(),
                    scores=scores.detach().cpu(),
                    masks=restore_full_binary_masks(
                        mask_crops,
                        mask_origins_xy,
                        cpu_image_tensor.shape[-2:],
                    ),
                    output_path=output_path.with_suffix(".jpg"),
                    show_masks=args.show_masks,
                    class_names=class_names,
                )
        processed_count += len(batch_paths)
            
    if args.archive != "none":
        archive_directory(
            args.output_dir,
            archive_type=args.archive,
        )
            
if __name__ == "__main__":
    main()
