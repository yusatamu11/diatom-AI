"""
dataset.py

Dataset utilities for training Mask R-CNN.

This module provides a custom PyTorch Dataset for:
- loading microscopy images and COCO annotations
- converting annotations into Mask R-CNN targets
- applying synchronized image/mask augmentations
- preparing class-balanced sampling weights
"""

import os

import numpy as np
import torch
import torchvision.transforms.functional as F

from PIL import Image
from torch.utils.data import Dataset

from pycocotools.coco import COCO


class BasicDiatomAugmentation:
    """Apply conservative microscopy augmentation to an image and its masks."""

    def __init__(self, brightness=0.10, contrast=0.10):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, image, target):
        masks = target["masks"]

        if torch.rand(()) < 0.5:
            image = F.hflip(image)
            masks = F.hflip(masks)
        if torch.rand(()) < 0.5:
            image = F.vflip(image)
            masks = F.vflip(masks)

        quarter_turns = int(torch.randint(0, 4, ()).item())
        if quarter_turns:
            image = torch.rot90(image, quarter_turns, dims=(-2, -1))
            masks = torch.rot90(masks, quarter_turns, dims=(-2, -1))

        brightness_factor = 1.0 + (
            float(torch.rand(()).item()) * 2.0 - 1.0
        ) * self.brightness
        contrast_factor = 1.0 + (
            float(torch.rand(()).item()) * 2.0 - 1.0
        ) * self.contrast
        image = F.adjust_brightness(image, brightness_factor)
        image = F.adjust_contrast(image, contrast_factor)

        target = dict(target)
        target["masks"] = masks
        target = _rebuild_boxes_and_areas(target)
        return image, target


class StrongDiatomAugmentation(BasicDiatomAugmentation):
    """Apply stronger, microscopy-safe photometric augmentation."""

    def __init__(
        self,
        brightness=0.20,
        contrast=0.20,
        gamma_range=(0.85, 1.15),
        blur_probability=0.20,
        noise_probability=0.20,
        noise_std=0.02,
    ):
        super().__init__(brightness=brightness, contrast=contrast)
        self.gamma_range = gamma_range
        self.blur_probability = blur_probability
        self.noise_probability = noise_probability
        self.noise_std = noise_std

    def __call__(self, image, target):
        image, target = super().__call__(image, target)

        gamma_min, gamma_max = self.gamma_range
        gamma = gamma_min + float(torch.rand(()).item()) * (
            gamma_max - gamma_min
        )
        image = F.adjust_gamma(image, gamma)

        if torch.rand(()) < self.blur_probability:
            sigma = 0.3 + float(torch.rand(()).item()) * 0.7
            image = F.gaussian_blur(
                image,
                kernel_size=[5, 5],
                sigma=[sigma, sigma],
            )

        if torch.rand(()) < self.noise_probability:
            image = torch.clamp(
                image + torch.randn_like(image) * self.noise_std,
                min=0.0,
                max=1.0,
            )

        return image, target


def _rebuild_boxes_and_areas(target):
    """Rebuild boxes from transformed masks and discard any empty masks."""
    masks = target["masks"]
    valid_indices = []
    boxes = []
    areas = []

    for index, mask in enumerate(masks):
        y_coordinates, x_coordinates = torch.where(mask > 0)
        if x_coordinates.numel() == 0:
            continue
        valid_indices.append(index)
        boxes.append(
            torch.stack(
                [
                    x_coordinates.min(),
                    y_coordinates.min(),
                    x_coordinates.max() + 1,
                    y_coordinates.max() + 1,
                ]
            ).to(dtype=torch.float32)
        )
        areas.append(mask.sum().to(dtype=torch.float32))

    if valid_indices:
        indices = torch.as_tensor(valid_indices, dtype=torch.long)
        target["masks"] = masks[indices]
        target["labels"] = target["labels"][indices]
        target["iscrowd"] = target["iscrowd"][indices]
        target["boxes"] = torch.stack(boxes)
        target["area"] = torch.stack(areas)
    else:
        height, width = masks.shape[-2:]
        target["masks"] = masks.new_zeros((0, height, width))
        target["labels"] = target["labels"].new_zeros((0,))
        target["iscrowd"] = target["iscrowd"].new_zeros((0,))
        target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        target["area"] = torch.zeros((0,), dtype=torch.float32)

    return target


def make_class_balanced_sample_weights(dataset, max_weight=5.0):
    """Return capped inverse-square-root weights for images with rare classes."""
    if max_weight < 1.0:
        raise ValueError("max_weight must be 1.0 or greater")

    annotation_counts = {
        category_id: len(dataset.coco.getAnnIds(catIds=[category_id]))
        for category_id in dataset.category_names
    }
    positive_counts = [count for count in annotation_counts.values() if count > 0]
    if not positive_counts:
        category_weights = {category_id: 1.0 for category_id in annotation_counts}
        return (
            torch.ones(len(dataset), dtype=torch.double),
            annotation_counts,
            category_weights,
        )

    reference_count = float(np.percentile(positive_counts, 75))
    category_weights = {
        category_id: min(
            max_weight,
            max(1.0, (reference_count / count) ** 0.5),
        )
        if count > 0
        else 1.0
        for category_id, count in annotation_counts.items()
    }

    sample_weights = []
    for image_id in dataset.ids:
        annotations = dataset.coco.imgToAnns.get(image_id, [])
        present_categories = {int(ann["category_id"]) for ann in annotations}
        sample_weights.append(
            max(
                (category_weights[category_id] for category_id in present_categories),
                default=1.0,
            )
        )

    return (
        torch.as_tensor(sample_weights, dtype=torch.double),
        annotation_counts,
        category_weights,
    )


class CocoDiatomDataset(Dataset):
    def __init__(self, image_dir, ann_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.category_names = {
            int(category_id): str(category["name"])
            for category_id, category in self.coco.cats.items()
        }
        category_ids = sorted(self.category_names)
        expected_ids = list(range(1, len(category_ids) + 1))
        if category_ids != expected_ids:
            raise ValueError(
                "Torchvision Mask R-CNN requires contiguous foreground "
                "category IDs starting at 1. "
                f"Expected {expected_ids}, got {category_ids}."
            )
        self.num_classes = len(category_ids) + 1  # foreground classes + background
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        image_id = self.ids[index]

        img_info = self.coco.loadImgs(image_id)[0]
        
        image_path = os.path.join(
                self.image_dir,
                img_info["file_name"]
            )

        image = Image.open(image_path).convert("RGB")
        image = F.to_tensor(image)
        
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            x, y, w, h = ann["bbox"]
            
            if w <=0 or h <= 0:
                continue
            
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            masks.append(self.coco.annToMask(ann))
            areas.append(ann["area"])
            iscrowd.append(ann.get("iscrowd", 0))
            
        if len(boxes) == 0:
            _, height, width = image.shape

            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, height, width), dtype=torch.uint8)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([image_id]),
            "area": areas,
            "iscrowd": iscrowd,
        }

        if self.transform is not None:
            image, target = self.transform(image, target)
        
        return image, target
