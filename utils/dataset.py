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


class ClassUniformCopyPasteAugmentation:
    """Paste uniformly sampled annotated instances, then apply basic augmentation.

    Donors are selected in two stages: first choose a foreground category
    uniformly, then choose one annotation from that category uniformly. This
    increases exposure to rare categories without repeating an entire donor
    image. Only annotations from the training COCO object are used.
    """

    def __init__(
        self,
        image_dir,
        coco,
        probability=0.30,
        max_instances=1,
        max_overlap=0.10,
        placement_attempts=20,
        brightness=0.10,
        contrast=0.10,
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Copy-Paste probability must be between 0 and 1")
        if max_instances < 1:
            raise ValueError("Copy-Paste max_instances must be at least 1")
        if not 0.0 <= max_overlap <= 1.0:
            raise ValueError("Copy-Paste max_overlap must be between 0 and 1")
        if placement_attempts < 1:
            raise ValueError("Copy-Paste placement_attempts must be at least 1")

        self.image_dir = image_dir
        self.coco = coco
        self.probability = probability
        self.max_instances = max_instances
        self.max_overlap = max_overlap
        self.placement_attempts = placement_attempts
        self.basic_augmentation = BasicDiatomAugmentation(
            brightness=brightness,
            contrast=contrast,
        )
        self.annotations_by_category = {
            int(category_id): [
                annotation
                for annotation in self.coco.loadAnns(
                    self.coco.getAnnIds(catIds=[category_id], iscrowd=False)
                )
                if annotation.get("bbox", [0, 0, 0, 0])[2] > 0
                and annotation.get("bbox", [0, 0, 0, 0])[3] > 0
            ]
            for category_id in sorted(self.coco.cats)
        }
        self.available_categories = [
            category_id
            for category_id, annotations in self.annotations_by_category.items()
            if annotations
        ]
        if not self.available_categories:
            raise ValueError("No valid donor annotations are available for Copy-Paste")

    def _sample_donor_annotation(self, recipient_image_id):
        category_index = int(
            torch.randint(0, len(self.available_categories), ()).item()
        )
        category_id = self.available_categories[category_index]
        annotations = self.annotations_by_category[category_id]
        other_image_annotations = [
            annotation
            for annotation in annotations
            if int(annotation["image_id"]) != recipient_image_id
        ]
        candidates = other_image_annotations or annotations
        annotation_index = int(torch.randint(0, len(candidates), ()).item())
        return candidates[annotation_index]

    def _load_donor_crop(self, annotation):
        image_info = self.coco.loadImgs(int(annotation["image_id"]))[0]
        image_path = os.path.join(self.image_dir, image_info["file_name"])
        donor_image = F.to_tensor(Image.open(image_path).convert("RGB"))
        donor_mask = torch.as_tensor(
            self.coco.annToMask(annotation),
            dtype=torch.uint8,
        )

        y_coordinates, x_coordinates = torch.where(donor_mask > 0)
        if x_coordinates.numel() == 0:
            return None
        x_min = int(x_coordinates.min().item())
        x_max = int(x_coordinates.max().item()) + 1
        y_min = int(y_coordinates.min().item())
        y_max = int(y_coordinates.max().item()) + 1
        donor_image = donor_image[:, y_min:y_max, x_min:x_max]
        donor_mask = donor_mask[y_min:y_max, x_min:x_max]

        if torch.rand(()) < 0.5:
            donor_image = F.hflip(donor_image)
            donor_mask = F.hflip(donor_mask)
        if torch.rand(()) < 0.5:
            donor_image = F.vflip(donor_image)
            donor_mask = F.vflip(donor_mask)
        quarter_turns = int(torch.randint(0, 4, ()).item())
        if quarter_turns:
            donor_image = torch.rot90(donor_image, quarter_turns, dims=(-2, -1))
            donor_mask = torch.rot90(donor_mask, quarter_turns, dims=(-2, -1))

        return donor_image, donor_mask

    def _paste_one(self, image, target):
        recipient_image_id = int(target["image_id"].reshape(-1)[0].item())
        annotation = self._sample_donor_annotation(recipient_image_id)
        donor = self._load_donor_crop(annotation)
        if donor is None:
            return image, target
        donor_image, donor_mask = donor

        image_height, image_width = image.shape[-2:]
        donor_height, donor_width = donor_mask.shape[-2:]
        if donor_height > image_height or donor_width > image_width:
            return image, target

        existing_union = torch.any(target["masks"] > 0, dim=0) if len(
            target["masks"]
        ) else torch.zeros(
            (image_height, image_width),
            dtype=torch.bool,
        )
        donor_area = int((donor_mask > 0).sum().item())
        if donor_area == 0:
            return image, target

        placement = None
        for _ in range(self.placement_attempts):
            top = int(
                torch.randint(0, image_height - donor_height + 1, ()).item()
            )
            left = int(
                torch.randint(0, image_width - donor_width + 1, ()).item()
            )
            overlap_pixels = torch.logical_and(
                existing_union[top : top + donor_height, left : left + donor_width],
                donor_mask > 0,
            ).sum()
            overlap_fraction = float(overlap_pixels.item()) / donor_area
            if overlap_fraction <= self.max_overlap:
                placement = (top, left)
                break
        if placement is None:
            return image, target

        top, left = placement
        donor_mask_bool = donor_mask > 0
        image = image.clone()
        target = dict(target)
        masks = target["masks"].clone()

        image_region = image[:, top : top + donor_height, left : left + donor_width]
        image[:, top : top + donor_height, left : left + donor_width] = torch.where(
            donor_mask_bool.unsqueeze(0),
            donor_image,
            image_region,
        )

        if len(masks):
            masks[:, top : top + donor_height, left : left + donor_width] *= (
                ~donor_mask_bool
            ).to(dtype=masks.dtype)
        pasted_mask = torch.zeros(
            (image_height, image_width),
            dtype=masks.dtype,
        )
        pasted_mask[top : top + donor_height, left : left + donor_width] = (
            donor_mask_bool.to(dtype=masks.dtype)
        )
        target["masks"] = torch.cat([masks, pasted_mask.unsqueeze(0)], dim=0)
        target["labels"] = torch.cat(
            [
                target["labels"],
                torch.tensor(
                    [int(annotation["category_id"])],
                    dtype=target["labels"].dtype,
                ),
            ]
        )
        target["iscrowd"] = torch.cat(
            [
                target["iscrowd"],
                torch.zeros((1,), dtype=target["iscrowd"].dtype),
            ]
        )
        target = _rebuild_boxes_and_areas(target)
        return image, target

    def __call__(self, image, target):
        if torch.rand(()) < self.probability:
            instance_count = int(
                torch.randint(1, self.max_instances + 1, ()).item()
            )
            for _ in range(instance_count):
                image, target = self._paste_one(image, target)
        return self.basic_augmentation(image, target)


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
