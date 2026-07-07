"""
dataset.py

Dataset utilities for training Mask R-CNN.

This module provides a custom PyTorch Dataset for:
- loading microscopy images
- loading LabelMe annotations
- converting annotations into Mask R-CNN targets
- preparing training and validation samples
"""

import os

import numpy as np
import torch
import torchvision.transforms.functional as F

from PIL import Image
from torch.utils.data import Dataset

from pycocotools.coco import COCO

class CocoDiatomDataset(Dataset):
    def __init__(self, image_dir, ann_file):
        self.image_dir = image_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        
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
            
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(np.array(masks), dtype=torch.uint8),
            "image_id": torch.tensor([image_id]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64)
        }
        
        return image, target