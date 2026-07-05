import os

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
        
        return image