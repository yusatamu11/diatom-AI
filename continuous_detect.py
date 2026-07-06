import argparse
import os

import torch
import torchvision.transforms.functional as F
from PIL import Image

from models.maskrcnn import get_model
from utils.visualize import save_visualization

from pathlib import Path #detect.pyと違う．一気に画像を取得可能

NUM_CLASSES = 20

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
        "--save_image",
        action="store_true",
        help="Save visualization images",
    )

    parser.add_argument(
        "--show_masks",
        action="store_true",
        help="Overlay masks on visualization",
    )
    
    return parser.parse_args()

def main():
    args = get_args()
    
        
    os.makedirs(args.output_dir, exist_ok=True)# exists_ok=Trueで既に存在していてもエラーにならない
    
    image_dir = Path(args.image_dir)

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]:
        image_paths.extend(image_dir.glob(ext))#append だとリストそのものを追加してしまうが，extendだとリストを展開して１つずつ追加できる

    image_paths = sorted(image_paths)
    print(f"Found {len(image_paths)} images.")
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = get_model(NUM_CLASSES)
    model.load_state_dict(
        torch.load(args.weights, map_location=device)
    )
    model.to(device)
    model.eval()
    
    
    for i, image_path in enumerate(image_paths, start=1):
        print(f"[{i}/{len(image_paths)}] Processing: {image_path.name}")
    
        image = Image.open(image_path).convert("RGB")
        image_tensor = F.to_tensor(image).to(device)
        
        # 推論
        with torch.no_grad():
            outputs = model([image_tensor])

        output = outputs[0]

        scores = output["scores"]
        keep = scores >= args.score_thresh

        boxes = output["boxes"][keep]
        labels = output["labels"][keep]
        masks = output["masks"][keep]
        scores = scores[keep]
        
        stem = image_path.stem#stemはPathオブジェクトが持っている拡張子を除いたファイル名を取得する

        output_path = Path(args.output_dir) / f"{stem}.pt"
        
        result = {
            "boxes": boxes.detach().cpu(),
            "labels": labels.detach().cpu(),
            "scores": scores.detach().cpu(),
            "masks": masks.detach().cpu(),
            "image_path": str(image_path),
        }
        
        torch.save(result, output_path)
        print(f"Saved: {output_path}")
        
        if args.save_image:
            save_visualization(
                image=image,
                boxes=boxes.detach().cpu(),
                labels=labels.detach().cpu(),
                scores=scores.detach().cpu(),
                masks=masks.detach().cpu(),
                output_path=output_path.with_suffix(".jpg"),#with_suffixで拡張子を変更できる
                show_masks=args.show_masks,
            )
            
if __name__ == "__main__":
    main()