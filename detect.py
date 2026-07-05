import argparse
import os

import torch
import torchvision.transforms.functional as F
from PIL import Image

from models.maskrcnn import get_model
from utils.visualize import save_visualization


NUM_CLASSES = 20


def get_args():
    parser = argparse.ArgumentParser(
        description="Run inference with trained Mask R-CNN."
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained model weights",
    )

    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.5,
        help="Score threshold for detections",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="inference/result.pt",
        help="Path to save prediction result",
    )

    parser.add_argument(
        "--save_image",
        action="store_true",
        help="Save visualization image",
    )#この引数を持ったら　args.save_image=True　となる
    
    parser.add_argument(
        "--show_masks",
        action="store_true",
        help="Overlay masks on visualization",
    )

    return parser.parse_args()


def main():
    args = get_args()
    
    output_dir = os.path.dirname(args.output)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Model
    model = get_model(NUM_CLASSES)
    model.load_state_dict(
        torch.load(args.weights, map_location=device)
    )
    model.to(device)
    model.eval()

    # Image
    image = Image.open(args.image).convert("RGB")
    image_tensor = F.to_tensor(image).to(device)

    # Inference
    with torch.no_grad():
        outputs = model([image_tensor])

    output = outputs[0]

    scores = output["scores"]
    keep = scores >= args.score_thresh

    boxes = output["boxes"][keep]
    labels = output["labels"][keep]
    masks = output["masks"][keep]
    scores = scores[keep]
    
    result = {
        "boxes": boxes.detach().cpu(),
        "labels": labels.detach().cpu(),
        "scores": scores.detach().cpu(),
        "masks": masks.detach().cpu(),
        "image_path": args.image,
    }

    torch.save(result, args.output)
    print(f"Saved prediction: {args.output}")
    
    if args.save_image:
        save_visualization(
            image=image,
            boxes=boxes.detach().cpu(),
            labels=labels.detach().cpu(),
            scores=scores.detach().cpu(),
            masks=masks.detach().cpu(),
            output_path=args.output.replace(".pt", ".jpg"),
            show_masks=args.show_masks,
        )

    print("Detections:", len(scores))
    print("Boxes:", boxes.shape)
    print("Labels:", labels)
    print("Scores:", scores)


if __name__ == "__main__":
    main()