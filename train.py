import os
import argparse


import torch
from torch.utils.data import DataLoader

from models.maskrcnn import get_model
from utils.dataset import CocoDiatomDataset


NUM_CLASSES = 20

OUTPUT_DIR = "runs"


def collate_fn(batch):
    return tuple(zip(*batch))

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
    
    
    return parser.parse_args()


def main():
    args = get_args()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
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
        print(f"Epoch [{epoch + 1}/{args.epochs}], loss: {avg_loss:.4f}")

        save_path = os.path.join(
            OUTPUT_DIR,
            f"maskrcnn_epoch_{epoch + 1}.pth"
        )

        torch.save(
            model.state_dict(),
            save_path
        )
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()