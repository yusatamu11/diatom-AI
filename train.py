import os

import torch
from torch.utils.data import DataLoader

from models.maskrcnn import get_model
from utils.dataset import CocoDiatomDataset


NUM_CLASSES = 20
TRAIN_IMAGE_DIR = "dataset/train/images"
TRAIN_ANN_FILE = "dataset/train/annotations.json"

NUM_EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 0.005
OUTPUT_DIR = "runs"


def collate_fn(batch):
    return tuple(zip(*batch))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Dataset
    train_dataset = CocoDiatomDataset(
        TRAIN_IMAGE_DIR,
        TRAIN_ANN_FILE,
    )

    # DataLoader for training
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
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
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=0.0005,
    )
    
    
    # Training
    for epoch in range(NUM_EPOCHS):
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
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], loss: {avg_loss:.4f}")

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