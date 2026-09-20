"""Losslessly compact thresholded tile masks for prediction storage."""

import torch


MASK_FORMAT = "cropped_uint8_v1"


def encode_cropped_binary_masks(masks, threshold=0.5):
    """Threshold full-image masks and crop each one to its nonzero extent.

    Returns a list of 2-D uint8 tensors and an ``[N, 2]`` int32 tensor whose
    columns are the crop's x and y origins in tile coordinates.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("mask threshold must be between 0 and 1")
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(
            "Expected masks with shape [N,H,W] or [N,1,H,W], got "
            f"{tuple(masks.shape)}"
        )

    binary_masks = (masks >= threshold).to(dtype=torch.uint8, device="cpu")
    crops = []
    origins = []
    for mask in binary_masks:
        nonzero = torch.nonzero(mask, as_tuple=False)
        if nonzero.numel() == 0:
            crops.append(torch.zeros((0, 0), dtype=torch.uint8))
            origins.append((0, 0))
            continue
        y_min = int(nonzero[:, 0].min().item())
        y_max = int(nonzero[:, 0].max().item()) + 1
        x_min = int(nonzero[:, 1].min().item())
        x_max = int(nonzero[:, 1].max().item()) + 1
        crops.append(mask[y_min:y_max, x_min:x_max].contiguous())
        origins.append((x_min, y_min))

    origin_tensor = torch.tensor(origins, dtype=torch.int32)
    if not origins:
        origin_tensor = torch.empty((0, 2), dtype=torch.int32)
    return crops, origin_tensor


def restore_full_binary_masks(crops, origins_xy, canvas_size_hw):
    """Restore cropped masks to ``[N,1,H,W]`` for optional visualization."""
    height, width = (int(value) for value in canvas_size_hw)
    restored = torch.zeros(
        (len(crops), 1, height, width), dtype=torch.uint8
    )
    for index, (crop, origin) in enumerate(zip(crops, origins_xy)):
        x_min, y_min = (int(value) for value in origin)
        crop_height, crop_width = crop.shape
        restored[
            index, 0,
            y_min:y_min + crop_height,
            x_min:x_min + crop_width,
        ] = crop
    return restored
