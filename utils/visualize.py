"""
visualize.py

Visualization utilities for Mask R-CNN predictions.

This module provides helper functions to:
- draw bounding boxes
- overlay segmentation masks
- display class labels and confidence scores
- save visualization images
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import to_rgb


# Match the label colors used in the current CVAT project.
CVAT_CLASS_COLORS = {
    "A.subarctica": "#FBAAD1",
    "cyclostephanoids": "#B1A9F2",
    "circle": "#A3B088",
    "A.ambigua": "#7CE9FE",
    "A.glanulate": "#AFE9A6",
    "buble": "#6676BD",
    "band": "#FF9D91",
    "Bacillariophyceae": "#C7F5E0",
    "Fragilariophycea": "#FDFA8F",
    "debri": "#8A8BE0",
    "plant fragment": "#DDBB86",
    "S.suzuki": "#CD89F5",
}

DEFAULT_CLASS_NAMES = {
    class_id: class_name
    for class_id, class_name in enumerate(CVAT_CLASS_COLORS, start=1)
}


def _class_name(label, class_names):
    label = int(label)
    if class_names:
        return str(class_names.get(label, label))
    return DEFAULT_CLASS_NAMES.get(label, str(label))


def _text_color(background_color):
    red, green, blue = to_rgb(background_color)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance > 0.62 else "white"


def save_visualization(
    image,
    boxes,
    labels,
    scores,
    masks,
    output_path,
    show_masks=True,
    class_names=None,
    mask_thresh=0.5,
    mask_alpha=0.35,
):
    width, height = image.size
    fig, ax = plt.subplots(figsize=(8, 8 * height / width))
    ax.imshow(image)

    if show_masks:
        for mask, label in zip(masks, labels):
            binary_mask = mask.squeeze(0).numpy() >= mask_thresh
            if not np.any(binary_mask):
                continue

            class_name = _class_name(label.item(), class_names)
            color = CVAT_CLASS_COLORS.get(class_name, "#FFFFFF")
            overlay = np.zeros((*binary_mask.shape, 4), dtype=np.float32)
            overlay[binary_mask, :3] = to_rgb(color)
            overlay[binary_mask, 3] = mask_alpha
            ax.imshow(overlay)
            ax.contour(
                binary_mask.astype(np.uint8),
                levels=[0.5],
                colors=[color],
                linewidths=0.8,
            )

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.numpy()
        class_name = _class_name(label.item(), class_names)
        color = CVAT_CLASS_COLORS.get(class_name, "#FFFFFF")

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        ax.text(
            x1,
            y1,
            f"{class_name} {score.item():.2f}",
            color=_text_color(color),
            fontsize=8,
            va="bottom",
            bbox={
                "facecolor": color,
                "edgecolor": color,
                "alpha": 0.9,
                "pad": 1.5,
            },
        )

    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=200)
    plt.close()
    
    print(f"Saved visualization: {output_path}")
    
