import matplotlib.pyplot as plt
import matplotlib.patches as patches


def save_visualization(
    image,
    boxes,
    labels,
    scores,
    masks,
    output_path,
    show_masks=True,
):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)

    if show_masks:
        for mask in masks:
            mask = mask.squeeze(0).numpy()
            ax.imshow(mask > 0.5, alpha=0.3)

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.numpy()

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)

        ax.text(
            x1,
            y1,
            f"{label.item()}:{score.item():.2f}",
            color="yellow",
            fontsize=8,
        )

    ax.axis("off")
    plt.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close()
    
    print(f"Saved visualization: {output_path}")
    

