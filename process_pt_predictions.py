"""Merge tile-level ``.pt`` predictions and measure mask morphology.

The input files are prediction dictionaries written by ``continuous_detect.py``.
Model checkpoint files (``.pth``) are not accepted by this script.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from skimage.measure import regionprops

from process_json_predictions import (
    AXIS_LENGTH_CLASSES,
    MORPHOLOGY_COLUMNS,
    MORPHOLOGY_KEYS,
    parse_tile_xy,
    remove_adjacent_duplicates,
)


def get_args():
    """Define command-line arguments for one-sample ``.pt`` processing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction_dir",
        required=True,
        help="Directory containing one sample's tile prediction .pt files",
    )
    parser.add_argument(
        "--csv_output",
        default="morphology.csv",
        help="Output path for the instance-level morphology CSV",
    )
    parser.add_argument(
        "--merged_output",
        default=None,
        help="Optional path for duplicate-removed merged predictions (.pt)",
    )
    parser.add_argument(
        "--class_map",
        default=None,
        help=(
            "Optional JSON class map or COCO annotation JSON. Without it, "
            "names such as class_3 are used."
        ),
    )
    parser.add_argument("--tile_size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=180)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--mask_thresh", type=float, default=0.5)
    return parser.parse_args()


def load_class_map(path):
    """Load class ID-to-name mappings from a simple or COCO-style JSON file."""
    if path is None:
        return {}
    with Path(path).open(encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return {index: str(name) for index, name in enumerate(data)}
    if not isinstance(data, dict):
        raise ValueError("Class map JSON must contain a list or object.")
    if "categories" in data:
        return {
            int(category["id"]): str(category["name"])
            for category in data["categories"]
        }
    if "class_names" in data:
        names = data["class_names"]
        if isinstance(names, list):
            return {index: str(name) for index, name in enumerate(names)}
        if isinstance(names, dict):
            return {int(class_id): str(name) for class_id, name in names.items()}
        raise ValueError("class_names must be a list or object.")
    return {int(class_id): str(name) for class_id, name in data.items()}


def _torch_load_prediction(path):
    """Load tensor-only prediction data, retaining compatibility with old PyTorch."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # ``weights_only`` is unavailable in older PyTorch releases. Only load
        # prediction files created by this project when this fallback is used.
        return torch.load(path, map_location="cpu")


def _embedded_class_map(prediction):
    """Read an optional class-name mapping stored inside a prediction file."""
    names = prediction.get("class_names")
    if isinstance(names, (list, tuple)):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(class_id): str(name) for class_id, name in names.items()}
    return {}


def _validate_prediction(prediction, path):
    """Validate the fields and leading dimensions of one prediction dictionary."""
    if not isinstance(prediction, dict):
        raise TypeError(f"Prediction must be a dictionary: {path}")
    required = ("boxes", "labels", "scores", "masks")
    missing = [key for key in required if key not in prediction]
    if missing:
        raise KeyError(f"Missing {', '.join(missing)} in prediction: {path}")
    lengths = {key: len(prediction[key]) for key in required}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Prediction fields have different lengths in {path}: {lengths}")


def load_pt_tiles(prediction_dir, tile_size, overlap, mask_thresh, class_map=None):
    """Load tile predictions and convert their boxes to global slide coordinates."""
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile_size")

    prediction_paths = sorted(Path(prediction_dir).glob("*.pt"))
    if not prediction_paths:
        raise FileNotFoundError(f"No tile .pt files found in: {prediction_dir}")

    explicit_class_map = dict(class_map or {})
    tiles = {}
    missing_class_ids = set()
    for prediction_path in prediction_paths:
        tile_x, tile_y = parse_tile_xy(prediction_path)
        if (tile_x, tile_y) in tiles:
            raise ValueError(f"Duplicate tile coordinates in: {prediction_path}")

        prediction = _torch_load_prediction(prediction_path)
        _validate_prediction(prediction, prediction_path)
        file_class_map = _embedded_class_map(prediction)

        boxes = prediction["boxes"].detach().cpu().to(torch.float32)
        labels = prediction["labels"].detach().cpu().to(torch.int64)
        scores = prediction["scores"].detach().cpu().to(torch.float32)
        masks = prediction["masks"].detach().cpu()
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if masks.ndim != 3:
            raise ValueError(
                f"Expected masks with shape [N,H,W] or [N,1,H,W], got "
                f"{tuple(masks.shape)} in {prediction_path}"
            )
        masks = masks >= mask_thresh

        origin_x = (tile_x - 1) * stride
        origin_y = (tile_y - 1) * stride
        instances = []
        for index in range(len(boxes)):
            class_id = int(labels[index].item())
            class_name = explicit_class_map.get(
                class_id, file_class_map.get(class_id, f"class_{class_id}")
            )
            if class_name == f"class_{class_id}":
                missing_class_ids.add(class_id)
            local_box = [float(value) for value in boxes[index].tolist()]
            global_box = [
                local_box[0] + origin_x,
                local_box[1] + origin_y,
                local_box[2] + origin_x,
                local_box[3] + origin_y,
            ]
            instances.append({
                "class_id": class_id,
                "class_name": class_name,
                "score": float(scores[index].item()),
                "local_bbox_xyxy": local_box,
                "global_bbox_xyxy": global_box,
                "local_mask": masks[index],
                "source_image": str(
                    prediction.get("image_path", prediction_path.name)
                ),
                "source_tile_x": tile_x,
                "source_tile_y": tile_y,
                "tile_origin_x": origin_x,
                "tile_origin_y": origin_y,
            })
        tiles[(tile_x, tile_y)] = instances

    if missing_class_ids:
        ids = ", ".join(str(class_id) for class_id in sorted(missing_class_ids))
        print(
            "Warning: no class names were provided for IDs "
            f"{ids}; fallback names will be used."
        )
    return tiles


def _empty_morphology():
    """Return missing morphology values for an empty predicted mask."""
    return {key: math.nan for key in MORPHOLOGY_KEYS}


def mask_morphology(mask, origin_x=0, origin_y=0):
    """Measure a binary instance mask and return global centroid coordinates."""
    mask_array = mask.detach().cpu().numpy().astype(np.uint8, copy=False)
    if mask_array.ndim != 2 or not np.any(mask_array):
        return _empty_morphology()
    regions = regionprops(mask_array)
    if not regions:
        return _empty_morphology()

    region = regions[0]
    area = float(region.area)
    perimeter = float(region.perimeter)
    perimeter_crofton = float(region.perimeter_crofton)
    major_axis = float(region.axis_major_length)
    minor_axis = float(region.axis_minor_length)
    return {
        "area": area,
        "perimeter": perimeter,
        "perimeter_crofton": perimeter_crofton,
        "centroid_x": float(region.centroid[1] + origin_x),
        "centroid_y": float(region.centroid[0] + origin_y),
        "major_axis": major_axis,
        "minor_axis": minor_axis,
        "aspect_ratio": major_axis / minor_axis if minor_axis > 0 else math.nan,
        "equivalent_diameter": float(region.equivalent_diameter_area),
        "eccentricity": float(region.eccentricity),
        "circularity": 4 * math.pi * area / perimeter**2 if perimeter > 0 else math.nan,
        "circularity_crofton": (
            4 * math.pi * area / perimeter_crofton**2
            if perimeter_crofton > 0 else math.nan
        ),
        "orientation_deg": float(np.degrees(region.orientation)),
        "extent": float(region.extent),
        "solidity": float(region.solidity),
    }


def build_morphology_rows(instances):
    """Convert duplicate-removed instances into the shared morphology CSV schema."""
    rows = []
    for instance_id, instance in enumerate(instances, start=1):
        box = instance["global_bbox_xyxy"]
        width = max(0.0, box[2] - box[0])
        height = max(0.0, box[3] - box[1])
        morphology = mask_morphology(
            instance["local_mask"],
            origin_x=instance["tile_origin_x"],
            origin_y=instance["tile_origin_y"],
        )
        morphology_valid = not math.isnan(morphology["area"])
        axis_length_calculated = (
            morphology_valid and instance["class_name"] in AXIS_LENGTH_CLASSES
        )
        rows.append({
            "instance_id": instance_id,
            "class_id": instance["class_id"],
            "class_name": instance["class_name"],
            "score": instance["score"],
            "source_image": instance["source_image"],
            "source_tile_x": instance["source_tile_x"],
            "source_tile_y": instance["source_tile_y"],
            "bbox_x1_px": box[0], "bbox_y1_px": box[1],
            "bbox_x2_px": box[2], "bbox_y2_px": box[3],
            "bbox_width_px": width, "bbox_height_px": height,
            "bbox_area_px2": width * height,
            "morphology_valid": morphology_valid,
            "mask_area_px2": morphology["area"],
            "perimeter_px": morphology["perimeter"],
            "perimeter_crofton_px": morphology["perimeter_crofton"],
            "centroid_x_px": morphology["centroid_x"],
            "centroid_y_px": morphology["centroid_y"],
            "axis_length_calculated": axis_length_calculated,
            "major_axis_length_px": (
                morphology["major_axis"] if axis_length_calculated else ""
            ),
            "minor_axis_length_px": (
                morphology["minor_axis"] if axis_length_calculated else ""
            ),
            "aspect_ratio": (
                morphology["aspect_ratio"] if axis_length_calculated else ""
            ),
            "equivalent_diameter_px": morphology["equivalent_diameter"],
            "eccentricity": morphology["eccentricity"],
            "circularity": morphology["circularity"],
            "circularity_crofton": morphology["circularity_crofton"],
            "orientation_deg": morphology["orientation_deg"],
            "extent": morphology["extent"],
            "solidity": morphology["solidity"],
        })
    return rows


def save_morphology_csv(rows, output_path):
    """Save instance-level measurements using the JSON pipeline's CSV columns."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MORPHOLOGY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def save_merged_pt(instances, output_path, metadata):
    """Save duplicate-removed predictions without padding their local masks."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_instances = []
    for instance_id, instance in enumerate(instances, start=1):
        serializable_instances.append({
            "instance_id": instance_id,
            "class_id": instance["class_id"],
            "class_name": instance["class_name"],
            "score": instance["score"],
            "bbox_xyxy": torch.tensor(
                instance["global_bbox_xyxy"], dtype=torch.float32
            ),
            "mask": instance["local_mask"].to(torch.bool),
            "source_image": instance["source_image"],
            "source_tile_x": instance["source_tile_x"],
            "source_tile_y": instance["source_tile_y"],
            "tile_origin_x": instance["tile_origin_x"],
            "tile_origin_y": instance["tile_origin_y"],
        })
    torch.save({**metadata, "instances": serializable_instances}, output_path)


def process_pt_directory(
    prediction_dir,
    csv_output,
    merged_output=None,
    class_map=None,
    tile_size=1200,
    overlap=180,
    iou_thresh=0.5,
    mask_thresh=0.5,
):
    """Load, deduplicate, measure and save one sample's tile predictions."""
    if isinstance(class_map, (str, Path)):
        class_map = load_class_map(class_map)
    tiles = load_pt_tiles(
        prediction_dir, tile_size, overlap, mask_thresh, class_map=class_map
    )
    before_count = sum(len(instances) for instances in tiles.values())
    instances, removed_count = remove_adjacent_duplicates(
        tiles, tile_size, overlap, iou_thresh
    )
    rows = build_morphology_rows(instances)
    save_morphology_csv(rows, csv_output)

    metadata = {
        "format": "diatom-ai-merged-mask-v1",
        "coordinate_system": "global_slide_pixels",
        "tile_size": tile_size,
        "overlap": overlap,
        "iou_thresh": iou_thresh,
        "mask_thresh": mask_thresh,
        "num_tiles": len(tiles),
        "num_instances": len(instances),
        "num_removed_duplicates": removed_count,
    }
    if merged_output is not None:
        save_merged_pt(instances, merged_output, metadata)

    print(f"Loaded tiles: {len(tiles)}")
    print(f"Instances before duplicate removal: {before_count}")
    print(f"Removed duplicate detections: {removed_count}")
    print(f"Remaining instances: {len(instances)}")
    print(f"Morphology CSV saved to: {csv_output}")
    if merged_output is not None:
        print(f"Merged predictions saved to: {merged_output}")
    return rows, metadata


def main():
    """Process command-line arguments for one sample."""
    args = get_args()
    process_pt_directory(
        prediction_dir=args.prediction_dir,
        csv_output=args.csv_output,
        merged_output=args.merged_output,
        class_map=args.class_map,
        tile_size=args.tile_size,
        overlap=args.overlap,
        iou_thresh=args.iou_thresh,
        mask_thresh=args.mask_thresh,
    )


if __name__ == "__main__":
    main()
