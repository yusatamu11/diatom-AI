"""
merge_predictions.py

Merge tile-level predictions into slide-level predictions.

This script:
- loads prediction (.pt) files
- converts tile coordinates to slide coordinates
- merges all detections into a single prediction file

Duplicate removal is performed in a later step.
"""

import argparse
import re
from pathlib import Path

from torchvision.ops import box_iou

import torch


def get_args():
    parser = argparse.ArgumentParser(
        description="Merge tile-level predictions into slide-level predictions."
    )

    parser.add_argument(
        "--prediction_dir",
        type=str,
        required=True,
        help="Directory containing tile prediction .pt files",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="merged_predictions.pt",
        help="Path to save merged prediction file",
    )

    parser.add_argument(
        "--tile_size",
        type=int,
        default=1200,
        help="Tile size in pixels",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=180,
        help="Tile overlap in pixels",
    )

    parser.add_argument(
        "--iou_thresh",
        type=float,
        default=0.5,
        help="IoU threshold for duplicate removal",
    )

    return parser.parse_args()

# =====================================
# Tile coordinate utilities
# =====================================

def parse_tile_xy(path):
    name = Path(path).stem

    match = re.search(r"x(\d+)_y(\d+)", name)

    if match is None:
        raise ValueError(f"Cannot parse tile coordinates from: {name}")

    tile_x = int(match.group(1))
    tile_y = int(match.group(2))

    return tile_x, tile_y #タイル座標を抽出

def shift_boxes_to_global(# グローバル座標に変換する関数
    boxes,
    tile_x,
    tile_y,
    tile_size=1200,
    overlap=180,
):
    stride = tile_size - overlap

    x_offset = (tile_x - 1) * stride
    y_offset = (tile_y - 1) * stride

    global_boxes = boxes.clone()

    global_boxes[:, [0, 2]] += x_offset
    global_boxes[:, [1, 3]] += y_offset

    return global_boxes

# =====================================
# Overlap utilities
# =====================================

#Bounding boxの中心座標を計算する関数
def get_box_centers(boxes):
    x_centers = (boxes[:, 0] + boxes[:, 2]) / 2
    y_centers = (boxes[:, 1] + boxes[:, 3]) / 2

    return x_centers, y_centers

# オーバーラップ領域に含まれるかどうかを判定する関数
def is_in_overlap_region(
    boxes,
    tile_size=1200,
    overlap=180,
):
    x_centers, y_centers = get_box_centers(boxes)

    in_left_overlap = x_centers < overlap
    in_right_overlap = x_centers > (tile_size - overlap)
    in_top_overlap = y_centers < overlap
    in_bottom_overlap = y_centers > (tile_size - overlap)

    in_overlap = (
        in_left_overlap
        | in_right_overlap
        | in_top_overlap
        | in_bottom_overlap
    )

    return in_overlap#中心がオーバーラップ領域に含まれる場合はTrue, それ以外はFalseを返す

# =====================================
# Adjacent-tile duplicate removal
# =====================================

def get_edge_indices(
    local_boxes,
    edge,
    tile_size=1200,
    overlap=180,
):
    x_centers, y_centers = get_box_centers(local_boxes)

    if edge == "left":
        return torch.where(x_centers < overlap)[0]
    if edge == "right":
        return torch.where(x_centers >= (tile_size - overlap))[0]
    if edge == "top":
        return torch.where(y_centers < overlap)[0]
    if edge == "bottom":
        return torch.where(y_centers >= (tile_size - overlap))[0]

    raise ValueError(f"Unsupported edge: {edge}")


#　隣接タイル間の二重に検出した珪藻の候補を探し出し，どちらを削除するか決める関数
def find_duplicate_indices_between_tiles(
    tile_a,
    tile_b,
    edge_a,
    edge_b,
    tile_size=1200,
    overlap=180,
    iou_thresh=0.5,
):
    local_boxes_a = tile_a["local_boxes"]
    global_boxes_a = tile_a["boxes"]
    labels_a = tile_a["labels"]
    scores_a = tile_a["scores"]

    local_boxes_b = tile_b["local_boxes"]
    global_boxes_b = tile_b["boxes"]
    labels_b = tile_b["labels"]
    scores_b = tile_b["scores"]

    idx_a = get_edge_indices(
        local_boxes_a,
        edge_a,
        tile_size=tile_size,
        overlap=overlap,
    )
    idx_b = get_edge_indices(
        local_boxes_b,
        edge_b,
        tile_size=tile_size,
        overlap=overlap,
    )

    remove_a = []
    remove_b = []

    if len(idx_a) == 0 or len(idx_b) == 0:
        return remove_a, remove_b

    matched_a = set()
    matched_b = set()

    shared_labels = torch.unique(
        torch.cat([labels_a[idx_a], labels_b[idx_b]])
    )

    for label in shared_labels:
        class_idx_a = idx_a[labels_a[idx_a] == label]
        class_idx_b = idx_b[labels_b[idx_b] == label]

        if len(class_idx_a) == 0 or len(class_idx_b) == 0:
            continue

        # IoUを計算して、重複する候補を見つける
        ious = box_iou(
            global_boxes_a[class_idx_a],# Tile A側の候補Box
            global_boxes_b[class_idx_b],# Tile B側の候補Box
        )

        pair_rows, pair_cols = torch.where(ious >= iou_thresh) # IoUが閾値以上のペアのインデックスを取得
        candidate_pairs = []

        for row, col in zip(pair_rows.tolist(), pair_cols.tolist()):
            candidate_pairs.append(
                (
                    ious[row, col].item(),
                    class_idx_a[row].item(),
                    class_idx_b[col].item(),
                )
            )

        candidate_pairs.sort(reverse=True)

        for _, a_idx, b_idx in candidate_pairs:
            if a_idx in matched_a or b_idx in matched_b:
                continue

            matched_a.add(a_idx)
            matched_b.add(b_idx)

            # score を比較している
            if scores_a[a_idx] >= scores_b[b_idx]:
                remove_b.append(b_idx)
            else:
                remove_a.append(a_idx)

    return remove_a, remove_b


def main():
    args = get_args()
    prediction_files = sorted(
        Path(args.prediction_dir).glob("*.pt")
    )
    
    print(f"Found {len(prediction_files)} prediction files.")
    
    if len(prediction_files) == 0:
        raise FileNotFoundError(
            f"No prediction files found in {args.prediction_dir}"
        )
    
    
    tile_predictions = {}

    for pred_file in prediction_files:
        
        prediction = torch.load(
            pred_file,
            map_location="cpu",
        )
        
        tile_x, tile_y = parse_tile_xy(pred_file)

        boxes = prediction["boxes"]
        labels = prediction["labels"]
        scores = prediction["scores"]

        global_boxes = shift_boxes_to_global(
            boxes,
            tile_x,
            tile_y,
            tile_size=args.tile_size,
            overlap=args.overlap,
        )

        overlap_flags = is_in_overlap_region(
            boxes,
            tile_size=args.tile_size,
            overlap=args.overlap,
        )

        tile_predictions[(tile_x, tile_y)] = {
            "local_boxes": boxes,
            "boxes": global_boxes,
            "labels": labels,
            "scores": scores,
            "overlap_flags": overlap_flags,
        }
        
    print(f"Loaded {len(tile_predictions)} tiles.")
    print(list(tile_predictions.keys())[:5])
    
    # Record local detection indices that should be removed from each tile.
    remove_indices_by_tile = {
        tile_key: set()
        for tile_key in tile_predictions
    }

    for (tile_x, tile_y), current_tile in tile_predictions.items():
        right_neighbor = (tile_x + 1, tile_y)
        bottom_neighbor = (tile_x, tile_y + 1)
        
        # 右隣のタイルとの重複を検出して削除するインデックスを取得
        if right_neighbor in tile_predictions:
            remove_current, remove_right = find_duplicate_indices_between_tiles(
                current_tile,
                tile_predictions[right_neighbor],
                edge_a="right",
                edge_b="left",
                tile_size=args.tile_size,
                overlap=args.overlap,
                iou_thresh=args.iou_thresh,
            )
            remove_indices_by_tile[(tile_x, tile_y)].update(remove_current)
            remove_indices_by_tile[right_neighbor].update(remove_right)
            
        # 下隣のタイルとの重複を検出して削除するインデックスを取得
        if bottom_neighbor in tile_predictions:
            remove_current, remove_bottom = find_duplicate_indices_between_tiles(
                current_tile,
                tile_predictions[bottom_neighbor],
                edge_a="bottom",
                edge_b="top",
                tile_size=args.tile_size,
                overlap=args.overlap,
                iou_thresh=args.iou_thresh,
            )
            remove_indices_by_tile[(tile_x, tile_y)].update(remove_current)
            remove_indices_by_tile[bottom_neighbor].update(remove_bottom)

    all_boxes = []
    all_labels = []
    all_scores = []
    removed_count = 0

    for tile_key, tile in tile_predictions.items():
        num_detections = len(tile["boxes"])
        keep_mask = torch.ones(num_detections, dtype=torch.bool)

        indices_to_remove = sorted(remove_indices_by_tile[tile_key])
        if indices_to_remove:
            remove_tensor = torch.tensor(indices_to_remove, dtype=torch.long)
            keep_mask[remove_tensor] = False
            removed_count += len(indices_to_remove)

        all_boxes.append(tile["boxes"][keep_mask])
        all_labels.append(tile["labels"][keep_mask])
        all_scores.append(tile["scores"][keep_mask])

    merged_boxes = torch.cat(all_boxes, dim=0)
    merged_labels = torch.cat(all_labels, dim=0)
    merged_scores = torch.cat(all_scores, dim=0)

    merged_prediction = {
        "boxes": merged_boxes,
        "labels": merged_labels,
        "scores": merged_scores,
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "num_tiles": len(prediction_files),
        "num_removed_duplicates": removed_count,
    }

    output_path = Path(args.output)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(merged_prediction, output_path)
    print(f"Removed duplicate detections: {removed_count}")
    print(f"Merged predictions saved to: {output_path}")


    
if __name__ == "__main__":
    main()