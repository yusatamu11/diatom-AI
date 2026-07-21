"""Merge tile JSON predictions and calculate polygon morphology."""

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from skimage.draw import polygon as draw_polygon
from skimage.measure import regionprops


AXIS_LENGTH_CLASSES = {
    "A.subarctica",
    "A.ambigua",
    "cyclostephanoids",
}

MORPHOLOGY_COLUMNS = [
    "instance_id", "class_id", "class_name", "score", "source_image",
    "source_tile_x", "source_tile_y", "bbox_x1_px", "bbox_y1_px",
    "bbox_x2_px", "bbox_y2_px", "bbox_width_px", "bbox_height_px",
    "bbox_area_px2", "morphology_valid", "mask_area_px2", "perimeter_px",
    "perimeter_crofton_px", "centroid_x_px", "centroid_y_px",
    "axis_length_calculated", "major_axis_length_px", "minor_axis_length_px",
    "aspect_ratio", "equivalent_diameter_px", "eccentricity", "circularity",
    "circularity_crofton", "orientation_deg", "extent", "solidity",
]

MORPHOLOGY_KEYS = (
    "area", "perimeter", "perimeter_crofton", "centroid_x", "centroid_y",
    "major_axis", "minor_axis", "aspect_ratio", "equivalent_diameter",
    "eccentricity", "circularity", "circularity_crofton", "orientation_deg",
    "extent", "solidity",
)


def get_args():
    """コマンドライン引数を定義し、入力された設定値を返す。"""
    parser = argparse.ArgumentParser(
        description="Merge tile JSON predictions and calculate morphology."
    )
    parser.add_argument("--prediction_dir", required=True)
    parser.add_argument("--merged_output", default="merged_predictions.json")
    parser.add_argument("--csv_output", default="morphology.csv")
    parser.add_argument("--tile_size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=180)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    return parser.parse_args()


def parse_tile_xy(path):
    """ファイル名の `x番号_y番号` からタイル座標を取得する。"""
    match = re.search(r"x(\d+)_y(\d+)", Path(path).stem)
    if match is None:
        raise ValueError(f"Cannot parse tile coordinates from: {path}")
    return int(match.group(1)), int(match.group(2))


def box_iou(box_a, box_b):
    """2つのバウンディングボックスが重なる割合（IoU）を計算する。"""
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(
        0.0, float(box_a[3]) - float(box_a[1])
    )
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(
        0.0, float(box_b[3]) - float(box_b[1])
    )
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _empty_morphology():
    """形態解析できない個体に使用する欠損値の辞書を作る。"""
    return {key: math.nan for key in MORPHOLOGY_KEYS}


def polygon_morphology(polygon):
    """輪郭座標を二値マスクに変換し、regionpropsで形態量を測定する。"""
    points = np.asarray(polygon, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        return _empty_morphology()

    origin_x = math.floor(float(points[:, 0].min())) - 1
    origin_y = math.floor(float(points[:, 1].min())) - 1
    max_x = math.ceil(float(points[:, 0].max())) + 1
    max_y = math.ceil(float(points[:, 1].max())) + 1
    width = max_x - origin_x + 1
    height = max_y - origin_y + 1

    rows, columns = draw_polygon(
        points[:, 1] - origin_y,
        points[:, 0] - origin_x,
        shape=(height, width),
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[rows, columns] = 1
    regions = regionprops(mask)
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


def load_tiles(prediction_dir, tile_size, overlap):
    """各タイルのJSONを読み込み、座標をスライド全体座標へ変換する。"""
    stride = tile_size - overlap
    tiles = {}
    for json_path in sorted(Path(prediction_dir).glob("*.json")):
        tile_x, tile_y = parse_tile_xy(json_path)
        with json_path.open(encoding="utf-8") as file:
            data = json.load(file)
        origin_x = (tile_x - 1) * stride
        origin_y = (tile_y - 1) * stride
        instances = []
        for instance in data.get("instances", []):
            local_box = [float(value) for value in instance["bbox_xyxy"]]
            global_box = [
                local_box[0] + origin_x, local_box[1] + origin_y,
                local_box[2] + origin_x, local_box[3] + origin_y,
            ]
            global_polygon = [
                [float(point[0]) + origin_x, float(point[1]) + origin_y]
                for point in instance.get("polygon", [])
            ]
            instances.append({
                "class_id": int(instance["class_id"]),
                "class_name": str(instance.get("class_name", "")),
                "score": float(instance["score"]),
                "local_bbox_xyxy": local_box,
                "global_bbox_xyxy": global_box,
                "global_polygon": global_polygon,
                "source_image": data.get("image_name", json_path.name),
                "source_tile_x": tile_x,
                "source_tile_y": tile_y,
            })
        tiles[(tile_x, tile_y)] = instances
    return tiles


def _edge_indices(instances, edge, tile_size, overlap):
    """指定したタイル端のオーバーラップ領域にある個体番号を返す。"""
    indices = []
    for index, instance in enumerate(instances):
        box = instance["local_bbox_xyxy"]
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        left = center_x < overlap
        right = center_x >= tile_size - overlap
        top = center_y < overlap
        bottom = center_y >= tile_size - overlap
        in_edge = {
            "left": left, "right": right, "top": top, "bottom": bottom,
            "top_left": top and left, "top_right": top and right,
            "bottom_left": bottom and left, "bottom_right": bottom and right,
        }[edge]
        if in_edge:
            indices.append(index)
    return indices


def find_duplicates(instances_a, instances_b, edge_a, edge_b, tile_size, overlap, iou_thresh):
    """隣接する2タイル間で重複個体を探し、削除対象を決める。"""
    indices_a = _edge_indices(instances_a, edge_a, tile_size, overlap)
    indices_b = _edge_indices(instances_b, edge_b, tile_size, overlap)
    candidates = []
    for index_a in indices_a:
        for index_b in indices_b:
            instance_a = instances_a[index_a]
            instance_b = instances_b[index_b]
            if instance_a["class_id"] != instance_b["class_id"]:
                continue
            iou = box_iou(
                instance_a["global_bbox_xyxy"], instance_b["global_bbox_xyxy"]
            )
            if iou >= iou_thresh:
                candidates.append((iou, index_a, index_b))

    matched_a, matched_b, remove_a, remove_b = set(), set(), set(), set()
    for _, index_a, index_b in sorted(candidates, reverse=True):
        if index_a in matched_a or index_b in matched_b:
            continue
        matched_a.add(index_a)
        matched_b.add(index_b)
        if instances_a[index_a]["score"] >= instances_b[index_b]["score"]:
            remove_b.add(index_b)
        else:
            remove_a.add(index_a)
    return remove_a, remove_b


def remove_adjacent_duplicates(tiles, tile_size, overlap, iou_thresh):
    """上下左右と斜めに隣接するタイル間の重複個体を除去する。"""
    remove_by_tile = {tile_key: set() for tile_key in tiles}
    for tile_key, instances in tiles.items():
        tile_x, tile_y = tile_key
        for neighbor_key, edge_a, edge_b in (
            ((tile_x + 1, tile_y), "right", "left"),
            ((tile_x, tile_y + 1), "bottom", "top"),
            ((tile_x + 1, tile_y + 1), "bottom_right", "top_left"),
            ((tile_x - 1, tile_y + 1), "bottom_left", "top_right"),
        ):
            if neighbor_key not in tiles:
                continue
            remove_a, remove_b = find_duplicates(
                instances, tiles[neighbor_key], edge_a, edge_b,
                tile_size, overlap, iou_thresh,
            )
            remove_by_tile[tile_key].update(remove_a)
            remove_by_tile[neighbor_key].update(remove_b)

    kept_instances = []
    for tile_key, instances in tiles.items():
        kept_instances.extend(
            instance for index, instance in enumerate(instances)
            if index not in remove_by_tile[tile_key]
        )
    return kept_instances, sum(len(indices) for indices in remove_by_tile.values())


def save_outputs(instances, removed_count, merged_output, csv_output, tile_size, overlap, iou_thresh):
    """統合済みJSONと個体別の形態解析CSVを保存する。"""
    merged_path = Path(merged_output)
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged_data = {
        "format": "diatom-ai-merged-polygon-v1",
        "coordinate_system": "global_slide_pixels",
        "tile_size": tile_size,
        "overlap": overlap,
        "iou_thresh": iou_thresh,
        "num_instances": len(instances),
        "num_removed_duplicates": removed_count,
        "instances": [],
    }
    rows = []
    for instance_id, instance in enumerate(instances, start=1):
        box = instance["global_bbox_xyxy"]
        width = max(0.0, box[2] - box[0])
        height = max(0.0, box[3] - box[1])
        morphology = polygon_morphology(instance["global_polygon"])
        morphology_valid = len(instance["global_polygon"]) >= 3
        axis_length_calculated = (
            morphology_valid and instance["class_name"] in AXIS_LENGTH_CLASSES
        )
        merged_instance = dict(instance)
        merged_instance.pop("local_bbox_xyxy", None)
        merged_instance["instance_id"] = instance_id
        merged_data["instances"].append(merged_instance)
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
            "major_axis_length_px": morphology["major_axis"] if axis_length_calculated else "",
            "minor_axis_length_px": morphology["minor_axis"] if axis_length_calculated else "",
            "aspect_ratio": morphology["aspect_ratio"] if axis_length_calculated else "",
            "equivalent_diameter_px": morphology["equivalent_diameter"],
            "eccentricity": morphology["eccentricity"],
            "circularity": morphology["circularity"],
            "circularity_crofton": morphology["circularity_crofton"],
            "orientation_deg": morphology["orientation_deg"],
            "extent": morphology["extent"],
            "solidity": morphology["solidity"],
        })

    with merged_path.open("w", encoding="utf-8") as file:
        json.dump(merged_data, file, ensure_ascii=False)
    csv_path = Path(csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MORPHOLOGY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Removed duplicate detections: {removed_count}")
    print(f"Remaining instances: {len(instances)}")
    print(f"Merged JSON saved to: {merged_path}")
    print(f"Morphology CSV saved to: {csv_path}")


def process_json_directory(prediction_dir, merged_output, csv_output, tile_size=1200, overlap=180, iou_thresh=0.5):
    """JSON読込、重複除去、形態解析、結果保存を順番に実行する。"""
    tiles = load_tiles(prediction_dir, tile_size, overlap)
    if not tiles:
        raise FileNotFoundError(f"No tile JSON files found in: {prediction_dir}")
    before_count = sum(len(instances) for instances in tiles.values())
    instances, removed_count = remove_adjacent_duplicates(
        tiles, tile_size, overlap, iou_thresh
    )
    print(f"Loaded tiles: {len(tiles)}")
    print(f"Instances before duplicate removal: {before_count}")
    save_outputs(
        instances, removed_count, merged_output, csv_output,
        tile_size, overlap, iou_thresh,
    )


def main():
    """引数を受け取り、JSON予測の後処理を開始する。"""
    args = get_args()
    process_json_directory(
        args.prediction_dir, args.merged_output, args.csv_output,
        args.tile_size, args.overlap, args.iou_thresh,
    )


if __name__ == "__main__":
    main()
