"""Extract an archive, remove duplicates, and calculate morphology."""

import argparse
import subprocess
import sys
from pathlib import Path

from utils.archive import extract_tar_zst


def get_args():
    """単一アーカイブの後処理に必要なコマンドライン引数を返す。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output_dir", default="postprocess")
    parser.add_argument("--tile_size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=180)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    return parser.parse_args()


def find_json_directory(extracted_dir):
    """展開先から予測JSONが最も多いフォルダを選ぶ。"""
    counts_by_parent = {}
    for path in Path(extracted_dir).rglob("*.json"):
        counts_by_parent[path.parent] = counts_by_parent.get(path.parent, 0) + 1
    if not counts_by_parent:
        raise FileNotFoundError(f"No prediction JSON files found under: {extracted_dir}")
    return max(counts_by_parent, key=counts_by_parent.get)


def main():
    """アーカイブ展開から重複除去・形態解析までをまとめて実行する。"""
    args = get_args()
    project_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir = extract_tar_zst(
        args.archive, output_dir=output_dir / "extracted"
    )
    prediction_dir = find_json_directory(extracted_dir)
    merged_path = output_dir / "merged_predictions.json"
    morphology_path = output_dir / "morphology.csv"
    subprocess.run([
        sys.executable,
        str(project_dir / "process_json_predictions.py"),
        "--prediction_dir", str(prediction_dir),
        "--merged_output", str(merged_path),
        "--csv_output", str(morphology_path),
        "--tile_size", str(args.tile_size),
        "--overlap", str(args.overlap),
        "--iou_thresh", str(args.iou_thresh),
    ], check=True)
    print(f"Merged predictions: {merged_path}")
    print(f"Morphology CSV: {morphology_path}")


if __name__ == "__main__":
    main()
