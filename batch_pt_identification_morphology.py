"""Batch-process multiple samples of tile-level ``.pt`` predictions."""

import argparse
import csv
import re
import tempfile
from collections import Counter
from pathlib import Path

from batch_identification_morphology import (
    EXCLUDED_FROM_RELATIVE_ABUNDANCE,
    VALVE_COUNT_MULTIPLIERS,
    make_plots,
    write_identification_summary,
)
from process_json_predictions import MORPHOLOGY_COLUMNS
from process_pt_predictions import load_class_map, process_pt_directory
from utils.archive import extract_tar_zst


def natural_key(value):
    """Build a natural-sort key for sample names containing numbers."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def get_args():
    """Define command-line arguments for multi-sample ``.pt`` processing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        required=True,
        help=(
            "Directory containing one subdirectory per sample and/or .tar.zst "
            "prediction archives"
        ),
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--class_map",
        default=None,
        help="Optional class map JSON or COCO annotation JSON",
    )
    parser.add_argument("--tile_size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=180)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--mask_thresh", type=float, default=0.5)
    parser.add_argument(
        "--save_merged",
        action="store_true",
        help="Also save duplicate-removed merged .pt files for each sample",
    )
    return parser.parse_args()


def _archive_sample_name(path):
    """Remove common prediction-archive suffixes from a sample name."""
    name = Path(path).name
    for suffix in ("_infer_pt.tar.zst", "_inference.tar.zst", ".tar.zst"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return Path(path).stem


def discover_sample_sources(input_dir):
    """Find sample directories and ``.tar.zst`` archives below an input root."""
    input_dir = Path(input_dir)
    sources = []
    direct_pt_files = list(input_dir.glob("*.pt"))
    if direct_pt_files:
        sources.append((input_dir.name, "directory", input_dir))

    for directory in sorted(
        (path for path in input_dir.iterdir() if path.is_dir()),
        key=lambda path: natural_key(path.name),
    ):
        if any(directory.rglob("*.pt")):
            sources.append((directory.name, "directory", directory))
    for archive in sorted(input_dir.glob("*.tar.zst"), key=natural_key):
        sources.append((_archive_sample_name(archive), "archive", archive))

    sample_names = [sample for sample, _, _ in sources]
    duplicates = sorted(
        sample for sample, count in Counter(sample_names).items() if count > 1
    )
    if duplicates:
        raise ValueError(
            "Duplicate sample sources found for: " + ", ".join(duplicates)
        )
    if not sources:
        raise FileNotFoundError(
            f"No sample directories or .tar.zst archives containing .pt files in: {input_dir}"
        )
    return sorted(sources, key=lambda item: natural_key(item[0]))


def find_pt_directory(root):
    """Choose the extracted directory containing the most tile ``.pt`` files."""
    counts = Counter(path.parent for path in Path(root).rglob("*.pt"))
    if not counts:
        raise FileNotFoundError(f"No .pt predictions found under: {root}")
    return counts.most_common(1)[0][0]


def _append_combined_rows(writer, sample, rows):
    """Write one sample's morphology rows to the combined CSV."""
    for row in rows:
        writer.writerow({"sample": sample, **row})


def _summary_rows_for_sample(sample, rows):
    """Count classes and calculate valve-corrected relative abundance."""
    counts = Counter(row["class_name"] for row in rows)
    class_ids = {row["class_name"]: row["class_id"] for row in rows}
    relative_total = sum(
        count * VALVE_COUNT_MULTIPLIERS.get(class_name, 1)
        for class_name, count in counts.items()
        if class_name not in EXCLUDED_FROM_RELATIVE_ABUNDANCE
    )
    summary_rows = []
    for class_name in sorted(counts):
        count = counts[class_name]
        valve_count = count * VALVE_COUNT_MULTIPLIERS.get(class_name, 1)
        summary_rows.append({
            "sample": sample,
            "class_id": class_ids[class_name],
            "class_name": class_name,
            "count": count,
            "valve_equivalent_count": valve_count,
            "relative_abundance_percent": (
                ""
                if class_name in EXCLUDED_FROM_RELATIVE_ABUNDANCE
                else 100 * valve_count / relative_total if relative_total else 0
            ),
        })
    return summary_rows


def process_batch(
    input_dir,
    output_dir,
    class_map=None,
    tile_size=1200,
    overlap=180,
    iou_thresh=0.5,
    mask_thresh=0.5,
    save_merged=False,
):
    """Process all samples, combine their CSVs and write identification plots."""
    output_dir = Path(output_dir)
    per_sample_dir = output_dir / "per_sample"
    merged_dir = output_dir / "merged_pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_dir.mkdir(parents=True, exist_ok=True)
    if save_merged:
        merged_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(class_map, (str, Path)):
        class_map = load_class_map(class_map)
    sources = discover_sample_sources(input_dir)
    combined_path = output_dir / "morphology_all_samples.csv"
    summary_rows = []
    fieldnames = ["sample", *MORPHOLOGY_COLUMNS]

    with combined_path.open("w", newline="", encoding="utf-8") as combined_file:
        writer = csv.DictWriter(combined_file, fieldnames=fieldnames)
        writer.writeheader()
        for index, (sample, source_type, source_path) in enumerate(sources, start=1):
            print(f"[{index}/{len(sources)}] Processing {sample}")
            with tempfile.TemporaryDirectory(prefix=f"diatom_pt_{sample}_") as temp_name:
                if source_type == "archive":
                    extracted = extract_tar_zst(
                        source_path, output_dir=Path(temp_name) / "extracted"
                    )
                    prediction_dir = find_pt_directory(extracted)
                else:
                    prediction_dir = find_pt_directory(source_path)

                sample_csv = per_sample_dir / f"{sample}_morphology.csv"
                merged_output = (
                    merged_dir / f"{sample}_merged_predictions.pt"
                    if save_merged else None
                )
                rows, _ = process_pt_directory(
                    prediction_dir=prediction_dir,
                    csv_output=sample_csv,
                    merged_output=merged_output,
                    class_map=class_map,
                    tile_size=tile_size,
                    overlap=overlap,
                    iou_thresh=iou_thresh,
                    mask_thresh=mask_thresh,
                )
                _append_combined_rows(writer, sample, rows)
                summary_rows.extend(_summary_rows_for_sample(sample, rows))

    summary_path = output_dir / "identification_summary.csv"
    write_identification_summary(summary_rows, summary_path)
    if summary_rows:
        make_plots(summary_rows, output_dir)
    print(f"Combined morphology: {combined_path}")
    print(f"Identification summary: {summary_path}")
    return combined_path, summary_path


def main():
    """Process all sample sources supplied on the command line."""
    args = get_args()
    process_batch(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        class_map=args.class_map,
        tile_size=args.tile_size,
        overlap=args.overlap,
        iou_thresh=args.iou_thresh,
        mask_thresh=args.mask_thresh,
        save_merged=args.save_merged,
    )


if __name__ == "__main__":
    main()
