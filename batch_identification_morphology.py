"""Batch-process prediction archives and summarize diatom identifications."""

import argparse
import csv
import json
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from process_json_predictions import process_json_directory
from utils.archive import extract_tar_zst


TARGET_COLORS = {
    "A.subarctica": "#F3A6B8",
    "A.ambigua": "#76C9D8",
    "Fragilariophycea": "#F1DD63",
    "cyclostephanoids": "#A98AD9",
    "plant fragment": "#8FA36B",
}

# 独立した珪藻個体として数えないため、相対産出率の分母から除外するクラス。
EXCLUDED_FROM_RELATIVE_ABUNDANCE = {
    "plant fragment",
    "buble",
    "debri",
    "band",
}

# 1検出が2 valvesに相当する分類群は、相対産出率計算時に2倍する。
VALVE_COUNT_MULTIPLIERS = {
    "cyclostephanoids": 2,
    "Fragilariophycea": 2,
}


def natural_key(value):
    """試料名に含まれる数字を数値として扱う自然順ソート用キーを作る。"""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def archive_sample_name(path):
    """予測アーカイブのファイル名から試料名を取り出す。"""
    name = Path(path).name
    for suffix in ("_infer_json-2.tar.zst", "_infer_json.tar.zst", ".tar.zst"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return Path(path).stem


def find_json_directory(extracted_dir):
    """展開先からJSONが最も多く入っているフォルダを見つける。"""
    counts = Counter(path.parent for path in Path(extracted_dir).rglob("*.json"))
    if not counts:
        raise FileNotFoundError(f"No JSON predictions found under: {extracted_dir}")
    return counts.most_common(1)[0][0]


def get_args():
    """一括処理に使用するコマンドライン引数を定義して返す。"""
    parser = argparse.ArgumentParser(
        description="Run identification and morphology analysis for all .tar.zst archives."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tile_size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=180)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    return parser.parse_args()


def write_identification_summary(rows, output_path):
    """試料・分類群ごとの個体数と相対産出率をCSVへ保存する。"""
    fields = [
        "sample", "class_id", "class_name", "count",
        "valve_equivalent_count",
        "relative_abundance_percent",
    ]
    with Path(output_path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(summary_rows, output_dir):
    """珪藻の相対産出率とplant fragment検出数を別ファイルで保存する。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = sorted({row["sample"] for row in summary_rows}, key=natural_key)
    community_class_order = [
        "cyclostephanoids", "Fragilariophycea", "A.ambigua", "A.subarctica",
    ]
    lookup = {(row["sample"], row["class_name"]): row for row in summary_rows}

    for value_key, axis_label, stem in ((
        "relative_abundance_percent", "Valve relative abundance (%)",
        "diatom_identification_relative_abundance",
    ),):
        class_order = community_class_order
        shared_xmax = max(
            float(lookup.get((sample, "A.subarctica"), {}).get(value_key, 0))
            for sample in samples
        ) * 1.05
        fig, axes = plt.subplots(
            1, len(class_order),
            figsize=(13.5, 7.5),
            sharey=True, constrained_layout=True,
        )
        y_positions = list(range(len(samples)))
        for axis, class_name in zip(axes, class_order):
            values = [
                float(lookup.get((sample, class_name), {}).get(value_key, 0))
                for sample in samples
            ]
            color = TARGET_COLORS[class_name]
            axis.barh(
                y_positions, values, color=color, alpha=0.72,
                edgecolor=color, linewidth=0.8,
            )
            axis.plot(values, y_positions, "o--", color="black", linewidth=1, markersize=3)
            axis.set_title(
                class_name,
                color=color,
                fontstyle="normal" if class_name == "plant fragment" else "italic",
                fontsize=13,
            )
            axis.set_xlabel(axis_label)
            axis.xaxis.set_label_position("top")
            axis.xaxis.tick_top()
            axis.grid(axis="x", color="#D7D7D7", linewidth=0.6)
            axis.set_axisbelow(True)
            axis.set_xlim(0, shared_xmax)
            axis.margins(y=0.04)
        axes[0].set_yticks(y_positions, labels=samples)
        axes[0].tick_params(axis="y", labelleft=True)
        axes[0].invert_yaxis()
        axes[0].set_ylabel("Sample")
        png_path = Path(output_dir) / f"{stem}.png"
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(
                Path(output_dir) / f"{stem}.{suffix}",
                dpi=300 if suffix == "png" else None,
                bbox_inches="tight",
            )
        plt.close(fig)
        print(f"Saved plot: {png_path}")

    values = [
        float(lookup.get((sample, "plant fragment"), {}).get("count", 0))
        for sample in samples
    ]
    y_positions = list(range(len(samples)))
    fig, axis = plt.subplots(figsize=(5.5, 7.5), constrained_layout=True)
    axis.barh(
        y_positions, values, color=TARGET_COLORS["plant fragment"], alpha=0.72,
        edgecolor=TARGET_COLORS["plant fragment"], linewidth=0.8,
    )
    axis.plot(values, y_positions, "o--", color="black", linewidth=1, markersize=3)
    axis.set_title("plant fragment", color=TARGET_COLORS["plant fragment"], fontsize=13)
    axis.set_xlabel("Count")
    axis.xaxis.set_label_position("top")
    axis.xaxis.tick_top()
    axis.grid(axis="x", color="#D7D7D7", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.set_xlim(0, max(values) * 1.05 if max(values) > 0 else 1)
    axis.set_yticks(y_positions, labels=samples)
    axis.invert_yaxis()
    axis.set_ylabel("Sample")
    stem = "plant_fragment_counts"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            Path(output_dir) / f"{stem}.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def main():
    """全アーカイブを順に解析し、統合CSVとグラフを作成する。"""
    args = get_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    per_sample_dir = output_dir / "per_sample"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_dir.mkdir(parents=True, exist_ok=True)
    archives = sorted(input_dir.glob("*.tar.zst"), key=natural_key)
    if not archives:
        raise FileNotFoundError(f"No .tar.zst archives found in: {input_dir}")

    combined_morphology_path = output_dir / "morphology_all_samples.csv"
    summary_rows = []
    with combined_morphology_path.open("w", newline="", encoding="utf-8") as combined_file:
        combined_writer = None
        for archive_index, archive_path in enumerate(archives, start=1):
            sample = archive_sample_name(archive_path)
            print(f"[{archive_index}/{len(archives)}] Processing {sample}")
            with tempfile.TemporaryDirectory(prefix=f"diatom_{sample}_") as temp_name:
                temp_dir = Path(temp_name)
                extracted_dir = extract_tar_zst(
                    archive_path, output_dir=temp_dir / "extracted"
                )
                prediction_dir = find_json_directory(extracted_dir)
                merged_path = temp_dir / "merged_predictions.json"
                morphology_path = temp_dir / "morphology.csv"
                process_json_directory(
                    prediction_dir, merged_path, morphology_path,
                    tile_size=args.tile_size, overlap=args.overlap,
                    iou_thresh=args.iou_thresh,
                )
                shutil.copy2(
                    morphology_path, per_sample_dir / f"{sample}_morphology.csv"
                )
                with morphology_path.open(encoding="utf-8") as file:
                    reader = csv.DictReader(file)
                    fieldnames = ["sample"] + reader.fieldnames
                    if combined_writer is None:
                        combined_writer = csv.DictWriter(combined_file, fieldnames=fieldnames)
                        combined_writer.writeheader()
                    counts, class_ids, total = Counter(), {}, 0
                    for row in reader:
                        total += 1
                        counts[row["class_name"]] += 1
                        class_ids[row["class_name"]] = int(row["class_id"])
                        combined_writer.writerow({"sample": sample, **row})
                for class_name in sorted(counts):
                    count = counts[class_name]
                    valve_count = count * VALVE_COUNT_MULTIPLIERS.get(class_name, 1)
                    relative_total = sum(
                        class_count * VALVE_COUNT_MULTIPLIERS.get(name, 1)
                        for name, class_count in counts.items()
                        if name not in EXCLUDED_FROM_RELATIVE_ABUNDANCE
                    )
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

    summary_path = output_dir / "identification_summary.csv"
    write_identification_summary(summary_rows, summary_path)
    with (output_dir / "color_mapping.json").open("w", encoding="utf-8") as file:
        json.dump(TARGET_COLORS, file, ensure_ascii=False, indent=2)
    make_plots(summary_rows, output_dir)
    print(f"Combined morphology: {combined_morphology_path}")
    print(f"Identification summary: {summary_path}")


if __name__ == "__main__":
    main()
