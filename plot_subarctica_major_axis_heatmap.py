"""A. subarcticaの長軸頻度分布を年代軸ヒートマップとして出力する。"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


TARGET_CLASS = "A.subarctica"


def natural_key(value):
    """試料名の数字を数値として扱い、層序に沿った自然順キーを作る。"""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def get_args():
    """入力CSV、画素サイズ、bin設定、出力先をコマンドラインから受け取る。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--morphology_csv", required=True)
    parser.add_argument("--age_mapping_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pixel_size_um", type=float, default=0.1369)
    parser.add_argument("--major_min_um", type=float, default=6.0)
    parser.add_argument("--major_max_um", type=float, default=17.0)
    parser.add_argument("--bin_width_um", type=float, default=0.5)
    return parser.parse_args()


def load_age_mapping(path):
    """年代対応表から年代が割り当てられた試料だけを読み込む。"""
    mapping = {}
    with Path(path).open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["mapping_status"] == "matched" and row["age_calBP_plot"]:
                mapping[row["sample"]] = float(row["age_calBP_plot"])
    return mapping


def load_major_axes(path, pixel_size_um):
    """形態解析CSVからA. subarcticaの長軸を読み、pixelからµmへ変換する。"""
    values_by_sample = defaultdict(list)
    with Path(path).open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["class_name"] != TARGET_CLASS or not row["major_axis_length_px"]:
                continue
            values_by_sample[row["sample"]].append(
                float(row["major_axis_length_px"]) * pixel_size_um
            )
    return values_by_sample


def make_edges(minimum, maximum, width):
    """指定範囲を一定幅に区切った長軸bin境界を作る。"""
    return np.arange(minimum, maximum + width / 2, width, dtype=float)


def make_irregular_edges(centers):
    """不等間隔の年代点の中点から隙間のない年代bin境界を作る。"""
    centers = np.asarray(centers, dtype=float)
    if len(centers) == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    midpoints = (centers[:-1] + centers[1:]) / 2
    first = centers[0] - (midpoints[0] - centers[0])
    last = centers[-1] + (centers[-1] - midpoints[-1])
    return np.concatenate(([first], midpoints, [last]))


def aggregate_frequency_by_age(values_by_sample, age_mapping, major_edges):
    """試料ごとの頻度分布を計算し、同年代の試料間で算術平均する。"""
    sample_histograms = defaultdict(list)
    samples_by_age = defaultdict(list)
    included = 0
    total = 0

    for sample, values in values_by_sample.items():
        if sample not in age_mapping:
            continue
        values = np.asarray(values, dtype=float)
        counts, _ = np.histogram(values, bins=major_edges)
        in_range = int(counts.sum())
        total += len(values)
        included += in_range
        frequency = counts / in_range * 100 if in_range else np.zeros_like(counts)
        age = age_mapping[sample]
        sample_histograms[age].append((counts.astype(float), frequency))
        samples_by_age[age].append(sample)

    ages = np.array(sorted(sample_histograms), dtype=float)
    count_matrix = np.vstack(
        [
            np.mean([item[0] for item in sample_histograms[age]], axis=0)
            for age in ages
        ]
    )
    frequency_matrix = np.vstack(
        [
            np.mean([item[1] for item in sample_histograms[age]], axis=0)
            for age in ages
        ]
    )
    return ages, count_matrix, frequency_matrix, samples_by_age, included, total


def save_frequency_csv(
    output_path, ages, major_edges, count_matrix, frequency_matrix, samples_by_age
):
    """年代・長軸binごとの平均個体数と平均頻度を長形式CSVで保存する。"""
    fields = [
        "age_calBP",
        "samples",
        "n_samples",
        "major_axis_bin_left_um",
        "major_axis_bin_right_um",
        "major_axis_bin_center_um",
        "count_mean",
        "frequency_percent_mean",
    ]
    with Path(output_path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for age_index, age in enumerate(ages):
            samples = sorted(samples_by_age[age])
            for bin_index, (left, right) in enumerate(
                zip(major_edges[:-1], major_edges[1:])
            ):
                writer.writerow(
                    {
                        "age_calBP": age,
                        "samples": ";".join(samples),
                        "n_samples": len(samples),
                        "major_axis_bin_left_um": left,
                        "major_axis_bin_right_um": right,
                        "major_axis_bin_center_um": (left + right) / 2,
                        "count_mean": count_matrix[age_index, bin_index],
                        "frequency_percent_mean": frequency_matrix[
                            age_index, bin_index
                        ],
                    }
                )


def aggregate_frequency_by_layer(values_by_sample, major_edges):
    """各試料の長軸頻度分布を、平均せず層序順に並べる。"""
    samples = sorted(values_by_sample, key=natural_key)
    count_rows = []
    frequency_rows = []
    included = 0
    total = 0
    for sample in samples:
        values = np.asarray(values_by_sample[sample], dtype=float)
        counts, _ = np.histogram(values, bins=major_edges)
        in_range = int(counts.sum())
        total += len(values)
        included += in_range
        count_rows.append(counts.astype(float))
        frequency_rows.append(
            counts / in_range * 100 if in_range else np.zeros_like(counts)
        )
    return (
        samples,
        np.vstack(count_rows),
        np.vstack(frequency_rows),
        included,
        total,
    )


def save_layer_frequency_csv(
    output_path,
    samples,
    major_edges,
    count_matrix,
    frequency_matrix,
    age_mapping,
):
    """層序番号・試料・長軸binごとの個体数と頻度をCSVで保存する。"""
    fields = [
        "layer_order",
        "sample",
        "age_calBP",
        "major_axis_bin_left_um",
        "major_axis_bin_right_um",
        "major_axis_bin_center_um",
        "count",
        "frequency_percent",
    ]
    with Path(output_path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for sample_index, sample in enumerate(samples):
            for bin_index, (left, right) in enumerate(
                zip(major_edges[:-1], major_edges[1:])
            ):
                writer.writerow(
                    {
                        "layer_order": sample_index + 1,
                        "sample": sample,
                        "age_calBP": age_mapping.get(sample, ""),
                        "major_axis_bin_left_um": left,
                        "major_axis_bin_right_um": right,
                        "major_axis_bin_center_um": (left + right) / 2,
                        "count": count_matrix[sample_index, bin_index],
                        "frequency_percent": frequency_matrix[
                            sample_index, bin_index
                        ],
                    }
                )


def make_heatmap(output_dir, ages, major_edges, frequency_matrix):
    """長軸binの頻度を色で表し、年代が下向きに古くなるヒートマップを描く。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    age_edges = make_irregular_edges(ages)
    fig, axis = plt.subplots(figsize=(7.2, 9.0), constrained_layout=True)
    mesh = axis.pcolormesh(
        major_edges,
        age_edges,
        frequency_matrix,
        shading="flat",
        cmap="viridis",
        vmin=0,
    )
    colorbar = fig.colorbar(mesh, ax=axis, pad=0.025)
    colorbar.set_label("Frequency (%)")

    first_tick = int(np.ceil(ages.min() / 50) * 50)
    last_tick = int(np.floor(ages.max() / 50) * 50)
    axis.set_yticks(np.arange(first_tick, last_tick + 1, 50))
    axis.set_ylabel("Age (cal BP)")
    axis.set_xlabel("Major axis (µm)")
    axis.xaxis.set_label_position("top")
    axis.xaxis.tick_top()
    axis.set_title(
        r"$\it{A.\ subarctica}$ major-axis frequency",
        pad=38,
    )
    axis.invert_yaxis()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "a_subarctica_major_axis_frequency_heatmap_by_age"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def make_layer_heatmap(output_dir, samples, major_edges, frequency_matrix):
    """各層の長軸頻度を色で表し、試料を層序順に並べたヒートマップを描く。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layer_edges = np.arange(len(samples) + 1, dtype=float) - 0.5
    fig, axis = plt.subplots(figsize=(7.2, 9.5), constrained_layout=True)
    mesh = axis.pcolormesh(
        major_edges,
        layer_edges,
        frequency_matrix,
        shading="flat",
        cmap="viridis",
        vmin=0,
    )
    colorbar = fig.colorbar(mesh, ax=axis, pad=0.025)
    colorbar.set_label("Frequency (%)")

    layer_positions = np.arange(len(samples))
    axis.set_yticks(layer_positions)
    axis.set_yticklabels(samples)
    axis.set_ylim(len(samples) - 0.5, -0.5)
    axis.set_ylabel("Layer order (sample)")
    axis.set_xlabel("Major axis (µm)")
    axis.xaxis.set_label_position("top")
    axis.xaxis.tick_top()
    axis.set_title(
        r"$\it{A.\ subarctica}$ major-axis frequency",
        pad=38,
    )

    output_dir = Path(output_dir)
    stem = output_dir / "a_subarctica_major_axis_frequency_heatmap_by_layer_order"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def main():
    """長軸値と年代を結合し、頻度表と年代ヒートマップを作成する。"""
    args = get_args()
    major_edges = make_edges(
        args.major_min_um, args.major_max_um, args.bin_width_um
    )
    age_mapping = load_age_mapping(args.age_mapping_csv)
    values_by_sample = load_major_axes(args.morphology_csv, args.pixel_size_um)
    (
        ages,
        count_matrix,
        frequency_matrix,
        samples_by_age,
        included,
        total,
    ) = aggregate_frequency_by_age(values_by_sample, age_mapping, major_edges)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_frequency_csv(
        output_dir / "a_subarctica_major_axis_frequency_by_age.csv",
        ages,
        major_edges,
        count_matrix,
        frequency_matrix,
        samples_by_age,
    )
    make_heatmap(output_dir, ages, major_edges, frequency_matrix)
    (
        layer_samples,
        layer_count_matrix,
        layer_frequency_matrix,
        layer_included,
        layer_total,
    ) = aggregate_frequency_by_layer(values_by_sample, major_edges)
    save_layer_frequency_csv(
        output_dir / "a_subarctica_major_axis_frequency_by_layer_order.csv",
        layer_samples,
        major_edges,
        layer_count_matrix,
        layer_frequency_matrix,
        age_mapping,
    )
    make_layer_heatmap(
        output_dir, layer_samples, major_edges, layer_frequency_matrix
    )
    print(f"Age bins: {len(ages)}")
    print(f"Major-axis bins: {len(major_edges) - 1}")
    print(f"Included in {major_edges[0]:g}-{major_edges[-1]:g} µm: {included}/{total}")
    print(f"Layers: {len(layer_samples)}")
    print(
        f"Layer plot included in {major_edges[0]:g}-{major_edges[-1]:g} µm: "
        f"{layer_included}/{layer_total}"
    )


if __name__ == "__main__":
    main()
