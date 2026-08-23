"""形態・群集解析の集計結果を年代軸で可視化する。"""

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


TARGET_COLORS = {
    "A.subarctica": "#F3A6B8",
    "A.ambigua": "#76C9D8",
    "Fragilariophycea": "#F1DD63",
    "cyclostephanoids": "#A98AD9",
    "plant fragment": "#8FA36B",
}

COMMUNITY_CLASS_ORDER = [
    "cyclostephanoids",
    "Fragilariophycea",
    "A.ambigua",
    "A.subarctica",
]

COUNT_CLASS_ORDER = [
    *COMMUNITY_CLASS_ORDER,
    "plant fragment",
]

# 1検出が2 valvesに相当する分類群。
VALVE_COUNT_MULTIPLIERS = {
    "cyclostephanoids": 2,
    "Fragilariophycea": 2,
}

EXCLUDED_FROM_RELATIVE_ABUNDANCE = {
    "plant fragment", "buble", "debri", "band",
}


def get_args():
    """年代軸グラフの入力CSVと出力先をコマンドラインから受け取る。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--age_mapping_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def natural_key(value):
    """試料名に含まれる数字を数値として扱う層序順ソート用キーを作る。"""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def load_age_mapping(path):
    """対応表CSVから、対応が確認できた試料の年代だけを読み込む。"""
    mapping = {}
    with Path(path).open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["mapping_status"] == "matched" and row["age_calBP_plot"]:
                mapping[row["sample"]] = float(row["age_calBP_plot"])
    return mapping


def load_layer_metadata(path):
    """対応表CSVから層ID・年代・対応状態を試料ごとに読み込む。"""
    metadata = {}
    with Path(path).open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            metadata[row["sample"]] = row
    return metadata


def load_summary(path):
    """検出数を読み込み、valve数補正後の相対産出率を計算する。"""
    values = {}
    samples = set()
    source_rows = []
    with Path(path).open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            source_rows.append(row)
            sample = row["sample"]
            samples.add(sample)

    valve_totals = defaultdict(float)
    for row in source_rows:
        if row["class_name"] in EXCLUDED_FROM_RELATIVE_ABUNDANCE:
            continue
        multiplier = VALVE_COUNT_MULTIPLIERS.get(row["class_name"], 1)
        valve_totals[row["sample"]] += float(row["count"]) * multiplier

    for row in source_rows:
        sample = row["sample"]
        class_name = row["class_name"]
        if class_name not in TARGET_COLORS:
            continue
        multiplier = VALVE_COUNT_MULTIPLIERS.get(class_name, 1)
        valve_count = float(row["count"]) * multiplier
        relative_value = (
            100 * valve_count / valve_totals[sample]
            if class_name not in EXCLUDED_FROM_RELATIVE_ABUNDANCE
            and valve_totals[sample] else None
        )
        values[(sample, class_name)] = {
            "count": float(row["count"]),
            "valve_equivalent_count": valve_count,
            "relative_abundance_percent": relative_value,
        }
    return samples, values


def save_valve_corrected_summary(samples, values, output_path):
    """valve数補正の内訳と相対産出率を試料ごとに保存する。"""
    fields = [
        "sample", "class_name", "detection_count", "valve_multiplier",
        "valve_equivalent_count", "relative_abundance_percent",
    ]
    with Path(output_path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for sample in sorted(samples, key=natural_key):
            for class_name in COUNT_CLASS_ORDER:
                observation = values.get(
                    (sample, class_name),
                    {
                        "count": 0.0,
                        "valve_equivalent_count": 0.0,
                        "relative_abundance_percent": None,
                    },
                )
                writer.writerow({
                    "sample": sample,
                    "class_name": class_name,
                    "detection_count": observation["count"],
                    "valve_multiplier": VALVE_COUNT_MULTIPLIERS.get(class_name, 1),
                    "valve_equivalent_count": observation["valve_equivalent_count"],
                    "relative_abundance_percent": (
                        observation["relative_abundance_percent"]
                        if class_name in COMMUNITY_CLASS_ORDER else ""
                    ),
                })


def aggregate_by_age(samples, values, age_mapping):
    """同じ年代に対応する複数試料を分類群ごとに平均する。"""
    grouped = defaultdict(lambda: defaultdict(list))
    samples_by_age = defaultdict(list)
    for sample in sorted(samples):
        if sample not in age_mapping:
            continue
        age = age_mapping[sample]
        samples_by_age[age].append(sample)
        for class_name in COUNT_CLASS_ORDER:
            grouped[age][class_name].append(
                values.get(
                    (sample, class_name),
                    {"count": 0.0, "relative_abundance_percent": 0.0},
                )
            )

    rows = []
    for age in sorted(grouped):
        for class_name in COUNT_CLASS_ORDER:
            observations = grouped[age][class_name]
            rows.append(
                {
                    "age_calBP": age,
                    "samples": ";".join(samples_by_age[age]),
                    "n_samples": len(samples_by_age[age]),
                    "class_name": class_name,
                    "count_mean": statistics.fmean(
                        item["count"] for item in observations
                    ),
                    "relative_abundance_percent_mean": (
                        statistics.fmean(
                            item["relative_abundance_percent"]
                            for item in observations
                        )
                        if class_name in COMMUNITY_CLASS_ORDER else ""
                    ),
                }
            )
    return rows


def save_age_summary(rows, output_path):
    """年代ごとに平均した群集組成を監査用CSVとして保存する。"""
    fields = [
        "age_calBP",
        "samples",
        "n_samples",
        "class_name",
        "count_mean",
        "relative_abundance_percent_mean",
    ]
    with Path(output_path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_layer_order_rows(samples, values, layer_metadata):
    """全試料を自然な層序順に並べ、分類群ごとの値を1層ずつ保持する。"""
    rows = []
    for layer_order, sample in enumerate(sorted(samples, key=natural_key), start=1):
        metadata = layer_metadata.get(sample, {})
        for class_name in COUNT_CLASS_ORDER:
            observation = values.get(
                (sample, class_name),
                {"count": 0.0, "relative_abundance_percent": 0.0},
            )
            rows.append(
                {
                    "layer_order": layer_order,
                    "sample": sample,
                    "layer_id": metadata.get("layer_id", ""),
                    "age_calBP_exact": metadata.get("age_calBP_exact", ""),
                    "age_calBP_plot": metadata.get("age_calBP_plot", ""),
                    "mapping_status": metadata.get("mapping_status", "not_found"),
                    "class_name": class_name,
                    "count": observation["count"],
                    "relative_abundance_percent": (
                        observation["relative_abundance_percent"]
                        if class_name in COMMUNITY_CLASS_ORDER else ""
                    ),
                }
            )
    return rows


def save_layer_order_summary(rows, output_path):
    """層序順に並べた試料別の群集組成をCSVとして保存する。"""
    fields = [
        "layer_order",
        "sample",
        "layer_id",
        "age_calBP_exact",
        "age_calBP_plot",
        "mapping_status",
        "class_name",
        "count",
        "relative_abundance_percent",
    ]
    with Path(output_path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_layer_order_plots(rows, output_dir):
    """珪藻の相対産出率とplant fragment検出数を別ファイルで描画する。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    samples = []
    for row in rows:
        if row["sample"] not in samples:
            samples.append(row["sample"])
    lookup = {(row["sample"], row["class_name"]): row for row in rows}
    y_positions = list(range(len(samples)))

    for value_key, axis_label, stem in ((
        "relative_abundance_percent",
        "Valve relative abundance (%)",
        "diatom_identification_relative_abundance_by_layer_order",
    ),):
        plot_class_order = COMMUNITY_CLASS_ORDER
        shared_xmax = max(
            float(lookup[(sample, "A.subarctica")][value_key])
            for sample in samples
        ) * 1.05
        fig, axes = plt.subplots(
            1,
            len(plot_class_order),
            figsize=(13.5, 9.0),
            sharey=True,
            constrained_layout=True,
        )
        for axis, class_name in zip(axes, plot_class_order):
            plot_values = [
                float(lookup[(sample, class_name)][value_key])
                for sample in samples
            ]
            color = TARGET_COLORS[class_name]
            axis.barh(
                y_positions,
                plot_values,
                height=1.0,
                color=color,
                alpha=0.72,
                edgecolor="#8A8A8A",
                linewidth=0.5,
            )
            axis.plot(
                plot_values,
                y_positions,
                "o--",
                color="black",
                linewidth=1,
                markersize=3,
            )
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
            axis.set_ylim(len(samples) - 0.5, -0.5)

        axes[0].set_yticks(y_positions, labels=samples)
        axes[0].tick_params(axis="y", labelleft=True)
        axes[0].set_ylabel("Layer order (sample)")
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(
                output_dir / f"{stem}.{suffix}",
                dpi=300 if suffix == "png" else None,
            )
        plt.close(fig)

    plant_values = [
        float(lookup[(sample, "plant fragment")]["count"])
        for sample in samples
    ]
    fig, axis = plt.subplots(figsize=(5.5, 9.0), constrained_layout=True)
    axis.barh(
        y_positions, plant_values, height=1.0,
        color=TARGET_COLORS["plant fragment"], alpha=0.72,
        edgecolor="#8A8A8A", linewidth=0.5,
    )
    axis.plot(plant_values, y_positions, "o--", color="black", linewidth=1, markersize=3)
    axis.set_title("plant fragment", color=TARGET_COLORS["plant fragment"], fontsize=13)
    axis.set_xlabel("Count")
    axis.xaxis.set_label_position("top")
    axis.xaxis.tick_top()
    axis.grid(axis="x", color="#D7D7D7", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.set_xlim(0, max(plant_values) * 1.05 if max(plant_values) > 0 else 1)
    axis.set_ylim(len(samples) - 0.5, -0.5)
    axis.set_yticks(y_positions, labels=samples)
    axis.set_ylabel("Layer order (sample)")
    stem = "plant_fragment_counts_by_layer_order"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            output_dir / f"{stem}.{suffix}",
            dpi=300 if suffix == "png" else None,
        )
    plt.close(fig)


def make_age_bin_edges(ages):
    """隣接年代の中点を境界にして、隙間のない年代binを作る。"""
    if len(ages) == 1:
        return [ages[0] - 0.5, ages[0] + 0.5]
    edges = [ages[0] - (ages[1] - ages[0]) / 2]
    edges.extend((left + right) / 2 for left, right in zip(ages, ages[1:]))
    edges.append(ages[-1] + (ages[-1] - ages[-2]) / 2)
    return edges


def make_age_plots(rows, output_dir):
    """珪藻の相対産出率とplant fragment検出数を年代軸で別々に描画する。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ages = sorted({float(row["age_calBP"]) for row in rows})
    lookup = {
        (float(row["age_calBP"]), row["class_name"]): row
        for row in rows
    }
    bin_edges = make_age_bin_edges(ages)
    bin_lowers = bin_edges[:-1]
    bin_heights = [upper - lower for lower, upper in zip(bin_edges, bin_edges[1:])]
    first_tick = math.ceil(min(ages) / 50) * 50
    last_tick = math.floor(max(ages) / 50) * 50
    age_ticks = list(range(first_tick, last_tick + 1, 50))

    for value_key, axis_label, stem in ((
        "relative_abundance_percent_mean",
        "Mean valve relative abundance (%)",
        "diatom_identification_relative_abundance_by_age",
    ),):
        plot_class_order = COMMUNITY_CLASS_ORDER
        shared_xmax = max(
            float(lookup[(age, "A.subarctica")][value_key])
            for age in ages
        ) * 1.05
        fig, axes = plt.subplots(
            1,
            len(plot_class_order),
            figsize=(13.5, 9.0),
            sharey=True,
            constrained_layout=True,
        )
        for axis, class_name in zip(axes, plot_class_order):
            values = [
                float(lookup[(age, class_name)][value_key])
                for age in ages
            ]
            color = TARGET_COLORS[class_name]
            axis.barh(
                bin_lowers,
                values,
                height=bin_heights,
                align="edge",
                color=color,
                alpha=0.72,
                edgecolor="#8A8A8A",
                linewidth=0.5,
            )
            axis.plot(values, ages, "o--", color="black", linewidth=1, markersize=3)
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

        axes[0].set_ylabel("Age (cal BP)")
        axes[0].set_yticks(age_ticks)
        axes[0].set_yticklabels([str(age) for age in age_ticks])
        axes[0].tick_params(axis="y", labelleft=True)
        axes[0].invert_yaxis()
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(
                output_dir / f"{stem}.{suffix}",
                dpi=300 if suffix == "png" else None,
            )
        plt.close(fig)

    plant_values = [
        float(lookup[(age, "plant fragment")]["count_mean"])
        for age in ages
    ]
    fig, axis = plt.subplots(figsize=(5.5, 9.0), constrained_layout=True)
    axis.barh(
        bin_lowers, plant_values, height=bin_heights, align="edge",
        color=TARGET_COLORS["plant fragment"], alpha=0.72,
        edgecolor="#8A8A8A", linewidth=0.5,
    )
    axis.plot(plant_values, ages, "o--", color="black", linewidth=1, markersize=3)
    axis.set_title("plant fragment", color=TARGET_COLORS["plant fragment"], fontsize=13)
    axis.set_xlabel("Mean count")
    axis.xaxis.set_label_position("top")
    axis.xaxis.tick_top()
    axis.grid(axis="x", color="#D7D7D7", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.set_xlim(0, max(plant_values) * 1.05 if max(plant_values) > 0 else 1)
    axis.set_ylabel("Age (cal BP)")
    axis.set_yticks(age_ticks)
    axis.set_yticklabels([str(age) for age in age_ticks])
    axis.invert_yaxis()
    stem = "plant_fragment_counts_by_age"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            output_dir / f"{stem}.{suffix}",
            dpi=300 if suffix == "png" else None,
        )
    plt.close(fig)


def main():
    """群集集計から年代別・層序順のCSVとグラフを作成する。"""
    args = get_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    age_mapping = load_age_mapping(args.age_mapping_csv)
    layer_metadata = load_layer_metadata(args.age_mapping_csv)
    samples, values = load_summary(args.summary_csv)
    save_valve_corrected_summary(
        samples, values, output_dir / "identification_summary_valve_corrected.csv"
    )
    rows = aggregate_by_age(samples, values, age_mapping)
    save_age_summary(rows, output_dir / "identification_summary_by_age.csv")
    make_age_plots(rows, output_dir)
    layer_rows = build_layer_order_rows(samples, values, layer_metadata)
    save_layer_order_summary(
        layer_rows, output_dir / "identification_summary_by_layer_order.csv"
    )
    make_layer_order_plots(layer_rows, output_dir)
    missing = sorted(samples - set(age_mapping))
    print(f"Mapped samples: {len(samples) - len(missing)}")
    print(f"Missing age mapping: {', '.join(missing) if missing else 'none'}")


if __name__ == "__main__":
    main()
