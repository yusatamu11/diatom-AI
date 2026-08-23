"""層準が一致するAIと目視結果だけを年代軸上で比較する。"""

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
}

CLASS_ORDER = [
    "cyclostephanoids",
    "Fragilariophycea",
    "A.ambigua",
    "A.subarctica",
]

# 画像が存在せずAI推論を行っていない層準。AIは空のbin、目視値は点で示す。
MISSING_AI_LAYERS = (
    {
        "layer_id": "A09-172",
        "age_calBP": 14734.2,
        "meme_sample": "DTM-191",
    },
)
MISSING_AI_AGES = {
    row["layer_id"]: row["age_calBP"] for row in MISSING_AI_LAYERS
}

MEME_GROUP_COLUMNS = {
    "cyclostephanoids": [
        "Stephanodiscus suzukii",
        "cyclostephanoids",
        "Puncticulata sp.",
    ],
    "Fragilariophycea": [
        "Asterionella spp.",
        "Synedra spp.",
        "Fragilaria like species",
    ],
    "A.ambigua": ["Aulacoseira ambigua"],
    "A.subarctica": [
        "Aulacoseira subarctica",
        "Aulacoseira subarctica short valves",
    ],
}

MEME_AGE_COLUMN = "Age mid (calBP)"
FIRST_DIATOM_COLUMN = "Aulacoseira subarctica"
LAST_DIATOM_COLUMN = "diatom resting spore?"
LAYER_ID_PATTERN = re.compile(r"^([A-Z])(\d{2})-([0-9.]+)$")
DISTANCE_TOLERANCE_CM = 0.051


def get_args():
    """AI層準別CSV、目視CSV、出力先をコマンドラインから受け取る。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ai_layer_summary", required=True)
    parser.add_argument("--meme_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--distance_tolerance_cm",
        type=float,
        default=DISTANCE_TOLERANCE_CM,
    )
    return parser.parse_args()


def to_float(value):
    """空欄を0としてCSV値を浮動小数点数へ変換する。"""
    text = str(value or "").strip()
    return float(text) if text else 0.0


def parse_layer_id(layer_id):
    """A09-158.8形式の層IDをHole、Section、Distanceへ分解する。"""
    match = LAYER_ID_PATTERN.fullmatch(str(layer_id).strip())
    if not match:
        raise ValueError(f"Invalid layer_id: {layer_id}")
    return {
        "hole": match.group(1),
        "section": match.group(2),
        "distance_cm": float(match.group(3)),
    }


def load_ai_layer_rows(path):
    """層準別AI集計から対象4分類群と層準情報を読み込む。"""
    rows = []
    with Path(path).open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if (
                row["class_name"] not in CLASS_ORDER
                or row["mapping_status"] != "matched"
                or not row["age_calBP_plot"]
            ):
                continue
            position = parse_layer_id(row["layer_id"])
            rows.append({
                "sample": row["sample"],
                "layer_id": row["layer_id"],
                "age_calBP": float(row["age_calBP_plot"]),
                "class_name": row["class_name"],
                "relative_abundance_percent": float(
                    row["relative_abundance_percent"]
                ),
                **position,
            })
    if not rows:
        raise ValueError(f"No AI layer-summary rows found: {path}")
    return rows


def load_meme_rows(path):
    """目視CSVから層準情報と分類群別相対量を計算して読み込む。"""
    observations = []
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        headers = reader.fieldnames or []
        required = {
            "Name",
            "Hole",
            "Section",
            "Distance (cm)",
            MEME_AGE_COLUMN,
            FIRST_DIATOM_COLUMN,
            LAST_DIATOM_COLUMN,
            *(column for columns in MEME_GROUP_COLUMNS.values() for column in columns),
        }
        missing = sorted(required - set(headers))
        if missing:
            raise ValueError(f"Missing Meme columns: {', '.join(missing)}")

        start_index = headers.index(FIRST_DIATOM_COLUMN)
        end_index = headers.index(LAST_DIATOM_COLUMN)
        diatom_columns = headers[start_index:end_index + 1]

        for row in reader:
            diatom_total = sum(to_float(row[column]) for column in diatom_columns)
            if diatom_total <= 0:
                continue
            for class_name, columns in MEME_GROUP_COLUMNS.items():
                grouped_count = sum(to_float(row[column]) for column in columns)
                observations.append({
                    "meme_sample": row["Name"],
                    "hole": str(row["Hole"]).strip(),
                    "section": str(row["Section"]).strip().zfill(2),
                    "distance_cm": to_float(row["Distance (cm)"]),
                    "meme_age_calBP": to_float(row[MEME_AGE_COLUMN]),
                    "class_name": class_name,
                    "grouped_valve_count": grouped_count,
                    "total_diatom_valve_count": diatom_total,
                    "relative_abundance_percent": 100 * grouped_count / diatom_total,
                })
    if not observations:
        raise ValueError(f"No Meme rows found: {path}")
    return observations


def match_ai_and_meme(ai_rows, meme_rows, tolerance_cm):
    """Hole・Section・Distanceが一致するAI層準と目視試料を一対一対応させる。"""
    ai_by_sample = defaultdict(dict)
    ai_metadata = {}
    for row in ai_rows:
        ai_by_sample[row["sample"]][row["class_name"]] = row
        ai_metadata[row["sample"]] = row

    meme_by_sample = defaultdict(dict)
    meme_metadata = {}
    for row in meme_rows:
        meme_by_sample[row["meme_sample"]][row["class_name"]] = row
        meme_metadata[row["meme_sample"]] = row

    matches = []
    used_meme_samples = set()
    for ai_sample, ai_values in sorted(ai_by_sample.items()):
        ai_meta = ai_metadata[ai_sample]
        candidates = [
            meme_meta
            for meme_sample, meme_meta in meme_metadata.items()
            if meme_sample not in used_meme_samples
            and meme_meta["hole"] == ai_meta["hole"]
            and meme_meta["section"] == ai_meta["section"]
        ]
        if not candidates:
            continue
        nearest = min(
            candidates,
            key=lambda row: abs(row["distance_cm"] - ai_meta["distance_cm"]),
        )
        distance_difference = abs(
            nearest["distance_cm"] - ai_meta["distance_cm"]
        )
        if distance_difference > tolerance_cm:
            continue

        meme_sample = nearest["meme_sample"]
        used_meme_samples.add(meme_sample)
        for class_name in CLASS_ORDER:
            ai_row = ai_values[class_name]
            meme_row = meme_by_sample[meme_sample][class_name]
            matches.append({
                "ai_sample": ai_sample,
                "ai_layer_id": ai_row["layer_id"],
                "ai_distance_cm": ai_row["distance_cm"],
                "meme_sample": meme_sample,
                "meme_distance_cm": meme_row["distance_cm"],
                "distance_difference_cm": distance_difference,
                "age_calBP": ai_row["age_calBP"],
                "meme_age_calBP": meme_row["meme_age_calBP"],
                "class_name": class_name,
                "ai_relative_abundance_percent": ai_row[
                    "relative_abundance_percent"
                ],
                "meme_grouped_valve_count": meme_row["grouped_valve_count"],
                "meme_total_diatom_valve_count": meme_row[
                    "total_diatom_valve_count"
                ],
                "meme_relative_abundance_percent": meme_row[
                    "relative_abundance_percent"
                ],
            })

    if not matches:
        raise ValueError("No AI and Meme layers matched by distance.")
    return matches


def aggregate_matches_by_age(matches):
    """同じ年代の対応ペアを分類群ごとに平均して目視描画値を作る。"""
    grouped = defaultdict(list)
    for row in matches:
        grouped[(row["age_calBP"], row["class_name"])].append(row)

    rows = []
    for (age, class_name), items in sorted(grouped.items()):
        rows.append({
            "age_calBP": age,
            "ai_samples": ";".join(item["ai_sample"] for item in items),
            "meme_samples": ";".join(item["meme_sample"] for item in items),
            "n_pairs": len(items),
            "class_name": class_name,
            "ai_relative_abundance_percent_mean": statistics.fmean(
                item["ai_relative_abundance_percent"] for item in items
            ),
            "meme_relative_abundance_percent_mean": statistics.fmean(
                item["meme_relative_abundance_percent"] for item in items
            ),
        })
    return rows


def aggregate_all_ai_by_age(ai_rows):
    """対応の有無にかかわらず全AI層準を年代・分類群ごとに平均する。"""
    grouped = defaultdict(list)
    samples_by_age = defaultdict(set)
    for row in ai_rows:
        grouped[(row["age_calBP"], row["class_name"])].append(row)
        samples_by_age[row["age_calBP"]].add(row["sample"])

    rows = []
    for (age, class_name), items in sorted(grouped.items()):
        rows.append({
            "age_calBP": age,
            "ai_samples": ";".join(sorted(samples_by_age[age])),
            "n_ai_samples": len(samples_by_age[age]),
            "class_name": class_name,
            "ai_relative_abundance_percent_mean": statistics.fmean(
                item["relative_abundance_percent"] for item in items
            ),
        })
    return rows


def add_meme_only_missing_layers(matched_age_rows, meme_rows):
    """AI画像がない層準について、存在する目視値だけを描画行へ追加する。"""
    rows = list(matched_age_rows)
    existing_keys = {
        (row["age_calBP"], row["class_name"]) for row in matched_age_rows
    }
    meme_lookup = {
        (row["meme_sample"], row["class_name"]): row for row in meme_rows
    }

    for layer in MISSING_AI_LAYERS:
        for class_name in CLASS_ORDER:
            key = (layer["age_calBP"], class_name)
            if key in existing_keys:
                continue
            meme_row = meme_lookup[(layer["meme_sample"], class_name)]
            rows.append({
                "age_calBP": layer["age_calBP"],
                "ai_samples": "",
                "meme_samples": layer["meme_sample"],
                "n_pairs": 0,
                "class_name": class_name,
                "ai_relative_abundance_percent_mean": "",
                "meme_relative_abundance_percent_mean": meme_row[
                    "relative_abundance_percent"
                ],
            })
    return sorted(rows, key=lambda row: (row["age_calBP"], row["class_name"]))


def save_rows(rows, output_path, fields):
    """指定した列順で監査用CSVを保存する。"""
    with Path(output_path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_age_bin_edges(ages):
    """隣接年代の中点を使い、AI棒グラフ用の隙間のないbin境界を作る。"""
    if len(ages) == 1:
        return [ages[0] - 0.5, ages[0] + 0.5]
    edges = [ages[0] - (ages[1] - ages[0]) / 2]
    edges.extend((left + right) / 2 for left, right in zip(ages, ages[1:]))
    edges.append(ages[-1] + (ages[-1] - ages[-2]) / 2)
    return edges


def make_comparison_plot(ai_age_rows, matched_age_rows, output_dir):
    """全AI層準と欠測binの棒に、利用可能な目視値の破線を重ねる。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    ai_ages = sorted({row["age_calBP"] for row in ai_age_rows})
    plot_ages = sorted({*ai_ages, *MISSING_AI_AGES.values()})
    meme_ages = sorted({row["age_calBP"] for row in matched_age_rows})
    ai_lookup = {
        (row["age_calBP"], row["class_name"]): row
        for row in ai_age_rows
    }
    meme_lookup = {
        (row["age_calBP"], row["class_name"]): row
        for row in matched_age_rows
    }
    bin_edges = make_age_bin_edges(plot_ages)
    bin_lowers = bin_edges[:-1]
    bin_heights = [
        upper - lower for lower, upper in zip(bin_edges, bin_edges[1:])
    ]

    first_tick = math.ceil(min(plot_ages) / 50) * 50
    last_tick = math.floor(max(plot_ages) / 50) * 50
    age_ticks = list(range(first_tick, last_tick + 1, 50))
    a_subarctica_values = [
        ai_lookup[(age, "A.subarctica")][
            "ai_relative_abundance_percent_mean"
        ]
        for age in ai_ages
    ] + [
        meme_lookup[(age, "A.subarctica")][
            "meme_relative_abundance_percent_mean"
        ]
        for age in meme_ages
    ]
    shared_xmax = max(a_subarctica_values) * 1.05

    fig, axes = plt.subplots(
        1, len(CLASS_ORDER), figsize=(13.5, 9.0),
        sharey=True, constrained_layout=True,
    )
    for axis, class_name in zip(axes, CLASS_ORDER):
        ai_values = [
            (
                ai_lookup[(age, class_name)]["ai_relative_abundance_percent_mean"]
                if (age, class_name) in ai_lookup
                else 0.0
            )
            for age in plot_ages
        ]
        meme_values = [
            meme_lookup[(age, class_name)][
                "meme_relative_abundance_percent_mean"
            ]
            for age in meme_ages
        ]
        color = TARGET_COLORS[class_name]
        axis.barh(
            bin_lowers,
            ai_values,
            height=bin_heights,
            align="edge",
            color=color,
            alpha=0.72,
            edgecolor="#8A8A8A",
            linewidth=0.5,
        )
        axis.plot(
            meme_values,
            meme_ages,
            "o--",
            color="black",
            linewidth=1.2,
            markersize=3.5,
            zorder=3,
        )
        axis.set_title(class_name, color=color, fontstyle="italic", fontsize=13)
        axis.set_xlabel("Valve relative abundance (%)")
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
    axes[-1].legend(
        handles=[
            Patch(facecolor=TARGET_COLORS["A.subarctica"], alpha=0.72, label="AI"),
            Line2D(
                [0], [0], color="black", linestyle="--", marker="o",
                linewidth=1.2, markersize=3.5, label="Meme",
            ),
        ],
        loc="upper right",
    )

    output_dir = Path(output_dir)
    stem = output_dir / "diatom_valve_relative_abundance_ai_vs_meme_by_age"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            stem.with_suffix(f".{suffix}"),
            dpi=300 if suffix == "png" else None,
        )
    plt.close(fig)


def main():
    """層準を照合し、対応表・年代集計・比較グラフを作成する。"""
    args = get_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ai_rows = load_ai_layer_rows(args.ai_layer_summary)
    meme_rows = load_meme_rows(args.meme_csv)
    matches = match_ai_and_meme(
        ai_rows, meme_rows, args.distance_tolerance_cm
    )
    matched_age_rows = aggregate_matches_by_age(matches)
    plot_meme_age_rows = add_meme_only_missing_layers(
        matched_age_rows, meme_rows
    )
    ai_age_rows = aggregate_all_ai_by_age(ai_rows)

    save_rows(
        matches,
        output_dir / "ai_meme_matched_layers.csv",
        [
            "ai_sample",
            "ai_layer_id",
            "ai_distance_cm",
            "meme_sample",
            "meme_distance_cm",
            "distance_difference_cm",
            "age_calBP",
            "meme_age_calBP",
            "class_name",
            "ai_relative_abundance_percent",
            "meme_grouped_valve_count",
            "meme_total_diatom_valve_count",
            "meme_relative_abundance_percent",
        ],
    )
    save_rows(
        matched_age_rows,
        output_dir / "ai_meme_matched_relative_abundance_by_age.csv",
        [
            "age_calBP",
            "ai_samples",
            "meme_samples",
            "n_pairs",
            "class_name",
            "ai_relative_abundance_percent_mean",
            "meme_relative_abundance_percent_mean",
        ],
    )
    make_comparison_plot(ai_age_rows, plot_meme_age_rows, output_dir)

    pair_count = len({row["ai_sample"] for row in matches})
    ai_age_count = len({row["age_calBP"] for row in ai_age_rows})
    meme_age_count = len({row["age_calBP"] for row in plot_meme_age_rows})
    print(f"Matched AI-Meme layer pairs: {pair_count}")
    print(f"AI age points shown: {ai_age_count}")
    print(f"Meme age points shown: {meme_age_count}")


if __name__ == "__main__":
    main()
