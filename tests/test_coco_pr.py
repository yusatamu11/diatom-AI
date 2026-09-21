from types import SimpleNamespace

import numpy as np

from utils.coco_pr import extract_coco_pr_curves, save_coco_pr_outputs


def make_coco_eval():
    precision = np.full((3, 3, 2, 1, 1), -1.0, dtype=float)
    precision[:, :, 0, 0, 0] = [
        [1.0, 0.8, 0.4],
        [0.9, 0.7, 0.3],
        [0.8, 0.6, 0.2],
    ]
    precision[:, :, 1, 0, 0] = [
        [0.8, 0.5, 0.1],
        [0.7, 0.4, 0.0],
        [0.6, 0.3, -1.0],
    ]
    params = SimpleNamespace(
        areaRngLbl=["all"],
        maxDets=[100],
        recThrs=np.array([0.0, 0.5, 1.0]),
        iouThrs=np.array([0.50, 0.75, 0.95]),
        catIds=[1, 2],
    )
    return SimpleNamespace(params=params, eval={"precision": precision})


def test_extract_coco_pr_curves_uses_standard_dimensions():
    curves = extract_coco_pr_curves(
        make_coco_eval(),
        {1: "A.subarctica", 2: "cyclostephanoids"},
    )

    assert curves["recall"].tolist() == [0.0, 0.5, 1.0]
    assert np.isclose(curves["overall"]["iou50"][0], 0.9)
    assert np.isclose(curves["per_class"][1]["mean"][1], 0.7)
    assert not np.isnan(curves["per_class"][2]["iou75"][2])
    assert not np.isnan(curves["per_class"][2]["mean"][2])


def test_save_coco_pr_outputs_writes_figures_and_csv(tmp_path):
    outputs = save_coco_pr_outputs(
        make_coco_eval(),
        {1: "A.subarctica", 2: "cyclostephanoids"},
        "segm",
        tmp_path,
    )

    assert (tmp_path / "coco_pr_segm_values.csv").is_file()
    for group in ("overall", "per_class"):
        for extension in ("png", "pdf", "svg"):
            assert outputs[group][extension].endswith(f".{extension}")
            assert (tmp_path / f"coco_pr_segm_{group}.{extension}").is_file()
