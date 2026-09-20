import tempfile
import unittest
from pathlib import Path

import torch

from process_pt_predictions import build_morphology_rows, load_pt_tiles
from utils.compact_masks import MASK_FORMAT, encode_cropped_binary_masks


class CompactMaskTests(unittest.TestCase):
    def test_compact_and_legacy_masks_produce_identical_morphology(self):
        masks = torch.zeros((2, 1, 12, 12), dtype=torch.float32)
        masks[0, 0, 2:7, 3:9] = 0.8
        masks[1, 0, 0:4, 0:2] = 0.9
        boxes = torch.tensor([[3, 2, 9, 7], [0, 0, 2, 4]], dtype=torch.float32)
        labels = torch.tensor([1, 2])
        scores = torch.tensor([0.9, 0.8])
        common = {
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
            "class_names": ["background", "A.subarctica", "cyclostephanoids"],
        }

        crops, origins = encode_cropped_binary_masks(masks, threshold=0.5)
        compact = {
            **common,
            "mask_format": MASK_FORMAT,
            "mask_threshold": 0.5,
            "mask_crops": crops,
            "mask_origins_xy": origins,
        }
        legacy = {**common, "masks": masks}

        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            legacy_dir = root / "legacy"
            compact_dir = root / "compact"
            legacy_dir.mkdir()
            compact_dir.mkdir()
            torch.save(legacy, legacy_dir / "sample_x0002_y0003.pt")
            torch.save(compact, compact_dir / "sample_x0002_y0003.pt")

            legacy_rows = build_morphology_rows(
                load_pt_tiles(legacy_dir, 12, 2, 0.5)[(2, 3)]
            )
            compact_rows = build_morphology_rows(
                load_pt_tiles(compact_dir, 12, 2, 0.5)[(2, 3)]
            )

        self.assertEqual(legacy_rows, compact_rows)

    def test_compact_masks_reject_a_different_analysis_threshold(self):
        masks = torch.ones((1, 1, 2, 2), dtype=torch.float32)
        crops, origins = encode_cropped_binary_masks(masks, threshold=0.5)
        prediction = {
            "boxes": torch.tensor([[0, 0, 2, 2]], dtype=torch.float32),
            "labels": torch.tensor([1]),
            "scores": torch.tensor([0.9]),
            "mask_format": MASK_FORMAT,
            "mask_threshold": 0.5,
            "mask_crops": crops,
            "mask_origins_xy": origins,
        }
        with tempfile.TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "sample_x0001_y0001.pt"
            torch.save(prediction, path)
            with self.assertRaisesRegex(ValueError, "stored at threshold"):
                load_pt_tiles(path.parent, 2, 0, 0.6)


if __name__ == "__main__":
    unittest.main()
