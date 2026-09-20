import tempfile
import unittest
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader

from continuous_detect import InferenceImageDataset, collate_image_batch


class InferenceBatchingTests(unittest.TestCase):
    def test_loader_preserves_order_and_supports_different_image_sizes(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            paths = [root / "tile_1.png", root / "tile_2.png"]
            Image.new("RGB", (12, 8), "red").save(paths[0])
            Image.new("RGB", (7, 5), "blue").save(paths[1])

            loader = DataLoader(
                InferenceImageDataset(paths),
                batch_size=2,
                num_workers=0,
                collate_fn=collate_image_batch,
            )
            loaded_paths, tensors = next(iter(loader))

        self.assertEqual(loaded_paths, paths)
        self.assertEqual(tuple(tensors[0].shape), (3, 8, 12))
        self.assertEqual(tuple(tensors[1].shape), (3, 5, 7))


if __name__ == "__main__":
    unittest.main()
