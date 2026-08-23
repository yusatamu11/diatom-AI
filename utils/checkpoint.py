"""Checkpoint helpers for class-aware Mask R-CNN training and inference."""

from collections.abc import Mapping

import torch


FORMAT_NAME = "diatom-ai-maskrcnn-checkpoint-v1"
CLASSIFIER_WEIGHT_KEY = "roi_heads.box_predictor.cls_score.weight"


def _safe_torch_load(path, map_location):
    """Load project checkpoints safely when the installed PyTorch supports it."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # PyTorch releases predating ``weights_only`` require the legacy call.
        # Only project-created checkpoints should be loaded through this path.
        return torch.load(path, map_location=map_location)


def infer_num_classes(state_dict):
    """Infer the number of model outputs from a Mask R-CNN state dictionary."""
    if CLASSIFIER_WEIGHT_KEY not in state_dict:
        raise KeyError(
            "Cannot infer the number of classes because the Mask R-CNN "
            f"classifier weight is missing: {CLASSIFIER_WEIGHT_KEY}"
        )
    return int(state_dict[CLASSIFIER_WEIGHT_KEY].shape[0])


def _normalize_category_names(categories):
    """Normalize saved category metadata to an integer ID-to-name mapping."""
    if categories is None:
        return {}
    if isinstance(categories, Mapping):
        return {int(category_id): str(name) for category_id, name in categories.items()}
    if isinstance(categories, (list, tuple)):
        result = {}
        for category in categories:
            if not isinstance(category, Mapping) or "id" not in category:
                raise ValueError("Saved categories must contain id and name fields.")
            result[int(category["id"])] = str(category["name"])
        return result
    raise ValueError("Unsupported category metadata in checkpoint.")


def make_training_checkpoint(model, category_names, epoch):
    """Build a checkpoint containing weights and the dataset class definition."""
    category_names = {
        int(category_id): str(name)
        for category_id, name in category_names.items()
    }
    return {
        "format": FORMAT_NAME,
        "model_state_dict": model.state_dict(),
        "num_classes": len(category_names) + 1,
        "categories": [
            {"id": category_id, "name": category_names[category_id]}
            for category_id in sorted(category_names)
        ],
        "epoch": int(epoch),
    }


def load_training_checkpoint(path, map_location="cpu"):
    """Load new metadata checkpoints or legacy state-dict-only checkpoints."""
    saved = _safe_torch_load(path, map_location=map_location)
    if not isinstance(saved, Mapping):
        raise TypeError(f"Checkpoint must contain a mapping: {path}")

    if "model_state_dict" in saved:
        state_dict = saved["model_state_dict"]
        category_names = _normalize_category_names(saved.get("categories"))
        inferred_num_classes = infer_num_classes(state_dict)
        num_classes = int(saved.get("num_classes", inferred_num_classes))
        if num_classes != inferred_num_classes:
            raise ValueError(
                "Checkpoint class metadata does not match its model weights: "
                f"metadata={num_classes}, weights={inferred_num_classes}"
            )
        if category_names and len(category_names) + 1 != num_classes:
            raise ValueError(
                "Checkpoint category count does not match num_classes: "
                f"categories={len(category_names)}, num_classes={num_classes}"
            )
        metadata = {
            "format": saved.get("format"),
            "epoch": saved.get("epoch"),
        }
    else:
        state_dict = saved
        num_classes = infer_num_classes(state_dict)
        category_names = {}
        metadata = {"format": "legacy-state-dict", "epoch": None}

    return state_dict, num_classes, category_names, metadata
