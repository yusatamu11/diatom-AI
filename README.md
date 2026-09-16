# Diatom AI

Deep-learning framework for automated detection and segmentation of fossil diatoms from microscope slide images.

## Features

- Mask R-CNN based segmentation
- Continuous slide inference
- Whole-slide image support
- Automatic counting of fossil diatoms

## Project structure

...

## Requirements

For tar.zst support:

```bash
pip install zstandard
```

## Training

`train.py` reads the foreground category IDs and names from the training COCO
annotation JSON. Category IDs must be contiguous and start at 1 because 0 is
reserved for the Mask R-CNN background class. The validation JSON must contain
the same category definition.

```bash
python train.py \
  --image_dir /path/to/dataset/train/images \
  --ann_file /path/to/dataset/train/annotations.json \
  --val_image_dir /path/to/dataset/validation/images \
  --val_ann_file /path/to/dataset/validation/annotations.json \
  --epochs 100 \
  --batch_size 2 \
  --augmentation none \
  --seed 42 \
  --early_stopping_min_epochs 20 \
  --early_stopping_patience 15 \
  --early_stopping_min_delta 0.001 \
  --output_dir runs/experiment_name
```

Each checkpoint stores the model weights, number of classes, category names,
and epoch. `detect.py` and `continuous_detect.py` read these values
automatically. Legacy checkpoints containing only a model state dictionary are
also accepted; their output size is inferred from the classifier weights, but
they do not contain class names.

With validation enabled, training also writes `class_metrics.csv` with bbox and
segmentation AP/AP50/AP75/AR100 for every foreground class, saves a class-wise
AP plot and a screenshot-friendly metrics table PNG for each epoch, and updates
`best_model.pth` whenever validation segmentation AP improves.

`metrics.csv` stores the total train/validation loss as well as image-weighted
epoch averages for the five Mask R-CNN loss components: classifier, box
regression, mask, RPN objectness, and RPN box regression.

Early stopping monitors the overall validation segmentation AP. By default,
training runs for at least 20 epochs and stops after 15 subsequent epochs
without an AP improvement greater than 0.001. Set
`--early_stopping_patience 0` to disable early stopping. The final completed
epoch is still checkpointed and logged before training stops.

Training augmentation is disabled by default. Use `--augmentation basic` to
apply synchronized horizontal/vertical flips, 90-degree rotations, and mild
brightness/contrast changes to training images and masks only. Use
`--augmentation strong` for the same geometry with stronger brightness and
contrast changes plus mild gamma, Gaussian blur, and Gaussian noise variation.
Validation and test images are never augmented. Use `--balanced_sampling` to
sample images containing rare classes more frequently; weights use a capped
inverse-square-root frequency ratio and default to a maximum of 5.0. Every run
saves all CLI settings to `training_config.json`.

For controlled comparisons, keep the split, seed, learning rate, and early-
stopping settings fixed:

```bash
# Augmentation only
python train.py ... \
  --augmentation basic \
  --seed 42 \
  --output_dir runs/augmentation_basic

# Stronger augmentation only
python train.py ... \
  --augmentation strong \
  --seed 42 \
  --output_dir runs/augmentation_strong

# Augmentation plus class-balanced image sampling
python train.py ... \
  --augmentation basic \
  --balanced_sampling \
  --balanced_sampling_max_weight 5 \
  --seed 42 \
  --output_dir runs/augmentation_balanced
```

Evaluate any saved metadata checkpoint without retraining:

```bash
python evaluate_checkpoint.py \
  --image_dir /path/to/dataset/validation/images \
  --ann_file /path/to/dataset/validation/annotations.json \
  --checkpoint runs/experiment_name/maskrcnn_epoch_1.pth \
  --output_dir runs/experiment_name/evaluation_epoch_1
```

## Prediction post-processing

### PyTorch prediction files (`.pt`)

The `.pt` files written by `continuous_detect.py` can be analyzed directly.
For one sample, merge overlapping tiles, remove duplicate detections, measure
the original binary masks, and write an instance-level CSV:

```bash
python process_pt_predictions.py \
  --prediction_dir /path/to/A09-175/inference \
  --csv_output /path/to/results/A09-175_morphology.csv \
  --class_map /path/to/coco_annotations.json
```

`--class_map` accepts either a COCO annotation JSON containing `categories`, a
JSON object such as `{"1": "A.subarctica"}`, or a JSON list whose positions
are class IDs. If it is omitted, fallback names such as `class_1` are used.
Use `--merged_output results/A09-175_merged.pt` when the duplicate-removed
predictions should also be retained.

To process multiple samples, place one prediction directory per sample under
an input directory. `.tar.zst` prediction archives are also accepted:

```text
prediction_samples/
├── A09-175/
│   ├── A09-175_x0001_y0001.pt
│   └── A09-175_x0002_y0001.pt
└── A09-176/
    ├── A09-176_x0001_y0001.pt
    └── A09-176_x0002_y0001.pt
```

```bash
python batch_pt_identification_morphology.py \
  --input_dir /path/to/prediction_samples \
  --output_dir /path/to/results \
  --class_map /path/to/coco_annotations.json
```

The batch command calls the one-sample processor for every sample, preserves
per-sample morphology CSVs, and creates `morphology_all_samples.csv`,
`identification_summary.csv`, and identification plots. Add `--save_merged`
to retain a duplicate-removed `.pt` file for every sample.

### JSON prediction files

Process every JSON prediction archive in a directory, remove duplicate
detections between overlapping tiles, calculate morphology with
`skimage.measure.regionprops`, and create identification summaries and plots:

```bash
python batch_identification_morphology.py \
  --input_dir /path/to/prediction_archives \
  --output_dir /path/to/results
```

Major- and minor-axis lengths are calculated for `A.subarctica`,
`A.ambigua`, and `cyclostephanoids`. The plots use pink, blue, yellow, and
purple for `A.subarctica`, `A.ambigua`, `Fragilariophycea`, and
`cyclostephanoids`, respectively.
