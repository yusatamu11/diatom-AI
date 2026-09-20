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
Use `--augmentation basic_copy_paste` to select a foreground class uniformly,
paste one CVAT-annotated training instance from that class into a training
image, and then apply the basic augmentation. Copy-Paste uses training
annotations only and is never applied to validation or test data. By default it
is applied to 30% of training images, permits at most 10% of the pasted mask to
overlap existing masks, and adds at most one instance per image.
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

# Basic augmentation plus class-uniform Copy-Paste only
python train.py ... \
  --augmentation basic_copy_paste \
  --copy_paste_probability 0.3 \
  --copy_paste_max_instances 1 \
  --copy_paste_max_overlap 0.1 \
  --copy_paste_placement_attempts 20 \
  --seed 42 \
  --output_dir runs/augmentation_basic_copy_paste

# Augmentation plus class-balanced image sampling
python train.py ... \
  --augmentation basic \
  --balanced_sampling \
  --balanced_sampling_max_weight 5 \
  --seed 42 \
  --output_dir runs/augmentation_balanced
```

Evaluate any saved metadata checkpoint without retraining. COCO AP/AR is
always evaluated without an additional confidence cutoff. To select one
operating confidence threshold, sweep only the validation set:

```bash
python evaluate_checkpoint.py \
  --image_dir /path/to/dataset/validation/images \
  --ann_file /path/to/dataset/validation/annotations.json \
  --checkpoint runs/experiment_name/best_model.pth \
  --output_dir runs/experiment_name/validation_thresholds \
  --threshold_mode sweep \
  --threshold_selection_metric macro_f1 \
  --match_iou_thresh 0.5
```

The sweep saves `threshold_metrics.csv` with class-wise and micro/macro
TP, FP, FN, precision, recall, and F1, plus `threshold_summary.json` containing
the selected bbox and segmentation thresholds. Freeze the selected validation
segmentation threshold before evaluating the test set:

```bash
python evaluate_checkpoint.py \
  --image_dir /path/to/dataset/test/images \
  --ann_file /path/to/dataset/test/annotations.json \
  --checkpoint runs/experiment_name/best_model.pth \
  --output_dir runs/experiment_name/test_evaluation \
  --threshold_mode fixed \
  --score_thresh 0.50 \
  --match_iou_thresh 0.5
```

Replace `0.50` with the segmentation threshold selected on validation. Never
retune the threshold from test results. Mask binarization remains controlled
separately by `--eval_mask_thresh`, whose default is 0.5.

## Prediction post-processing

### PyTorch prediction files (`.pt`)

The `.pt` files written by `continuous_detect.py` can be analyzed directly.
New predictions store each mask as a cropped binary `uint8` tensor rather than
a full-tile floating-point tensor. The default `--mask_thresh 0.5` is applied
when the prediction is saved, so use the same threshold chosen for the
workflow. Changing the mask threshold later requires rerunning inference.
Legacy `.pt` files containing full floating-point masks remain supported.

For compact prediction-only output, omit `--save_image`:

```bash
python continuous_detect.py \
  --image_dir /path/to/sample/images \
  --weights /path/to/best_model.pth \
  --score_thresh 0.65 \
  --mask_thresh 0.5 \
  --output_dir /path/to/sample/infer_pt
```

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
