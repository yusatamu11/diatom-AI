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

## Prediction post-processing

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
