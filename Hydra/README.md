# Hydra Pipeline

This folder contains a Hydra-configured wrapper for running the image-processing pipeline in three stages:

1. XTC preprocessing
2. MRCNN inference
3. Overlay generation

The entrypoint is [`run_hydra_pipeline.py`](/cis/home/pwu60/Phoebe/Hydra/run_hydra_pipeline.py), and the default configuration lives in [`conf/config.yaml`](/cis/home/pwu60/Phoebe/Hydra/conf/config.yaml).

## What It Does

For each `.tif` or `.tiff` file in the configured input directory, the pipeline:

- runs the XTC container on the raw image
- expects an output named `<input_stem>_REGISTERED_XTC_processed.tif`
- feeds that processed image into the MRCNN container
- expects an MRCNN output named `<xtc_stem>_0_segmentation.tif`
- runs `combine_overlay.py` on the XTC and MRCNN outputs
- writes timestamped results into separate XTC, MRCNN, and overlay output folders

## Requirements

- Python 3.10+
- Docker
- Local access to the Phoebe project assets referenced by the config:
  - XTC Docker build context
  - MRCNN Docker build context
  - MRCNN experiment/model directory
  - overlay script at `/cis/home/pwu60/qin/little_tools/combine/combine_overlay.py`

## Running The Pipeline

From the `Hydra/` directory:

Run one specific TIFF from the test data:

```bash

cd /cis/home/pwu60/Phoebe/Hydra
python run_hydra_pipeline.py \
  pipeline.input_dir=/cis/home/pwu60/Phoebe/test_data \
  'pipeline.glob_patterns=["RSc05_20190211_roi1_green.tif"]'

```

Run all TIFFs in a folder:

```bash
python run_hydra_pipeline.py \
  pipeline.input_dir=/path/to/input_images
```

With other Hydra overrides:

```bash
python run_hydra_pipeline.py \
  pipeline.input_dir=/path/to/input_images \
  docker.build_images=true \
  docker.use_gpus=true
```

## Key Config Options

### `pipeline`

- `input_dir`: directory containing input TIFF images
- `glob_patterns`: filename patterns used to collect images
- `xy_um_per_pixel`: XY pixel size passed to the XTC stage
- `z_um_per_pixel`: Z pixel size passed to the XTC stage

For the bundled `RSc05_20190211_roi1_green.tif` test volume, use:

- `xy_um_per_pixel=0.095`
- `z_um_per_pixel=1.0`

### `docker`

- `build_images`: rebuild Docker images before running
- `use_gpus`: add `--gpus` to `docker run`
- `gpus`: GPU selection passed to Docker, for example `all`
- `run_as_current_user`: runs containers as the current host UID/GID
- `container_input_dir`: mount point for input files inside containers
- `container_output_dir`: mount point for output files inside containers

### `xtc`

- `image`: Docker image tag for the XTC stage, currently `xtc:phoebe`
- `script`: script executed inside the XTC container
- `env`: environment variables passed into the XTC container

Default XTC env:

- `XTC_ONLY_ENHANCE=1`
- `XTC_SKIP_MATLAB=1`

### `mrcnn`

- `image`: Docker image tag for the MRCNN stage, currently `mrcnn:phoebe`
- `script`: inference script executed inside the MRCNN container
- `container_exp_dir`: model directory path inside the container
- `device`: compute device selection passed to the inference script
- `batch_size`: number of 3D patches processed per forward pass
- `num_workers`: DataLoader worker count for patch extraction
- `pin_memory`: enables pinned host memory for faster GPU transfers

The default Hydra config uses `batch_size=4`, `num_workers=4`, and `pin_memory=true` to avoid the very slow single-patch execution path.

### `overlay`

- `enabled`: whether to generate overlay TIFFs after MRCNN
- `script`: host-side overlay script path
- `alpha`: overlay transparency
- `threshold`: tracking mask threshold

## Outputs

Each run creates timestamped output directories such as:

```text
results/xtc_results/XTC_2026-03-24_14-30-00
results/mrcnn_results/MRCNN_2026-03-24_14-30-00
results/overlay_results/OVERLAY_2026-03-24_14-30-00
```

If the timestamp already exists, the script appends a numeric suffix.

At the end of a successful run, the script prints:

- number of processed images
- final XTC output directory
- final MRCNN output directory
- final overlay output directory
