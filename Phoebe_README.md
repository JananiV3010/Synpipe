# Project Layout

This folder contains the current working copy of the Phoebe image-processing pipeline. The main workflow is:

1. XTC preprocessing
2. MRCNN segmentation
3. Overlay generation

The project is organized so the pipeline wrapper, model code, inputs, and outputs are easy to keep separate.

## Main Folders

### [`Hydra`](/cis/home/pwu60/Phoebe/Hydra)

This folder contains the Hydra-based pipeline wrapper that runs the project end-to-end.

Important files:

- [`run_hydra_pipeline.py`](/cis/home/pwu60/Phoebe/Hydra/run_hydra_pipeline.py): main pipeline entrypoint
- [`conf/config.yaml`](/cis/home/pwu60/Phoebe/Hydra/conf/config.yaml): default Hydra configuration
- [`requirements-pipeline.txt`](/cis/home/pwu60/Phoebe/Hydra/requirements-pipeline.txt): Python dependencies for the Hydra wrapper
- [`README.md`](/cis/home/pwu60/Phoebe/Hydra/README.md): detailed Hydra usage notes

What it does:

- collects TIFF inputs from a folder
- runs XTC in Docker
- passes the XTC output to MRCNN in Docker
- runs the overlay script after MRCNN
- creates dated output folders under [`results`](/cis/home/pwu60/Phoebe/results)

______________________________

### [`xtc_copy`](/cis/home/pwu60/Phoebe/xtc_copy)

This folder contains the XTC preprocessing stage and its Docker build context.

Important contents:

- [`Dockerfile`](/cis/home/pwu60/Phoebe/xtc_copy/Dockerfile): Docker image build for XTC
- [`Synapse_detection_and_tracking_FULL_NM.py`](/cis/home/pwu60/Phoebe/xtc_copy/Synapse_detection_and_tracking_FULL_NM.py): main XTC pipeline script
- [`Checkpoints_TF`](/cis/home/pwu60/Phoebe/xtc_copy/Checkpoints_TF): model checkpoint directory
- [`functional`](/cis/home/pwu60/Phoebe/xtc_copy/functional): helper functions used by the XTC stage

What it is used for:

- reads raw TIFF input
- performs registration and XTC enhancement
- writes XTC-processed TIFFs used by MRCNN
______________________________

### [`MRCNN`](/cis/home/pwu60/Phoebe/MRCNN)

This folder contains the RegRCNN-based segmentation stage and its Docker build context.

Important contents:

- [`Dockerfile`](/cis/home/pwu60/Phoebe/MRCNN/Dockerfile): Docker image build for MRCNN
- [`RegRCNN`](/cis/home/pwu60/Phoebe/MRCNN/RegRCNN): main inference and model code
- [`regrcnn_Rsc03_96_96_32_nms02_edited_GN_shifted4_all`](/cis/home/pwu60/Phoebe/MRCNN/regrcnn_Rsc03_96_96_32_nms02_edited_GN_shifted4_all): trained experiment/model directory used by Hydra
- [`README.md`](/cis/home/pwu60/Phoebe/MRCNN/README.md): model notes
- `README_container.md`: container-specific notes if present in this folder

What it is used for:

- takes XTC-processed TIFFs
- runs segmentation inference
- writes segmentation outputs into the Hydra results area

### [`test_data`](/cis/home/pwu60/Phoebe/test_data)

This folder contains example TIFF inputs for testing the pipeline.

Typical use:

- point `pipeline.input_dir` at this folder
- optionally restrict Hydra to one test file with `pipeline.glob_patterns`

Example:

```bash
cd /cis/home/pwu60/Phoebe/Hydra
python run_hydra_pipeline.py \
  pipeline.input_dir=/cis/home/pwu60/Phoebe/test_data \
  'pipeline.glob_patterns=["RSc05_20190211_roi1_green.tif"]'
```
______________________________

### [`results`](/cis/home/pwu60/Phoebe/results)

This folder is where Hydra writes run outputs.

Current layout:

- [`xtc_results`](/cis/home/pwu60/Phoebe/results/xtc_results): dated XTC output folders
- [`mrcnn_results`](/cis/home/pwu60/Phoebe/results/mrcnn_results): dated MRCNN output folders
- [`overlay_results`](/cis/home/pwu60/Phoebe/results/overlay_results): dated overlay output folders

Each Hydra run creates folders such as:

- `XTC_YYYY-MM-DD_HH-MM-SS`
- `MRCNN_YYYY-MM-DD_HH-MM-SS`
- `OVERLAY_YYYY-MM-DD_HH-MM-SS`

This keeps runs separated so outputs from different runs do not overwrite each other.

______________________________

## Recommended Starting Point

If you are running the current pipeline, start here:

1. Read [`Hydra/README.md`](/cis/home/pwu60/Phoebe/Hydra/README.md)
2. Check [`Hydra/conf/config.yaml`](/cis/home/pwu60/Phoebe/Hydra/conf/config.yaml)
3. Run the pipeline from [`Hydra`](/cis/home/pwu60/Phoebe/Hydra)

## Typical Run Command

```bash
cd /cis/home/pwu60/Phoebe/Hydra
python run_hydra_pipeline.py \
  pipeline.input_dir=/cis/home/pwu60/Phoebe/test_data \
  'pipeline.glob_patterns=["RSc05_20190211_roi1_green.tif"]'
```
