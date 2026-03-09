# SynPipe — Synapse Detection & Tracking Pipeline

A simplified, end-to-end deep learning pipeline for **3D synapse detection, denoising, segmentation, and tracking** in microscopy volumes. The entire workflow runs in a single Jupyter notebook with an interactive slice-viewer GUI.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

SynPipe addresses a common challenge in connectomics and neuroscience imaging: going from noisy two-photon (2P) microscopy data to identified and tracked synapses, all within a unified framework. The notebook walks through four stages:

1. **Synthetic Data Generation**: Creates a realistic 3D volume with ~2,000 synapses and simulates 2P-style degradation (anisotropic blur, Poisson noise, Gaussian read noise).
2. **Image Restoration (XTC-style Denoising)**: Trains a 3D U-Net to recover clean, Airyscan-like signal from the noisy 2P input.
3. **Instance Segmentation**: Trains a second 3D U-Net (binary segmentation head) on the restored volume to predict synapse masks, then extracts individual instances via connected-component labeling.
4. **Centroid-Based Tracking**: Matches detected synapses across time-points using a KD-Tree nearest-neighbor approach on object centroids.

An **interactive ipywidgets slider** lets you scrub through Z-slices and compare the restored volume, predicted mask, and ground-truth labels side by side.

---

## Pipeline Architecture

```
Raw 2P Volume ──► 3D U-Net (Denoising) ──► Restored Volume
                                                │
                                                ▼
                                      3D U-Net (Segmentation)
                                                │
                                                ▼
                                      Connected-Component Labeling
                                                │
                                                ▼
                                      Centroid KD-Tree Tracking
```

---

## Key Components

### Synthetic Volume Generator
Generates a `(64, 128, 128)` volume with configurable voxel size `(0.096, 0.096, 1.0) µm` and spherical synapse annotations. The 2P simulation applies anisotropic Gaussian blur `σ = (1.5, 0.8, 0.8)`, Poisson photon noise, and additive Gaussian read noise.

### 3D U-Net
A lightweight encoder–decoder architecture with instance normalization and skip connections. Used for both the denoising and segmentation stages with MSE and BCE losses, respectively.

### Patch-Based Training
Volumes are divided into `(32, 64, 64)` patches with configurable stride for memory-efficient training on consumer GPUs.

### Interactive Viewer
An `ipywidgets` Z-slice slider renders three panels (Restored / Predicted Mask / GT Labels) for fast qualitative inspection.

---

## Getting Started

### Prerequisites

- Python ≥ 3.8
- CUDA-capable GPU (recommended)

### Installation

```bash
pip install numpy torch scipy matplotlib ipywidgets
```

### Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/JananiV3010/synpipe.git
   cd synpipe
   ```

2. Launch the notebook:
   ```bash
   jupyter notebook simplified_synpipe_with_slider_GUI.ipynb
   ```

3. Run all cells sequentially. Training takes ~2–5 minutes on a modern GPU.

4. Use the **Z-slice slider** at the bottom to explore results interactively.

---

## Configuration

Key parameters can be adjusted in the first cell of the notebook:

| Parameter | Default | Description |
|---|---|---|
| `VOL_SHAPE` | `(64, 128, 128)` | Volume dimensions (Z, Y, X) |
| `N_SYNAPSES` | `2000` | Number of synthetic synapses |
| `RADIUS_VOX` | `(1.0, 1.0, 1.0)` | Synapse radius in voxels |
| `VOXEL_SIZE` | `(0.096, 0.096, 1.0)` | Physical voxel size in µm |
| `PATCH_SIZE` | `(32, 64, 64)` | Training patch dimensions |
| Epochs | `20` | Training epochs per stage |

---

## Results

The pipeline produces:

- A **denoised volume** that recovers fine structure lost to 2P degradation.
- A **binary segmentation mask** of predicted synapse locations.
- **Instance labels** via connected-component analysis.
- **Track assignments** linking synapses across consecutive frames by centroid proximity.

---

## Project Structure

```
synpipe/
├── simplified_synpipe_with_slider_GUI.ipynb   # Full pipeline notebook
└── README.md
```

---

## Future Work

- Replace synthetic data with real paired Airyscan / 2P volumes
- Add domain adaptation and active learning for cross-dataset transfer
- Integrate a learned tracker (e.g., graph neural network) for multi-frame association
- Export predictions in standard formats (SWC, NIFTI)

---

## Acknowledgments

[Ongoing Project] Built as part of coursework in Biomedical Data Design at Johns Hopkins University. 
