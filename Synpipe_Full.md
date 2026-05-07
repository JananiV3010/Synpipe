# Synpipe: A Containerized GUI Pipeline for Automated Synapse Detection and Analysis

**Janani Vaiyapuriappan, Phoebe Wu, Siyang Qin**  
Department of Biomedical Engineering, Whiting School of Engineering, Johns Hopkins University  
*Supervised by Dr. Adam Charles · Acknowledged: Gabrielle Coste*  
*Presented at JHU BME Design Day 2026*

---

## Overview

> **Status:** Core pipeline (restoration + segmentation + GUI) complete and validated. Presented at JHU BME Design Day, April 2026. Longitudinal synapse tracking and extended cross-dataset validation are in active development.

Synpipe is a modular, containerized computational pipeline that integrates deep learning–based image restoration and synapse segmentation into a single reproducible framework, with a live GUI dashboard for real-time monitoring on HPC infrastructure.

Modern in vivo synaptic imaging generates large, noisy 3D two-photon microscopy datasets that require multiple sequential processing steps — image restoration, segmentation, and longitudinal tracking. Existing tools like XTC (cross-modality image restoration) and MR-CNN (synapse segmentation) operate in entirely separate software environments with incompatible Python, CUDA, TensorFlow, and PyTorch dependencies, making unified, reproducible analysis difficult.

Synpipe solves this by containerizing each tool independently using Docker and orchestrating the full pipeline with Hydra, eliminating dependency conflicts and enabling one-command execution across heterogeneous computing environments — local machines, HPC clusters, and cloud.

---

## Key Results

- **SSIM ≥ 0.99** and **Pearson r ≈ 1.00** across all output types (XTC-processed images, MR-CNN segmentations, overlays), confirming that containerization preserves computational fidelity
- **Zero spatial shift** detected across all comparisons
- **Hydra pipeline SSIM = 0.9999** on XTC output — statistically identical to original workflow
- **Live GUI dashboard** running on navon A100 GPU cluster, accessible via browser at `localhost:5000`
- **Real-time log streaming** — researchers can monitor XTC patch processing progress without terminal access

---

## System Architecture

```
Raw 3D 2P In Vivo Volume
        │
        ▼
┌──────────────────┐
│  GUI Initiation  │  ← User loads volumetric data, triggers workflow, manages parameters
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────┐
│           XTC Restoration Module                 │
│  • Reimplemented 3D U-Net XTC                    │
│  • Optimized sparse-array processing             │
│  Output: Restored Volume (High SNR, Super-Res)   │
└────────┬─────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────┐
│           MR-CNN Segmentation Module             │
│  • Adapted ResNet-101 FPN with Group Norm        │
│  • Volumetric + slice-by-slice normalization     │
│  Output: Segmented Synapses (3D Instance Masks)  │
└────────┬─────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────┐
│           Tracking System (in development)       │
│  • Multi-Stage Registration                      │
│  • Structured Sparse Learning for trajectories   │
│  Output: Tracked Synapse Trajectories            │
└────────┬─────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│  GUI & Validation│  ← Interactive QC/Manual Correction, statistical report, data export
└──────────────────┘
```

Each module runs in its own isolated Docker container. Hydra orchestrates execution order, parameter propagation, and data flow between stages via mounted volumes.

---

## Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Containerization | Docker | Isolated, reproducible environments per module |
| Orchestration | Hydra (Meta AI) | YAML-driven config, automated sequential execution |
| Image Restoration | XTC (3D U-Net) | Cross-modality supervised restoration of 2P volumes |
| Segmentation | MR-CNN (ResNet-101 FPN) | 3D instance segmentation of synaptic ROIs |
| GUI | Flask (web-based) | Live dashboard on HPC, real-time log streaming |
| GPU Infrastructure | navon A100 cluster | High-throughput execution |
| Validation | SSIM, Pearson r, MSE, phase cross-correlation | Output fidelity verification |

### Dependency Environments (Before and After)

| Module | Old Python | New Python | Old Framework | New Framework | Old CUDA | New CUDA |
|---|---|---|---|---|---|---|
| MR-CNN | 3.7.5 | 3.12 | PyTorch 1.4.0 | PyTorch 2.5.1 | 10.1 | 12.1 |
| XTC | 3.7 | 3.12 | TensorFlow 2.6.0 | TensorFlow 2.18.1 | 11.2 | 12.5 |

Both environments were fully upgraded and containerized. Due to incompatible TensorFlow/PyTorch dependency stacks, XTC and MR-CNN are maintained as separate containers — a core design decision that enables modular, conflict-free deployment.

---

## Validation

Containerized and Hydra-orchestrated pipeline outputs were compared against the original non-containerized workflow using three quantitative metrics:

**Containerized Run Comparison**

| Metric | XTC Processed | Overlay | MR-CNN Segmentation |
|---|---|---|---|
| SSIM | 0.999955 | 0.993076 | 0.990540 |
| Pearson r | 1.000000 | 0.997757 | 0.997413 |
| MSE | 0.000000 | 0.000188 | 0.000548 |
| Spatial Shift | 0.000 | 0.000 | 0.000 |

**Hydra Pipeline Comparison**

| Metric | XTC Processed | Overlay | MR-CNN Segmentation |
|---|---|---|---|
| SSIM | 0.999946 | 0.991638 | 0.989608 |
| Pearson r | 0.999990 | 0.997189 | 0.997264 |
| MSE | 0.000000 | 0.000232 | 0.000568 |
| Spatial Shift | 0.000 | 0.000 | 0.000 |

All outputs maintained identical 1024×1024 dimensions with no resizing. Minor numerical differences are attributable to expected floating-point precision variability, not algorithmic discrepancy.

---

## Hydra Configuration

The full pipeline is defined in a single YAML config file, enabling parameter changes without modifying code:

```yaml
pipeline:
  input_dir: /path/to/input
  output_root: /path/to/results
  run_timestamp: 2026-04-11_09-26-50

stages:
  xtc:
    enabled: true
    image: xtc:synpipe
    xy_resolution: 0.095
    z_resolution: 1.0

  mrcnn:
    enabled: true
    image: mrcnn:synpipe

  overlay:
    enabled: true

docker:
  use_gpus: true
  gpus: all
  run_as_current_user: true
```

Variable interpolation (e.g., `${pipeline.output_root}`) ensures outputs are organized consistently across runs without manual path editing.

---

## Code Availability

The full Synpipe pipeline — including Docker container configurations, Hydra orchestration scripts, and Flask GUI dashboard — was developed on the Johns Hopkins navon HPC cluster in collaboration with Phoebe Wu and Siyang Qin. Source code is available upon request.

A standalone simplified demonstration of the core segmentation pipeline is available in this repository as `simplified_synpipe_with_slider_GUI.ipynb`.

---

## Biological Context

Synpipe was built to address a real bottleneck in neuroscience research. Synaptic connections are the fundamental basis of neural communication — their dynamic localization and density at synapses regulate synaptic strength and plasticity, mediated by AMPA receptors (AMPARs). Recent transgenic models (SEP–GluA2) enable direct in vivo visualization of these dynamics using two-photon microscopy, generating large longitudinal 3D datasets.

The primary contribution of Synpipe is not new algorithms but reproducible computational infrastructure: transforming a fragmented, error-prone workflow into a unified, portable, one-command system that any research group can deploy.

---

## Current Limitations & Future Directions

The current implementation covers image restoration, segmentation, and the GUI dashboard. The following are pending or planned:

**In active development**
- **LAP-based synapse tracking** across timepoints — linking segmented synapses longitudinally to quantify plasticity over days/weeks
- **ITK-Elastix registration integration** for improved multi-session volume alignment

**Planned**
- **Docker Compose** for single-command full-stack deployment
- Extended validation across diverse imaging datasets and species
- Export in standard neuroscience formats (SWC, NIfTI)
- Cross-dataset benchmarking of runtime performance at scale

---

---

## References

- Chen Z, et al. Automatic detection of fluorescently labeled synapses in volumetric in vivo imaging data. *bioRxiv* (2025). doi: 10.1101/2025.01.22.634278
- Xu YKT, Graves AR, Coste GI, et al. Cross-modality supervised image restoration enables nanoscale tracking of synaptic plasticity in living mice. *Nature Methods* 20, 935–944 (2023).
- Graves AR, et al. Visualizing synaptic plasticity in vivo by large-scale imaging of endogenous AMPA receptors. *eLife* 10:e66809 (2021).
- Docker Inc. Docker Documentation. https://docs.docker.com/
- Meta AI. Hydra Documentation. https://hydra.cc/

---

## License

MIT License
