# Synpipe - Pipeline Consolidation for Synapse Detection

Unified, containerized pipeline for in vivo synapse analysis.
Integrates XTC restoration, MRCNN segmentation, SynapseFilter
post-processing, and LAP-based tracking into a single automated
framework with a web-based GUI.

## Repository Structure

| Folder | Description |
|---|---|
| `synapsepipe/` | Web GUI dashboard — Flask backend + 4 pipeline stage scripts |
| `Hydra/` | Hydra-orchestrated pipeline (XTC → MRCNN → Overlay) |
| `MRCNN/` | MRCNN segmentation Docker container and model files |
| `xtc_copy/` | XTC restoration Docker container |

## Quick Start

See [`synapsepipe/README.md`](synapsepipe/README.md) for GUI setup
and usage instructions.

See [`Hydra/README.md`](Hydra/README.md) for command-line pipeline
usage.

## Pipeline Overview

1. **Registration** — ITK-Elastix volumetric alignment
2. **Restoration** — XTC 3D deep learning denoising (Docker)
3. **Segmentation** — MRCNN instance segmentation (Docker + Hydra)
4. **Tracking** — LAP-based longitudinal synapse tracking (Python)

## References

- Xu et al. (2023) — XTC cross-modality image restoration
- Chen et al. (2025) — MRCNN synapse detection
- Graves et al. (2021) — AMPAR dynamics and biological constraints
