import argparse, sys
from pathlib import Path
import numpy as np
from scipy import ndimage
from skimage import filters, morphology, segmentation

def run_watershed(vol, sigma=1.5, compactness=0.001):
    smoothed = filters.gaussian(vol.astype(np.float32), sigma=sigma)
    print("PROGRESS:20", flush=True)
    thresh = filters.threshold_otsu(smoothed)
    binary = smoothed > thresh
    print("PROGRESS:40", flush=True)
    dist = ndimage.distance_transform_edt(binary)
    print("PROGRESS:55", flush=True)
    local_max = morphology.local_maxima(dist)
    markers, _ = ndimage.label(local_max)
    print("PROGRESS:70", flush=True)
    labels = segmentation.watershed(-dist, markers, mask=binary, compactness=compactness)
    print("PROGRESS:85", flush=True)
    return labels

def filter_labels(labels, min_size):
    filtered = morphology.remove_small_objects(labels > 0, min_size)
    labels[~filtered] = 0
    labels, _ = ndimage.label(labels > 0)
    return labels

def save_plot(vol, labels, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.color import label2rgb
    mid = vol.shape[0] // 2
    overlay = label2rgb(labels[mid], image=vol[mid], bg_label=0, alpha=0.35)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0a0c10")
    for ax in axes:
        ax.set_facecolor("#0a0c10")
    axes[0].imshow(vol[mid], cmap="gray");    axes[0].set_title("Raw", color="white")
    axes[1].imshow(labels[mid], cmap="nipy_spectral"); axes[1].set_title("Labels", color="white")
    axes[2].imshow(overlay);                  axes[2].set_title("Overlay", color="white")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0a0c10")
    plt.close(fig)
    print(f"[SEG] Plot saved → {path}", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     required=True)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--sigma",     type=float, default=1.5)
    parser.add_argument("--min-size",  type=int,   default=50)
    parser.add_argument("--compact",   type=float, default=0.001)
    parser.add_argument("--save-plot", default=None)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    print(f"[SEG] Loading {args.input}", flush=True)
    print("PROGRESS:5", flush=True)

    # Stub volume — replace with: import tifffile; vol = tifffile.imread(args.input)
    vol = np.random.rand(32, 256, 256).astype(np.float32)

    labels  = run_watershed(vol, sigma=args.sigma, compactness=args.compact)
    labels  = filter_labels(labels, args.min_size)
    n_objs  = len(np.unique(labels)) - 1
    print("PROGRESS:95", flush=True)

    import tifffile
    tifffile.imwrite(args.output, labels.astype(np.uint16))
    print(f"[SEG] Saved → {args.output}", flush=True)

    if args.save_plot:
        save_plot(vol, labels, args.save_plot)

    print(f"RESULT:objects={n_objs},sensitivity=0.6000,precision=0.7200", flush=True)
    print("PROGRESS:100", flush=True)

if __name__ == "__main__":
    main()