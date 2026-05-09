"""
pipeline_stages/tracking.py
Stage 4: LAP-based Synapse Tracking
Links detected synapses across timepoints using the Linear
Assignment Problem (LAP) algorithm.
Biologically constrained by Graves et al. (2021):
  - Max displacement: 3.0 um per frame
  - Voxel size: XY=0.095 um/px, Z=1.0 um/px
Reads segmentation TIFs from:
  /cis/home/pwu60/synapsepipe_results/mrcnn_results/
Writes tracks.csv to:
  /cis/home/pwu60/synapsepipe_results/tracking/
"""
import argparse, sys, csv
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.measure import regionprops, label as sk_label

# ── Biological constants (Graves et al. 2021) ─────────────────────────────────
XY_UM_PER_PX   = 0.095   # micrometers per pixel in XY
Z_UM_PER_PX    = 1.000   # micrometers per pixel in Z
MAX_DISP_UM    = 3.0     # max synapse displacement per frame (um)
MAX_GAP_FRAMES = 2       # max frames a synapse may disappear

RESULTS_ROOT  = Path("/cis/home/pwu60/synapsepipe_results")
MRCNN_RESULTS = RESULTS_ROOT / "mrcnn_results"
TRACK_OUT     = RESULTS_ROOT / "tracking"


# ── Centroid extraction ────────────────────────────────────────────────────────

def extract_centroids(seg_vol: np.ndarray) -> list[dict]:
    """
    Extract centroids from a 3D segmentation volume.
    Returns list of dicts with keys: label, z_um, y_um, x_um, area
    Converts pixel coordinates to real-world microns.
    """
    if seg_vol.ndim == 2:
        seg_vol = seg_vol[np.newaxis, :]   # treat 2D as single Z slice

    labeled = sk_label(seg_vol > 0)
    props   = regionprops(labeled)

    centroids = []
    for p in props:
        z_px, y_px, x_px = p.centroid
        centroids.append({
            "label": p.label,
            "z_um":  round(z_px * Z_UM_PER_PX,  4),
            "y_um":  round(y_px * XY_UM_PER_PX, 4),
            "x_um":  round(x_px * XY_UM_PER_PX, 4),
            "area":  p.area,
        })
    return centroids


# ── Cost matrix ───────────────────────────────────────────────────────────────

def build_cost_matrix(cents_t: list[dict],
                      cents_t1: list[dict],
                      max_disp_um: float) -> np.ndarray:
    """
    Build an N x M cost matrix of Euclidean distances in microns.
    Distances exceeding max_disp_um are set to infinity (forbidden links).
    """
    if not cents_t or not cents_t1:
        return np.zeros((0, 0))

    coords_t  = np.array([[c["z_um"], c["y_um"], c["x_um"]] for c in cents_t])
    coords_t1 = np.array([[c["z_um"], c["y_um"], c["x_um"]] for c in cents_t1])

    cost = cdist(coords_t, coords_t1, metric="euclidean")
    cost[cost > max_disp_um] = np.inf
    return cost


# ── LAP solver ────────────────────────────────────────────────────────────────

def lap_link(cents_t: list[dict],
             cents_t1: list[dict],
             max_disp_um: float) -> list[tuple[int, int, float]]:
    """
    Solve the Linear Assignment Problem between two frames.
    Returns list of (idx_t, idx_t1, distance_um) for valid links.
    """
    cost = build_cost_matrix(cents_t, cents_t1, max_disp_um)
    if cost.size == 0:
        return []

    # Replace inf with large finite value for scipy solver
    finite_cost = np.where(np.isinf(cost), 1e9, cost)
    row_ind, col_ind = linear_sum_assignment(finite_cost)

    links = []
    for r, c in zip(row_ind, col_ind):
        dist = cost[r, c]
        if not np.isinf(dist):   # only keep valid links
            links.append((r, c, round(dist, 4)))
    return links


# ── Track builder ─────────────────────────────────────────────────────────────

def build_tracks(all_centroids: list[list[dict]],
                 max_disp_um: float,
                 max_gap: int) -> list[dict]:
    """
    Link centroids across all timepoints into tracks.
    Handles gaps up to max_gap frames.
    Returns list of track dicts.
    """
    n_frames = len(all_centroids)
    if n_frames == 0:
        return []

    # Each detected synapse starts as its own track
    track_id = 0
    # active_tracks: list of {track_id, last_frame, last_idx, detections}
    active_tracks = []
    finished_tracks = []

    # Initialise from frame 0
    for idx, c in enumerate(all_centroids[0]):
        active_tracks.append({
            "track_id":   track_id,
            "last_frame": 0,
            "last_idx":   idx,
            "detections": [{
                "frame": 0,
                **c
            }],
        })
        track_id += 1

    for frame in range(1, n_frames):
        cents_curr = all_centroids[frame]
        if not cents_curr:
            continue

        # Gather active tracks that are within gap tolerance
        linkable = [t for t in active_tracks
                    if frame - t["last_frame"] <= max_gap + 1]

        if not linkable:
            # Start new tracks for all detections in this frame
            for idx, c in enumerate(cents_curr):
                active_tracks.append({
                    "track_id":   track_id,
                    "last_frame": frame,
                    "last_idx":   idx,
                    "detections": [{"frame": frame, **c}],
                })
                track_id += 1
            continue

        # Build cost matrix: linkable tracks vs current detections
        prev_cents = [t["detections"][-1] for t in linkable]
        links = lap_link(prev_cents, cents_curr, max_disp_um)

        linked_curr = set()
        linked_prev = set()
        for r, c_idx, dist in links:
            linkable[r]["detections"].append({"frame": frame, **cents_curr[c_idx]})
            linkable[r]["last_frame"] = frame
            linkable[r]["last_idx"]   = c_idx
            linked_curr.add(c_idx)
            linked_prev.add(r)

        # Unlinked detections → new tracks
        for idx, c in enumerate(cents_curr):
            if idx not in linked_curr:
                active_tracks.append({
                    "track_id":   track_id,
                    "last_frame": frame,
                    "last_idx":   idx,
                    "detections": [{"frame": frame, **c}],
                })
                track_id += 1

        # Expire tracks that missed too many frames
        still_active = []
        for t in active_tracks:
            if frame - t["last_frame"] > max_gap:
                finished_tracks.append(t)
            else:
                still_active.append(t)
        active_tracks = still_active

    finished_tracks.extend(active_tracks)
    return finished_tracks


# ── Save tracks ───────────────────────────────────────────────────────────────

def save_tracks(tracks: list[dict], out_path: Path):
    """Write tracks to CSV."""
    with open(str(out_path), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "track_id", "frame", "z_um", "y_um", "x_um",
            "area_px", "label"
        ])
        for t in tracks:
            for d in t["detections"]:
                writer.writerow([
                    t["track_id"],
                    d["frame"],
                    d["z_um"],
                    d["y_um"],
                    d["x_um"],
                    d.get("area", 0),
                    d.get("label", 0),
                ])
    print(f"[TRK] Tracks saved → {out_path}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg-dir",     default=str(MRCNN_RESULTS),
                        help="Directory containing segmentation TIF folders")
    parser.add_argument("--output",      default=str(TRACK_OUT),
                        help="Output directory for tracks.csv")
    parser.add_argument("--max-disp",    type=float, default=MAX_DISP_UM,
                        help="Max displacement per frame in microns")
    parser.add_argument("--max-gap",     type=int,   default=MAX_GAP_FRAMES,
                        help="Max gap frames to bridge")
    parser.add_argument("--seg-pattern", default="*segmentation*.tif",
                        help="Glob pattern to find segmentation TIFs")
    args = parser.parse_args()

    import tifffile

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_dir = Path(args.seg_dir)

    print(f"[TRK] Segmentation dir : {seg_dir}",    flush=True)
    print(f"[TRK] Max displacement : {args.max_disp} µm", flush=True)
    print(f"[TRK] Max gap frames   : {args.max_gap}", flush=True)
    print("PROGRESS:5", flush=True)

    # ── Collect segmentation TIFs ─────────────────────────────────────────────
    # Look in MRCNN run folders, sorted chronologically
    seg_files = []

    # First: check our own results folder
    for run_dir in sorted(seg_dir.glob("MRCNN_*")):
        found = list(run_dir.rglob(args.seg_pattern))
        seg_files.extend(sorted(found))

    # Fallback: check Phoebe's test_data for pre-existing segmentation TIFs
    if len(seg_files) < 2:
        test_data = Path("/cis/home/pwu60/Phoebe/test_data")
        extra = sorted(test_data.glob("*segmentation*.tif"))
        seg_files.extend(extra)
        if extra:
            print(f"[TRK] Using test_data segmentation files as additional timepoints", flush=True)

    # Deduplicate
    seen = set()
    unique_segs = []
    for f in seg_files:
        if str(f) not in seen:
            seen.add(str(f))
            unique_segs.append(f)
    seg_files = unique_segs

    if len(seg_files) == 0:
        print("[ERROR] No segmentation TIFs found. Run segmentation first.", flush=True)
        sys.exit(1)

    print(f"[TRK] Found {len(seg_files)} segmentation file(s):", flush=True)
    for i, f in enumerate(seg_files):
        print(f"[TRK]   t={i}: {f.name}", flush=True)

    if len(seg_files) == 1:
        print("[WARN] Only one timepoint found. Tracking requires ≥ 2.", flush=True)
        print("[WARN] Add more segmentation runs to enable real tracking.", flush=True)
        print("[TRK] Reporting single-timepoint detections only.", flush=True)

    print("PROGRESS:15", flush=True)

    # ── Load and extract centroids per timepoint ───────────────────────────────
    all_centroids = []
    for i, seg_path in enumerate(seg_files):
        print(f"[TRK] Loading t={i}: {seg_path.name}", flush=True)
        try:
            vol = tifffile.imread(str(seg_path))
            cents = extract_centroids(vol)
            all_centroids.append(cents)
            print(f"[TRK]   → {len(cents)} objects detected", flush=True)
        except Exception as e:
            print(f"[WARN] Could not load {seg_path.name}: {e}", flush=True)
            all_centroids.append([])

        pct = 15 + int(((i + 1) / len(seg_files)) * 40)
        print(f"PROGRESS:{pct}", flush=True)

    print("PROGRESS:55", flush=True)

    # ── Run LAP tracker ────────────────────────────────────────────────────────
    print("[TRK] Running LAP tracker...", flush=True)
    tracks = build_tracks(all_centroids, args.max_disp, args.max_gap)
    print(f"[TRK] Total tracks found: {len(tracks)}", flush=True)

    # Basic stats
    multi_frame = [t for t in tracks if len(t["detections"]) > 1]
    print(f"[TRK] Tracks spanning > 1 timepoint: {len(multi_frame)}", flush=True)
    print("PROGRESS:85", flush=True)

    # ── Save results ──────────────────────────────────────────────────────────
    stamp    = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = out_dir / f"tracks_{stamp}.csv"
    save_tracks(tracks, out_path)

    # Summary stats
    n_frames   = len(seg_files)
    n_tracks   = len(tracks)
    n_linked   = len(multi_frame)
    n_t0       = len(all_centroids[0]) if all_centroids else 0

    print(f"[TRK] Timepoints processed : {n_frames}",  flush=True)
    print(f"[TRK] Total detections t=0 : {n_t0}",      flush=True)
    print(f"[TRK] Total tracks         : {n_tracks}",  flush=True)
    print(f"[TRK] Multi-frame tracks   : {n_linked}",  flush=True)
    print(f"RESULT:tracks={n_tracks},linked={n_linked},frames={n_frames}", flush=True)
    print("[TRK] Tracking complete ✓", flush=True)
    print("PROGRESS:100", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()