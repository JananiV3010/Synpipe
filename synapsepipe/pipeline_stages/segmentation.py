"""
pipeline_stages/segmentation.py
Stage 2: MRCNN Segmentation
Takes XTC output and runs MRCNN Docker container.
Writes to /cis/home/pwu60/synapsepipe_results/mrcnn_results/
"""
import argparse, sys, subprocess, shutil
from pathlib import Path
from datetime import datetime
import os

PHOEBE_ROOT   = Path("/cis/home/pwu60/Phoebe")
MRCNN_CONTEXT = PHOEBE_ROOT / "MRCNN"
MRCNN_EXP_DIR = MRCNN_CONTEXT / "regrcnn_Rsc03_96_96_32_nms02_edited_GN_shifted4_all"
RESULTS_ROOT  = Path("/cis/home/pwu60/synapsepipe_results")
XTC_RESULTS   = RESULTS_ROOT / "xtc_results"
MRCNN_RESULTS = RESULTS_ROOT / "mrcnn_results"

def find_latest_xtc_output(input_stem: str) -> Path | None:
    """Find the most recent XTC output for a given input file stem."""
    candidates = []
    for run_dir in sorted(XTC_RESULTS.glob("XTC_*"), reverse=True):
        for tif in run_dir.glob(f"*{input_stem}*.tif"):
            candidates.append(tif)
        if candidates:
            break
    # Also check Phoebe's test_data for pre-existing XTC outputs
    if not candidates:
        test_data = PHOEBE_ROOT / "test_data"
        for tif in test_data.glob(f"*{input_stem}*XTC*.tif"):
            candidates.append(tif)
    return candidates[0] if candidates else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-size", type=int, default=20)
    parser.add_argument("--max-size", type=int, default=150)
    parser.add_argument("--input",     default="/cis/home/pwu60/Phoebe/test_data/RSc05_20190211_roi1_green.tif")
    parser.add_argument("--output",    default=str(MRCNN_RESULTS))
    parser.add_argument("--xtc-input", default=None,
                        help="Path to XTC output TIF. If not given, auto-detects latest.")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir    = MRCNN_RESULTS / f"MRCNN_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[MRCNN] Input      : {input_path}", flush=True)
    print(f"[MRCNN] Output dir : {out_dir}", flush=True)
    print("PROGRESS:5", flush=True)

    # ── Find XTC input ────────────────────────────────────────────────────────
    if args.xtc_input:
        xtc_path = Path(args.xtc_input)
    else:
        xtc_path = find_latest_xtc_output(input_path.stem)

    if xtc_path and xtc_path.exists():
        print(f"[MRCNN] XTC input  : {xtc_path}", flush=True)
    else:
        print("[MRCNN] No XTC output found. Running Hydra full pipeline instead.", flush=True)
        _run_full_hydra(input_path, out_dir)
        return

    print("PROGRESS:10", flush=True)

    # ── Run MRCNN Docker ──────────────────────────────────────────────────────
    xtc_dir = xtc_path.parent

    mrcnn_cmd = [
        "python",
        "RegRCNN/inference_sparse_parallel_qin_copy.py",
        "--input-image",  f"/pipeline/input/{xtc_path.name}",
        "--output-dir",   "/pipeline/output",
        "--exp-dir",      "/app/MRCNN/regrcnn_Rsc03_96_96_32_nms02_edited_GN_shifted4_all",
    ]

    docker_cmd = ["docker", "run", "--rm", "--gpus", "all",
                  "-u", f"{os.getuid()}:{os.getgid()}",
                  "-v", f"{xtc_dir}:/pipeline/input:ro",
                  "-v", f"{MRCNN_CONTEXT}:/app/MRCNN:ro",
                  "-v", f"{out_dir}:/pipeline/output:rw",
                  "-e", "HOME=/tmp",
                  "-e", "MPLCONFIGDIR=/tmp/mpl",
                  "mrcnn:phoebe"] + mrcnn_cmd

    print(f"[MRCNN] Running Docker...", flush=True)
    print("PROGRESS:15", flush=True)

    try:
        proc = subprocess.Popen(
            docker_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        progress = 15
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            print(f"[MRCNN] {line}", flush=True)
            if "patch" in line.lower() and "/" in line:
                try:
                    for part in line.split("|"):
                        if "/" in part:
                            nums = part.strip().split("/")
                            cur  = int(''.join(filter(str.isdigit, nums[0][-6:])))
                            tot  = int(''.join(filter(str.isdigit, nums[1][:6])))
                            if tot > 0:
                                progress = 15 + int((cur / tot) * 75)
                                print(f"PROGRESS:{min(progress, 90)}", flush=True)
                except:
                    pass

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"MRCNN exited with code {proc.returncode}")

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        sys.exit(1)

    print("PROGRESS:90", flush=True)

    # ── Find segmentation output ──────────────────────────────────────────────
    seg_files = list(out_dir.rglob("*segmentation*.tif"))
    if seg_files:
        seg_file = seg_files[0]
        print(f"[MRCNN] Segmentation output: {seg_file}", flush=True)
        print(f"SEG_OUTPUT:{seg_file}", flush=True)
    else:
        print("[WARN] No segmentation TIF found in output", flush=True)

    print("[MRCNN] Segmentation complete ✓", flush=True)
    print("PROGRESS:100", flush=True)
    sys.exit(0)


def _run_full_hydra(input_path: Path, out_dir: Path):
    """Fallback: run the full Hydra pipeline (XTC + MRCNN together)."""
    import sys as _sys
    _sys.path.insert(0, str(PHOEBE_ROOT / "Hydra"))

    hydra_dir = PHOEBE_ROOT / "Hydra"
    cmd = [
        "/cis/home/pwu60/my_env/bin/python",
        "run_hydra_pipeline.py",
        f"pipeline.input_dir={input_path.parent}",
        f"pipeline.glob_patterns=[\"{input_path.name}\"]",
        f"paths.xtc_results_root={RESULTS_ROOT}/xtc_results",
        f"paths.mrcnn_results_root={RESULTS_ROOT}/mrcnn_results",
        f"paths.overlay_results_root={RESULTS_ROOT}/overlay_results",
        "overlay.enabled=false",
    ]

    print(f"[PIPE] Running full Hydra pipeline...", flush=True)
    print("PROGRESS:10", flush=True)

    proc = subprocess.Popen(
        cmd, cwd=str(hydra_dir),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    progress = 10
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        print(f"[PIPE] {line}", flush=True)
        if "xtc" in line.lower() and "output" in line.lower():
            progress = 60
        elif "mrcnn" in line.lower() and "output" in line.lower():
            progress = 85
        print(f"PROGRESS:{progress}", flush=True)

    proc.wait()
    if proc.returncode != 0:
        print(f"[ERROR] Hydra pipeline failed", flush=True)
        _sys.exit(1)

    print("[PIPE] Full pipeline complete ✓", flush=True)
    print("PROGRESS:100", flush=True)
    _sys.exit(0)


if __name__ == "__main__":
    main()