"""
pipeline_stages/registration.py
Stage 1 (Pre-processing): ITK-Elastix Registration
Aligns moving image to fixed reference image before XTC restoration.
Writes output to /cis/home/pwu60/synapsepipe_results/
"""
import argparse, sys, subprocess
from pathlib import Path
from datetime import datetime

RESULTS_ROOT = Path("/cis/home/pwu60/synapsepipe_results")
PYTHON       = "/cis/home/pwu60/my_env/bin/python"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed",     default="/cis/home/pwu60/Phoebe/test_data/RSc03_20181211_4x__roi3_Green.tif")
    parser.add_argument("--moving",    default="/cis/home/pwu60/Phoebe/test_data/RSc05_20190211_roi1_green.tif")
    parser.add_argument("--transform", default="BSpline")
    parser.add_argument("--out-dir",   default=str(RESULTS_ROOT / "registration"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed_path  = Path(args.fixed)
    moving_path = Path(args.moving)

    print(f"[REG] Fixed   : {fixed_path}",    flush=True)
    print(f"[REG] Moving  : {moving_path}",   flush=True)
    print(f"[REG] Transform: {args.transform}", flush=True)
    print(f"[REG] Output  : {out_dir}",        flush=True)
    print("PROGRESS:5", flush=True)

    # ── Check ITK-Elastix is available ──────────────────────────────────────
    try:
        import itk
        print("[REG] ITK-Elastix found ✓", flush=True)
    except ImportError:
        print("[REG] ITK-Elastix not installed in my_env.", flush=True)
        print("[REG] Install with: pip install itk-elastix", flush=True)
        print("[REG] Skipping registration — using moving image as-is.", flush=True)
        print("PROGRESS:100", flush=True)
        sys.exit(0)

    print("PROGRESS:10", flush=True)

    # ── Real ITK-Elastix registration ────────────────────────────────────────
    try:
        import itk
        import numpy as np
        import tifffile

        print("[REG] Loading images...", flush=True)
        fixed_vol  = tifffile.imread(str(fixed_path)).astype(np.float32)
        moving_vol = tifffile.imread(str(moving_path)).astype(np.float32)
        print("PROGRESS:20", flush=True)

        fixed_itk  = itk.GetImageFromArray(fixed_vol)
        moving_itk = itk.GetImageFromArray(moving_vol)

        print(f"[REG] Fixed shape  : {fixed_vol.shape}", flush=True)
        print(f"[REG] Moving shape : {moving_vol.shape}", flush=True)
        print("PROGRESS:30", flush=True)

        # Parameter object
        parameter_object = itk.ParameterObject.New()
        parameter_map = parameter_object.GetDefaultParameterMap(args.transform.lower())
        parameter_object.AddParameterMap(parameter_map)
        print("PROGRESS:40", flush=True)

        print(f"[REG] Running {args.transform} registration...", flush=True)
        registered, result_transform = itk.elastix_registration_method(
            fixed_itk, moving_itk,
            parameter_object=parameter_object,
            log_to_console=True
        )
        print("PROGRESS:80", flush=True)

        # Save result
        result_array = itk.GetArrayFromImage(registered)
        out_path = out_dir / f"{moving_path.stem}_registered.tif"
        tifffile.imwrite(str(out_path), result_array)
        print(f"[REG] Registered output saved → {out_path}", flush=True)

        # Save transform parameters
        transform_path = out_dir / "transform_parameters.txt"
        result_transform.WriteParameterFile(
            result_transform.GetParameterMap(0),
            str(transform_path)
        )
        print(f"[REG] Transform saved → {transform_path}", flush=True)
        print("PROGRESS:95", flush=True)

    except Exception as e:
        print(f"[ERROR] Registration failed: {e}", flush=True)
        sys.exit(1)

    print("[REG] Registration complete ✓", flush=True)
    print("PROGRESS:100", flush=True)
    sys.exit(0)

if __name__ == "__main__":
    main()