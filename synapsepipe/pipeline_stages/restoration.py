"""
pipeline_stages/restoration.py
Stage 1: XTC Restoration
Runs only the XTC Docker container via a minimal Python script,
writing results to /cis/home/pwu60/synapsepipe_results/xtc_results/
"""
import argparse, sys, subprocess, time
from pathlib import Path
from datetime import datetime

PHOEBE_ROOT  = Path("/cis/home/pwu60/Phoebe")
XTC_CONTEXT  = PHOEBE_ROOT / "xtc_copy"
RESULTS_ROOT = Path("/cis/home/pwu60/synapsepipe_results/xtc_results")
PYTHON       = "/cis/home/pwu60/my_env/bin/python"

def docker_run(image, command, mounts, env_vars, use_gpus=True):
    cmd = ["docker", "run", "--rm"]
    if use_gpus:
        cmd += ["--gpus", "all"]
    import os
    cmd += ["-u", f"{os.getuid()}:{os.getgid()}"]
    for host_path, container_path, mode in mounts:
        cmd += ["-v", f"{host_path}:{container_path}:{mode}"]
    for k, v in env_vars.items():
        cmd += ["-e", f"{k}={v}"]
    cmd.append(image)
    cmd.extend(command)
    return cmd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="/cis/home/pwu60/Phoebe/test_data/RSc05_20190211_roi1_green.tif")
    parser.add_argument("--output", default=str(RESULTS_ROOT))
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir    = RESULTS_ROOT / f"XTC_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[XTC] Input  : {input_path}", flush=True)
    print(f"[XTC] Output : {out_dir}", flush=True)
    print("PROGRESS:5", flush=True)

    xtc_cmd = [
        "python",
        "Synapse_detection_and_tracking_FULL_NM.py",
        "--input-file", f"/pipeline/input/{input_path.name}",
        "--xy",  "0.095",
        "--z",   "1.0",
        "--output-dir", "/pipeline/output",
    ]

    mounts = [
        (str(input_path.parent), "/pipeline/input",  "ro"),
        (str(XTC_CONTEXT),       "/app",              "ro"),
        (str(out_dir),           "/pipeline/output",  "rw"),
    ]

    env = {
        "XTC_ONLY_ENHANCE": "1",
        "XTC_SKIP_MATLAB":  "1",
        "HOME":             "/tmp",
        "MPLCONFIGDIR":     "/tmp/mpl",
    }

    print("[XTC] Starting XTC Docker container...", flush=True)
    print("PROGRESS:10", flush=True)

    full_cmd = docker_run("xtc:phoebe", xtc_cmd, mounts, env)
    print(f"[XTC] Running: {' '.join(full_cmd)}", flush=True)

    try:
        proc = subprocess.Popen(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        progress = 10
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            print(f"[XTC] {line}", flush=True)
            if "patch" in line.lower() and "/" in line:
                try:
                    parts = line.split("|")
                    for p in parts:
                        if "/" in p and "patch" in p.lower():
                            nums = p.strip().split("/")
                            current = int(''.join(filter(str.isdigit, nums[0][-6:])))
                            total   = int(''.join(filter(str.isdigit, nums[1][:6])))
                            if total > 0:
                                progress = 10 + int((current / total) * 85)
                                print(f"PROGRESS:{min(progress, 95)}", flush=True)
                except:
                    pass

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"XTC exited with code {proc.returncode}")

    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        sys.exit(1)

    # Check output
    outputs = list(out_dir.glob("*.tif"))
    if outputs:
        print(f"[XTC] Output saved: {outputs[0]}", flush=True)
        print(f"XTC_OUTPUT:{outputs[0]}", flush=True)
    else:
        print("[WARN] No output TIF found", flush=True)

    print("[XTC] Restoration complete ✓", flush=True)
    print("PROGRESS:100", flush=True)
    sys.exit(0)

if __name__ == "__main__":
    main()