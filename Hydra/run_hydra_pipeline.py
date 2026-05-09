
# Hydra pipeline for running XTC and MRCNN containers on input images, with optional overlay generation

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable

import hydra
from omegaconf import DictConfig

# Resolves a given path to an absolute path, handling both absolute and project-relative inputs

def resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (project_root / path).resolve()

# Collects and deduplicates image file paths from a directory based on given patterns

def collect_input_images(input_dir: Path, patterns: Iterable[str]) -> list[Path]:
    images: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for match in sorted(input_dir.glob(pattern)):
            if match.is_file():
                resolved = match.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    images.append(resolved)
    return images


# Executes a shell command and raises an error if it fails

def run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)

# Builds a Docker image from the specified context directory with a given tag

def build_image(image: str, context_dir: Path) -> None:
    run_command(["docker", "build", "-t", image, str(context_dir)])

#Constructs and runs the overlay script with specified inputs and parameters

def run_overlay(script_path: Path, source_path: Path, track_path: Path, output_path: Path, alpha: float, threshold: float) -> None:
    command = [
        "python",
        str(script_path),
        "--source",
        str(source_path),
        "--track",
        str(track_path),
        "--output",
        str(output_path),
        "--alpha",
        str(alpha),
        "--threshold",
        str(threshold),
    ]
    run_command(command)

# Builds and runs a Docker container with mounts, environment variables, GPU settings, and optional user mapping

def docker_run(
    image: str,
    command: list[str],
    mounts: list[tuple[Path, str, str]],
    env_vars: dict[str, str],
    use_gpus: bool,
    gpus: str,
    run_as_current_user: bool,
) -> None:
    docker_command = ["docker", "run", "--rm"]
    if use_gpus:
        docker_command.extend(["--gpus", gpus])
    if run_as_current_user:
        docker_command.extend(["-u", f"{os.getuid()}:{os.getgid()}"])

    for host_path, container_path, mode in mounts:
        docker_command.extend(["-v", f"{host_path}:{container_path}:{mode}"])

    for key, value in env_vars.items():
        docker_command.extend(["-e", f"{key}={value}"])

    docker_command.append(image)
    docker_command.extend(command)
    run_command(docker_command)

#Creates a uniquely named run directory by appending a timestamp and incrementing suffix if needed

def make_run_dir(root_dir: Path, prefix: str, stamp: str) -> Path:
    run_dir = root_dir / f"{prefix}_{stamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = root_dir / f"{prefix}_{stamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

# Full pipeline with paths, inputs, running XTC and MRCNN containers, and generating overlays for each image 

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = resolve_path(Path(__file__).resolve().parent, cfg.paths.phoebe_root)
    input_dir = resolve_path(project_root, cfg.pipeline.input_dir)
    xtc_context = resolve_path(project_root, cfg.paths.xtc_context)
    mrcnn_context = resolve_path(project_root, cfg.paths.mrcnn_context)
    xtc_results_root = resolve_path(project_root, cfg.paths.xtc_results_root)
    mrcnn_results_root = resolve_path(project_root, cfg.paths.mrcnn_results_root)
    overlay_results_root = resolve_path(project_root, cfg.paths.overlay_results_root)
    mrcnn_exp_dir = resolve_path(project_root, cfg.paths.mrcnn_exp_dir)
    overlay_script = resolve_path(project_root, cfg.overlay.script)

    # Validates required directories and files exist before running the pipeline

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    if not xtc_context.is_dir():
        raise SystemExit(f"XTC context directory does not exist: {xtc_context}")
    if not mrcnn_context.is_dir():
        raise SystemExit(f"MRCNN context directory does not exist: {mrcnn_context}")
    if not mrcnn_exp_dir.is_dir():
        raise SystemExit(f"MRCNN model directory does not exist: {mrcnn_exp_dir}")
    if cfg.overlay.enabled and not overlay_script.is_file():
        raise SystemExit(f"Overlay script does not exist: {overlay_script}")

    #Creates result directories if they do not already exist

    xtc_results_root.mkdir(parents=True, exist_ok=True)
    mrcnn_results_root.mkdir(parents=True, exist_ok=True)
    overlay_results_root.mkdir(parents=True, exist_ok=True)

    # Collects input images, exits if none found, and optionally builds Docker images

    images = collect_input_images(input_dir, cfg.pipeline.glob_patterns)
    if not images:
        raise SystemExit(f"No TIFF images found in {input_dir}")

    if cfg.docker.build_images:
        build_image(cfg.xtc.image, xtc_context)
        build_image(cfg.mrcnn.image, mrcnn_context)

    # Generates a timestamp and creates uniquely named result directories for XTC, MRCNN, and overlay outputs

    run_stamp = datetime.now().strftime(str(cfg.run.stamp_format))
    xtc_results_dir = make_run_dir(xtc_results_root, str(cfg.run.xtc_dir_prefix), run_stamp)
    mrcnn_results_dir = make_run_dir(mrcnn_results_root, str(cfg.run.mrcnn_dir_prefix), run_stamp)
    overlay_results_dir = make_run_dir(overlay_results_root, str(cfg.run.overlay_dir_prefix), run_stamp)

    # Configures container paths, user settings, and environment variables for XTC and MRCNN runs

    xtc_container_input_dir = str(cfg.docker.container_input_dir)
    xtc_container_output_dir = str(cfg.docker.container_output_dir)
    run_as_current_user = bool(cfg.docker.run_as_current_user)
    xtc_env = {key: str(value) for key, value in cfg.xtc.env.items()}
    xtc_env["HOME"] = str(cfg.docker.home_dir)
    xtc_env["MPLCONFIGDIR"] = str(cfg.docker.mpl_config_dir)
    mrcnn_env = {
        "HOME": str(cfg.docker.home_dir),
        "MPLCONFIGDIR": str(cfg.docker.mpl_config_dir),
    }

    # Processes each image through XTC and MRCNN containers, validates outputs, and optionally generates overlay results

    for image_path in images:
        container_input_image = f"{xtc_container_input_dir}/{image_path.name}"
        xtc_command = [
            "python",
            cfg.xtc.script,
            "--input-file",
            container_input_image,
            "--xy",
            str(cfg.pipeline.xy_um_per_pixel),
            "--z",
            str(cfg.pipeline.z_um_per_pixel),
            "--output-dir",
            xtc_container_output_dir,
        ]
        # Runs the XTC container with input/output mounts, environment variables, and optional GPU and user settings

        docker_run(
            image=cfg.xtc.image,
            command=xtc_command,
            mounts=[
                (image_path.parent, xtc_container_input_dir, "ro"),
                (xtc_context, "/app", "ro"),
                (xtc_results_dir, xtc_container_output_dir, "rw"),
            ],
            env_vars=xtc_env,
            use_gpus=bool(cfg.docker.use_gpus),
            gpus=str(cfg.docker.gpus),
            run_as_current_user=run_as_current_user,
        )

        # Constructs the expected XTC output path and exits if the file was not created

        xtc_output = xtc_results_dir / f"{image_path.stem}_REGISTERED_XTC_processed.tif"
        if not xtc_output.exists():
            raise SystemExit(f"Expected XTC output was not created: {xtc_output}")
       
        # Builds the MRCNN command using the XTC output as input and specifying output and experiment directories

        mrcnn_command = [
            "python",
            cfg.mrcnn.script,
            "--input-image",
            f"{cfg.docker.container_input_dir}/{xtc_output.name}",
            "--output-dir",
            cfg.docker.container_output_dir,
            "--exp-dir",
            cfg.mrcnn.container_exp_dir,
            "--device",
            str(cfg.mrcnn.device),
            "--batch-size",
            str(cfg.mrcnn.batch_size),
            "--num-workers",
            str(cfg.mrcnn.num_workers),
        ]
        if bool(cfg.mrcnn.pin_memory):
            mrcnn_command.append("--pin-memory")

        # Runs the MRCNN container with appropriate mounts, environment variables, and optional GPU and user settings

        docker_run(
            image=cfg.mrcnn.image,
            command=mrcnn_command,
            mounts=[
                (xtc_results_dir, cfg.docker.container_input_dir, "ro"),
                (mrcnn_context, "/app/MRCNN", "ro"),
                (mrcnn_results_dir, cfg.docker.container_output_dir, "rw"),
            ],
            env_vars=mrcnn_env,
            use_gpus=bool(cfg.docker.use_gpus),
            gpus=str(cfg.docker.gpus),
            run_as_current_user=run_as_current_user,
        )

        # Checks for segmentation output and runs the overlay step if enabled

        if cfg.overlay.enabled:
            segmentation_dir = mrcnn_results_dir / "input_output_check5"
            segmentation_path = segmentation_dir / f"{xtc_output.stem}_0_segmentation.tif"
            if not segmentation_path.exists():
                raise SystemExit(f"Expected MRCNN segmentation output was not created: {segmentation_path}")
            overlay_path = overlay_results_dir / f"{xtc_output.stem}_overlay.tif"
            run_overlay(
                script_path=overlay_script,
                source_path=xtc_output,
                track_path=segmentation_path,
                output_path=overlay_path,
                alpha=float(cfg.overlay.alpha),
                threshold=float(cfg.overlay.threshold),
            )

    # Prints a summary of processed images and output directory locations

    print(f"Processed {len(images)} image(s)")
    print(f"XTC outputs: {xtc_results_dir}")
    print(f"MRCNN outputs: {mrcnn_results_dir}")
    if cfg.overlay.enabled:
        print(f"Overlay outputs: {overlay_results_dir}")

# Runs the main pipeline when the script is executed directly

if __name__ == "__main__":
    main()
