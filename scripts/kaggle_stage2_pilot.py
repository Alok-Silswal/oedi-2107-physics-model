"""Prepare and run separate finite Kaggle GPU pilot/training jobs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
import hashlib
import venv

JOB_MODE = "pilot"
ACCEPTED_PILOT_SHA256 = None
GPU_RUNTIME_PINS = []


def package(destination):
    import pandas as pd
    import pyarrow.parquet as pq
    from forecast_stage2_weather import DATA, SPLIT, training_blocks
    from oedi2107.real_data import discover_channels
    from train_stage2_neural import prepare_data

    repository = Path.cwd().resolve()
    destination = Path(destination).resolve()
    if not destination.is_relative_to(repository/"outputs"):
        raise ValueError("Package must stay under the repository's ignored outputs directory")
    destination.mkdir(parents=True, exist_ok=False)
    dataset = destination/"dataset"
    dataset.mkdir()
    with tempfile.TemporaryDirectory(prefix=".kaggle-package-", dir=destination) as temporary:
        staged = Path(temporary)
        previous_package = repository/"outputs/2107_kaggle_pilot/dataset/project.zip"
        reuse = previous_package.is_file()
        if reuse:
            extract(previous_package, staged)
        scripts = ["train_stage2_neural", "train_stage2_quantum_gpu", "compare_stage2_correction", "forecast_stage2_weather",
                   "audit_stage2_readiness", "calculate_stage2_physics", "benchmark_short_term_forecasting"]
        modules = ["__init__", "config", "pipeline", "dc_losses", "inverter", "plant", "array",
                   "module", "thermal", "inputs", "real_data", "research", "validation/__init__",
                   "validation/metrics", "validation/tables"]
        files = [Path("pyproject.toml"), Path("requirements-lock.txt")]
        files += [Path(f"scripts/{name}.py") for name in scripts]
        files += [Path(f"src/oedi2107/{name}.py") for name in modules]
        steps = [name for name, _, _ in training_blocks()]+["validation"]
        weather = Path("outputs/2107_stage2_weather")
        for step in steps:
            files += [weather/f"physics_{step}.parquet", weather/f"physics_{step}_settings.json", weather/f"{step}_settings.json"]
        files += [Path("outputs/2107_stage2_correction/hpo/settings.json"), Path("outputs/2107_stage2_correction/ridge.joblib")]
        for source in files:
            target = staged/source
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        reference = json.loads(Path("outputs/2107_stage2_neural/settings.json").read_text())
        for name, expected in reference["source_hashes"].items():
            source = staged/"scripts"/name.replace("\\", "/").rsplit("/",1)[-1]
            if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
                raise ValueError("Completed classical source fingerprint changed")
        (staged/"classical_reference_settings.json").write_text(json.dumps(reference, indent=2))
        (staged/"neural-constraints.txt").write_text("\n".join(
            f"{name}=={reference['versions'][name]}" for name in ("torch", "pennylane", "pennylane-lightning"))+"\n")
        first = min(pd.read_parquet(weather/f"physics_{step}.parquet").index.min() for step in steps)
        lower = str(first-pd.Timedelta(minutes=15))
        filters = [("measured_on", ">=", lower), ("measured_on", "<", SPLIT["test"])]
        if not reuse:
            mapping = discover_channels(pq.ParquetFile(DATA/"2107_electrical_clean.parquet").schema_arrow.names)
            electrical = ["measured_on", *mapping.dc_current, *mapping.dc_voltage, *mapping.ac_power]
            # Preserve exported values and schema channels required by discover_channels.
            for name, columns in [("electrical_clean", electrical), ("electrical_qc_events", ["measured_on"]),
                                  ("irradiance_clean", None), ("irradiance_qc_events", ["measured_on"])]:
                source = DATA/f"2107_{name}.parquet"
                target = staged/source
                target.parent.mkdir(parents=True, exist_ok=True)
                pd.read_parquet(source, columns=columns, filters=filters).to_parquet(target, index=False)
        previous = Path.cwd()
        try:
            os.chdir(staged)
            x, _, day, _, _ = prepare_data()
            print("Packaged eligibility:", len(x["training"]), len(x["validation"]), int(day.sum()), flush=True)
        finally:
            os.chdir(previous)
        with zipfile.ZipFile(dataset/"project.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(staged.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(staged).as_posix())
    (dataset/"dataset-metadata.json").write_text(json.dumps(dict(
        id="anokgaming/pv2107-quantum-inputs", title="PV2107 Quantum Inputs",
        description="Private project pilot inputs. Original data/code licensing remains applicable; no new redistribution license is granted.",
        licenses=[dict(name="other")]), indent=2))
    for mode, identifier, title in [("pilot", "pv2107-quantum-correction", "PV2107 Quantum Correction"),
                                    ("training", "pv2107-quantum-training", "PV2107 Quantum Training")]:
        kernel = destination/mode
        kernel.mkdir()
        text = Path(__file__).read_text().replace('JOB_MODE = "pilot"', f'JOB_MODE = "{mode}"', 1)
        (kernel/"kaggle_stage2_pilot.py").write_text(text)
        (kernel/"kernel-metadata.json").write_text(json.dumps(dict(
            id=f"anokgaming/{identifier}", title=title,
            code_file="kaggle_stage2_pilot.py", language="python", kernel_type="script", is_private=True,
            enable_gpu=True, machine_shape="NvidiaTeslaT4", enable_internet=True,
            dataset_sources=["anokgaming/pv2107-quantum-inputs"], competition_sources=[],
            kernel_sources=["anokgaming/pv2107-quantum-correction"] if mode=="training" else [], model_sources=[]), indent=2))
    print("Local package:", destination, "measurement support:", lower, "to", SPLIT["test"], "(exclusive)", flush=True)


def extract(archive_path, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            target = (destination/info.filename).resolve()
            if not target.is_relative_to(destination) or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe input archive member")
            if target.is_file():
                raise ValueError("Archive would overwrite an existing file")
        archive.extractall(destination)


def accept_pilot(path, destination):
    saved = json.loads(Path(path).read_text())
    if saved.get("status")!="complete" or not saved.get("correctness", {}).get("passed") or not saved.get("gpu", {}).get("verified"):
        raise ValueError("Cannot accept an unsuccessful/non-GPU pilot")
    with zipfile.ZipFile(destination/"dataset/project.zip") as archive:
        reference = json.loads(archive.read("classical_reference_settings.json"))
        for name, expected in saved["specification"]["source_hashes"].items():
            source = "scripts/"+name.replace("\\", "/").rsplit("/", 1)[-1]
            if hashlib.sha256(archive.read(source)).hexdigest()!=expected:
                raise ValueError("Pilot code differs from the prepared package")
        if saved["specification"]["frame_fingerprints"]!=reference["frame_fingerprints"]:
            raise ValueError("Pilot data differs from the prepared package")
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    pins = [f"{name}=={version}" for name, version in saved["specification"]["versions"].items()
            if name.startswith(("nvidia-", "custatevec-")) or name=="pennylane-lightning-gpu"]
    text = Path(__file__).read_text().replace('JOB_MODE = "pilot"', 'JOB_MODE = "training"', 1)
    text = text.replace("ACCEPTED_PILOT_SHA256 = None", f"ACCEPTED_PILOT_SHA256 = {digest!r}", 1)
    text = text.replace("GPU_RUNTIME_PINS = []", f"GPU_RUNTIME_PINS = {pins!r}", 1)
    (destination/"training/kaggle_stage2_pilot.py").write_text(text)
    print("Accepted GPU pilot SHA256:", digest, "; training launcher prepared locally; no submission")


def recovery(path, destination):
    source = Path(path).resolve()
    if source.name!="2107_stage2_quantum_gpu" or not (source/"settings.json").is_file():
        raise ValueError("Supply the downloaded 2107_stage2_quantum_gpu result directory")
    if not (source/"gpu_pilot.json").is_file():
        raise ValueError("Recovery requires the originally accepted GPU pilot artifact")
    settings = json.loads((source/"settings.json").read_text())
    pilot = json.loads((source/"gpu_pilot.json").read_text())
    if settings.get("settings", {}).get("backend")!="lightning.gpu" or settings!=pilot.get("specification"):
        raise ValueError("Recovery settings differ from the GPU pilot")
    with zipfile.ZipFile(destination/"dataset/resume.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        allowed = {"settings.json", "preprocessing.joblib", "gpu_pilot.json", "last.pt", "best.pt",
                   "history.csv", "metrics.json", "validation_predictions.parquet"}
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.name in allowed:
                archive.write(path, path.relative_to(source.parent).as_posix())
    metadata_path = destination/"training/kernel-metadata.json"
    saved = json.loads(metadata_path.read_text())
    saved["kernel_sources"] = []
    metadata_path.write_text(json.dumps(saved, indent=2))
    print("Recovery archive prepared locally; no checkpoint deserialization or remote action")


def run(project, work, stage):
    project, work, stage = Path(project).resolve(), Path(work).resolve(), Path(stage).resolve()
    status = work/"quantum_gpu_batch_status.json"
    log = work/"quantum_gpu_batch.log"
    result = dict(workload=JOB_MODE, status="failed", exit_code=1)
    try:
        # Refuse reuse rather than overwrite an earlier job's outputs/checkpoints.
        stage.mkdir(exist_ok=False)
        for name in ("scripts", "src", "data"):
            source = project/name
            if not source.is_dir():
                raise ValueError(f"Missing project input directory: {source}")
            (stage/name).symlink_to(source, target_is_directory=True)
        (stage/"outputs").mkdir()
        for name in ("2107_stage2_weather", "2107_stage2_correction"):
            source = project/"outputs"/name
            if not source.is_dir():
                raise ValueError(f"Missing saved input artifacts: {source}")
            (stage/"outputs"/name).symlink_to(source, target_is_directory=True)
        result_directory = work/"2107_stage2_quantum_gpu"
        result_directory.mkdir(exist_ok=False)
        (stage/"outputs/2107_stage2_quantum_gpu").symlink_to(result_directory, target_is_directory=True)
        if JOB_MODE=="training":
            if ACCEPTED_PILOT_SHA256 is None:
                raise ValueError("Training is gated: download/review the GPU pilot and run --accept-gpu-pilot locally first")
            recovery_archives = list(Path("/kaggle/input").rglob("resume.zip"))
            if len(recovery_archives)>1:
                raise ValueError("Ambiguous recovery archive")
            if recovery_archives:
                extract(recovery_archives[0], work)
            target = result_directory/"gpu_pilot.json"
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=ACCEPTED_PILOT_SHA256:
                raise ValueError("Recovery pilot differs from the accepted pilot")
            if not target.exists():
                pilots = [p for p in Path("/kaggle/input").rglob("gpu_pilot.json")
                          if hashlib.sha256(p.read_bytes()).hexdigest()==ACCEPTED_PILOT_SHA256]
                if not pilots:
                    raise ValueError("The explicitly accepted GPU pilot is not attached")
                shutil.copyfile(pilots[0], target)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(stage/"src")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = "0"
        extras = tomllib.loads((project/"pyproject.toml").read_text())["project"]["optional-dependencies"]
        reference = json.loads((project/"classical_reference_settings.json").read_text())
        print(f"Starting finite GPU {JOB_MODE}; log: {log}", flush=True)
        with log.open("x") as stream:
            result["phase"] = "environment"
            environment = stage/"venv"
            venv.EnvBuilder(with_pip=True).create(environment)
            python = str(environment/"bin/python")
            result["phase"] = "torch"
            child = subprocess.run([python, "-m", "pip", "install", "--index-url", "https://download.pytorch.org/whl/cpu",
                                    "torch=="+reference["versions"]["torch"]],
                                   cwd=stage, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if child.returncode == 0:
                result["phase"] = "dependencies"
                child = subprocess.run([python, "-m", "pip", "install", "-r", str(project/"requirements-lock.txt"),
                                        "-c", str(project/"neural-constraints.txt"), *extras["neural-quantum"],
                                        *extras["quantum-gpu"], *GPU_RUNTIME_PINS],
                                       cwd=stage, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if child.returncode == 0:
                libraries = list(environment.glob("lib/python*/site-packages/nvidia/*/lib"))
                libraries += list(environment.glob("lib/python*/site-packages/cuquantum/lib"))
                paths = [str(path) for path in libraries]
                if env.get("LD_LIBRARY_PATH"):
                    paths.append(env["LD_LIBRARY_PATH"])
                env["LD_LIBRARY_PATH"] = ":".join(paths)
                result["phase"] = JOB_MODE
                command = [python, "-B", "-u", str(stage/"scripts/train_stage2_quantum_gpu.py")]
                command += ["--pilot"] if JOB_MODE=="pilot" else ["--train-all", "--accept-pilot"]
                child = subprocess.run(command,
                                       cwd=stage, env=env, stdout=stream, stderr=subprocess.STDOUT)
        result.update(exit_code=child.returncode, status="complete" if child.returncode==0 else "failed")
    except Exception as error:
        result["error"] = str(error)
        print(str(error), file=sys.stderr, flush=True)
    finally:
        # Checkpoints/results stay in working even if the child fails.
        with status.open("x") as stream:
            json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)
    return result["exit_code"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", action="store_true", help="Build the local private input bundle/metadata only; no remote calls")
    parser.add_argument("--package-dir", type=Path, default=Path("outputs/2107_kaggle_quantum_gpu"))
    parser.add_argument("--accept-gpu-pilot", type=Path, help="Record manual acceptance of a downloaded successful GPU pilot; no launch")
    parser.add_argument("--resume-output", type=Path, help="Package downloaded GPU checkpoints for manual recovery; no launch")
    parser.add_argument("--project-input", type=Path, help="Attached project root; otherwise require exactly one matching input")
    args = parser.parse_args()
    if args.package:
        package(args.package_dir)
        return 0
    if args.accept_gpu_pilot:
        accept_pilot(args.accept_gpu_pilot, args.package_dir)
        return 0
    if args.resume_output:
        recovery(args.resume_output, args.package_dir)
        return 0
    working = Path("/kaggle/working")
    if not working.is_dir():
        parser.error("Run this entry point as a Kaggle Script batch")
    try:
        project = args.project_input
        if project is None:
            matches = list(Path("/kaggle/input").rglob("scripts/train_stage2_neural.py"))
            if len(matches)==1:
                project = matches[0].parent.parent
            else:
                archives = list(Path("/kaggle/input").rglob("project.zip"))
                if matches or len(archives)!=1:
                    raise ValueError("Attach exactly one packaged project input, or specify --project-input")
                project = Path("/tmp/oedi-2107-inputs")
                project.mkdir(exist_ok=False)
                extract(archives[0], project)
        if not project.resolve().is_relative_to(Path("/kaggle/input").resolve()):
            if project != Path("/tmp/oedi-2107-inputs"):
                raise ValueError("Project inputs must be attached under /kaggle/input")
        return run(project, working, Path("/tmp/oedi-2107-physics-model"))
    except Exception as error:
        path = working/"quantum_gpu_batch_status.json"
        if not path.exists():
            path.write_text(json.dumps(dict(workload=JOB_MODE, status="failed", exit_code=1, error=str(error)), indent=2))
        print(str(error), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
