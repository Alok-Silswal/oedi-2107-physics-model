"""Prepare and run separate finite Kaggle GPU pilot/training jobs."""
import argparse
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
import hashlib
import venv
import inspect

JOB_MODE = "pilot"
ACCEPTED_PILOT_SHA256 = None
GPU_RUNTIME_PINS = []
EXPECTED_INPUT_HASHES = {}
RUNTIME_FILES = ("classical_reference_settings.json", "neural-constraints.txt",
                 "pyproject.toml", "requirements-lock.txt")


def input_hashes(archive_path):
    with zipfile.ZipFile(archive_path) as archive:
        names = [entry.filename for entry in archive.infolist() if not entry.is_dir()]
        if len(names)!=len(set(names)):
            raise ValueError("Duplicate project archive entries")
        return {name:hashlib.sha256(archive.read(name)).hexdigest() for name in sorted(names)}


def verify_inputs(project):
    root = Path(project).resolve()
    if not EXPECTED_INPUT_HASHES:
        raise ValueError("Unprepared launcher: package the inputs before submission")
    missing = [name for name in RUNTIME_FILES if name not in EXPECTED_INPUT_HASHES or not (root/name).is_file()]
    if missing:
        raise ValueError(f"Missing required runtime inputs (file or fingerprint): {missing}")
    actual = {p.relative_to(root).as_posix():p for p in root.rglob("*") if p.is_file()}
    # These are transport containers, not consumed project files.
    actual = {name:p for name,p in actual.items() if name not in ("project.zip", "resume.zip")}
    if actual.keys()!=EXPECTED_INPUT_HASHES.keys():
        raise ValueError(f"Wrong project file set: missing={sorted(EXPECTED_INPUT_HASHES.keys()-actual.keys())}, unexpected={sorted(actual.keys()-EXPECTED_INPUT_HASHES.keys())}")
    for name, expected in EXPECTED_INPUT_HASHES.items():
        path = actual[name]
        if not path.resolve().is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError(f"Project SHA-256 mismatch: {name}; attach the dataset version matching this launcher")
    return hashlib.sha256(json.dumps(EXPECTED_INPUT_HASHES, sort_keys=True).encode()).hexdigest()


def portable_manifest(project):
    def unique(pairs):
        result = dict(pairs)
        if len(result)!=len(pairs):
            raise ValueError("Duplicate manifest key")
        return result
    directory = Path(project)/"outputs/2107_stage2_correction"
    path = directory/"hpo/settings.json"
    saved = json.loads(path.read_text(), object_pairs_hook=unique)
    hashes = {}
    for name, sha in saved["plan"]["baseline_hashes"].items():
        relative = PureWindowsPath(name)
        if relative.drive or relative.root or ".." in relative.parts or len(relative.parts)!=3 or relative.parts[:2]!=("outputs", "2107_stage2_correction"):
            raise ValueError(f"Unexpected baseline fingerprint path: {name}")
        key = relative.as_posix()
        if key in hashes:
            raise ValueError(f"Duplicate baseline fingerprint path: {key}")
        hashes[key] = sha
    source = directory/"ridge.joblib"
    key = "outputs/2107_stage2_correction/ridge.joblib"
    if key not in hashes or hashlib.sha256(source.read_bytes()).hexdigest()!=hashes[key]:
        raise ValueError("Saved training-fitted preprocessing fingerprint differs or is missing")
    saved["plan"]["baseline_hashes"] = hashes
    path.write_text(json.dumps(saved, indent=2))


def package(destination, diagnostic=False):
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
        # Normalize only the copied manifest; original provenance stays untouched.
        portable_manifest(staged)
        reference_path = Path("outputs/2107_stage2_neural/settings.json")
        reference = json.loads(reference_path.read_text())
        for name, expected in reference["source_hashes"].items():
            source = staged/"scripts"/name.replace("\\", "/").rsplit("/",1)[-1]
            if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
                raise ValueError("Completed classical source fingerprint changed")
        shutil.copyfile(reference_path, staged/"classical_reference_settings.json")
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
        workload = "diagnostic" if diagnostic and mode=="pilot" else mode
        text = Path(__file__).read_text().replace('JOB_MODE = "pilot"', f'JOB_MODE = "{workload}"', 1)
        text = text.replace("EXPECTED_INPUT_HASHES = {}", f"EXPECTED_INPUT_HASHES = {input_hashes(dataset/'project.zip')!r}", 1)
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
        seen = set()
        for info in archive.infolist():
            target = (destination/info.filename).resolve()
            if not target.is_relative_to(destination) or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe input archive member")
            if target in seen:
                raise ValueError("Duplicate input archive path")
            seen.add(target)
            if target.is_file():
                raise ValueError("Archive would overwrite an existing file")
        archive.extractall(destination)


def benchmark_notebook(package_directory):
    """Embed a small code overlay; reuse the existing, exact input archive."""
    archive = Path(package_directory)/"dataset/project.zip"
    hashes = input_hashes(archive)
    cpu_path = Path("outputs/2107_stage2_quantum_broadcast/benchmark_cpu.json")
    saved = json.loads(cpu_path.read_text())
    with zipfile.ZipFile(archive) as z:
        reference = json.loads(z.read("classical_reference_settings.json"))
        for key in ("settings","split","features","counts","dimension","daylight_n","target_scale_kw",
                    "frame_fingerprints","preprocessing_sha256","versions"):
            if reference[key]!=saved["reference_specification"][key]:
                raise ValueError(f"Packaged inputs differ from CPU benchmark: {key}")
        for name, expected in reference["source_hashes"].items():
            relative = "scripts/"+name.replace("\\","/").rsplit("/",1)[-1]
            if hashes.get(relative)!=expected:
                raise ValueError(f"Packaged reference code differs: {relative}")
    if saved.get("status")!="complete" or not all(saved["equivalence"][str(n)]["passed"] for n in (128,256)):
        raise ValueError("A completed, correct CPU benchmark is required")
    # Exercise the same extraction/content checks locally without loading data or models.
    global EXPECTED_INPUT_HASHES
    previous = EXPECTED_INPUT_HASHES
    try:
        EXPECTED_INPUT_HASHES = hashes
        with tempfile.TemporaryDirectory() as temporary:
            extract(archive,temporary)
            identity = verify_inputs(temporary)
    finally:
        EXPECTED_INPUT_HASHES = previous
    overlay = Path("scripts/train_stage2_quantum_gpu.py").read_text()
    preflight = "import hashlib, json, os, subprocess, sys, time, venv, zipfile\nfrom pathlib import Path\n"
    preflight += f"EXPECTED_INPUT_HASHES = {hashes!r}\nRUNTIME_FILES = {RUNTIME_FILES!r}\n"
    preflight += inspect.getsource(extract)+"\n"+inspect.getsource(verify_inputs)
    preflight += f'''
started = time.perf_counter()
archives = list(Path("/kaggle/input").rglob("project.zip"))
if len(archives)!=1:
    raise ValueError("Attach exactly one existing private PV2107 project.zip input")
if hashlib.sha256(archives[0].read_bytes()).hexdigest()!={hashlib.sha256(archive.read_bytes()).hexdigest()!r}:
    raise ValueError("Wrong dataset archive/version; stop and select the matching existing version, not a blind upload")
root = Path("/tmp/pv2107-native-benchmark")
root.mkdir(exist_ok=False)
project = root/"project"
extract(archives[0],project)
print("Verified input identity:", verify_inputs(project))
work = Path("/kaggle/working/native_benchmark")
work.mkdir(exist_ok=False)
(root/"train_stage2_quantum_gpu.py").write_text({overlay!r})
(root/"benchmark_cpu.json").write_text({cpu_path.read_text()!r})
print("Verified inputs; no fitting or timing has run.")
'''
    setup = '''
env = os.environ.copy()
env.update(PYTHONPATH=os.pathsep.join(map(str,(root,project/"scripts",project/"src"))),
           PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
venv.EnvBuilder(with_pip=False).create(root/"venv")
python = str(root/"venv/bin/python")
pip = [sys.executable,"-m","pip","--python",python]
reference = json.loads((project/"classical_reference_settings.json").read_text())
pins = [f"{name}=={reference['versions'][name]}" for name in ("pennylane","pennylane-lightning")]
constraints = root/"cuda-constraints.txt"
constraints.write_text("torch==2.11.0+cu128\\n"+"\\n".join(pins)+"\\n")
status = dict(status="failed", phase="environment", input_fingerprint=verify_inputs(project))
try:
    with (work/"setup.log").open("x") as log:
        subprocess.run([*pip,"install","--only-binary=:all:","--index-url","https://download.pytorch.org/whl/cu128",
                        "torch==2.11.0+cu128"], check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
        subprocess.run([*pip,"install","--only-binary=:all:","-r",str(project/"requirements-lock.txt"),
                        "-c",str(constraints),*pins], check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
    subprocess.run([python,"-c","import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"],check=True,env=env)
    status.update(status="prepared",phase="ready")
finally:
    status["elapsed_seconds"] = time.perf_counter()-started
    (work/"status.json").write_text(json.dumps(status,indent=2))
print("CUDA Torch ready. Setup log:",work/"setup.log")
'''
    benchmark = '''
if status["status"]!="prepared":
    raise ValueError("Environment preparation must finish successfully first")
status.update(status="failed",phase="benchmark")
runner = "from pathlib import Path; import train_stage2_quantum_gpu as b; b.OUT=Path('/kaggle/working/native_benchmark/unused');\\nwith b.threadpool_limits(limits=4): b.main()"
try:
    with (work/"benchmark.log").open("x") as log:
        subprocess.run([python,"-B","-u","-c",runner,"--broadcast-benchmark","--device","cuda",
                        "--cpu-reference",str(root/"benchmark_cpu.json"),"--benchmark-torch-release","2.11.0"],
                       cwd=project,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    result = json.loads((work/"2107_stage2_quantum_broadcast/benchmark_cuda.json").read_text())
    print(json.dumps({key:result[key] for key in ("status","native_broadcast","summary","cpu_to_gpu_speed_ratios","within_25_hour_compute_budget")},indent=2))
    status.update(status="complete",phase="finished")
finally:
    status["elapsed_seconds"] = time.perf_counter()-started
    (work/"status.json").write_text(json.dumps(status,indent=2))
    print("Download native_benchmark outputs and stop the session. Status:",status)
'''
    note = ("# PV2107 bounded native-batch GPU benchmark\n"
            "Keep this notebook private; attach the existing private `anokgaming/pv2107-quantum-inputs` dataset. "
            "Choose a single-GPU accelerator if available, enable Internet, then Save Version / Run All for a finite batch. "
            "Only CUDA device 0 is used; hiding a second allocated GPU does not release it. "
            "No dataset upload or full training is needed. Input verification precedes installation.\n\n"
            "Target: under one GPU-hour including setup, not a guaranteed wall-clock bound. "
            "Five warmed repetitions on the same 1,024 training rows at batches 128 and 256; "
            "the CPU row reference is used only for unchanged correctness checks. "
            "CUDA state/device checks and synchronization are mandatory. "
            "Torch 2.11.0+cu128 is an explicit isolated benchmark exception to CPU 2.14.1+cpu; "
            "other reference versions remain pinned. Timing differences include software/hardware effects. "
            "Batch 256 is a candidate, not approved training. No retry or transition to training.\n\n"
            "Download `native_benchmark/` (results, status and logs). If using an interactive session, stop it manually afterward. "
            "On input mismatch, identify the matching existing dataset version before continuing.\n")
    cells = [dict(cell_type="markdown",metadata={},source=note.splitlines(keepends=True))]
    for code in (preflight,setup,benchmark):
        compile(code,"<benchmark-notebook>","exec")
        cells.append(dict(cell_type="code",metadata={},source=code.splitlines(keepends=True),execution_count=None,outputs=[]))
    path = cpu_path.parent/"kaggle_benchmark.ipynb"
    with path.open("x",encoding="utf-8") as stream:
        json.dump(dict(nbformat=4,nbformat_minor=4,metadata=dict(kernelspec=dict(display_name="Python 3",language="python",name="python3")),cells=cells),stream,indent=1)
    print("Notebook only (no rebuild/upload):",path,"input identity:",identity)


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
    text = text.replace("EXPECTED_INPUT_HASHES = {}", f"EXPECTED_INPUT_HASHES = {input_hashes(destination/'dataset/project.zip')!r}", 1)
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
        result["phase"] = "inputs"
        result["input_fingerprint"] = verify_inputs(project)
        print(f"Verified project inputs: {result['input_fingerprint']}", flush=True)
        # Refuse reuse rather than overwrite an earlier job's outputs/checkpoints.
        stage.mkdir(exist_ok=False)
        for name in ("scripts", "src", "data"):
            source = project/name
            if not source.is_dir():
                raise ValueError(f"Missing project input directory: {source}")
            (stage/name).symlink_to(source, target_is_directory=True)
        (stage/"classical_reference_settings.json").symlink_to(project/"classical_reference_settings.json")
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
            # Kaggle Python may omit ensurepip; host pip can manage a pip-less venv.
            venv.EnvBuilder(with_pip=False).create(environment)
            python = str(environment/"bin/python")
            pip = [sys.executable, "-m", "pip", "--python", python]
            result["phase"] = "torch"
            child = subprocess.run([*pip, "install", "--index-url", "https://download.pytorch.org/whl/cpu",
                                    "torch=="+reference["versions"]["torch"]],
                                   cwd=stage, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if child.returncode == 0:
                result["phase"] = "dependencies"
                child = subprocess.run([*pip, "install", "-r", str(project/"requirements-lock.txt"),
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
                command += ["--"+JOB_MODE] if JOB_MODE in ("pilot", "diagnostic") else ["--train-all", "--accept-pilot"]
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
    parser.add_argument("--benchmark-notebook", action="store_true", help="Prepare native CUDA benchmark notebook using an existing --package-dir; no archive rebuild")
    parser.add_argument("--diagnostic", action="store_true", help="With --package, prepare the optional batching diagnostic instead of the normal pilot")
    parser.add_argument("--package-dir", type=Path, default=Path("outputs/2107_kaggle_quantum_gpu"))
    parser.add_argument("--accept-gpu-pilot", type=Path, help="Record manual acceptance of a downloaded successful GPU pilot; no launch")
    parser.add_argument("--resume-output", type=Path, help="Package downloaded GPU checkpoints for manual recovery; no launch")
    parser.add_argument("--project-input", type=Path, help="Attached project root; otherwise require exactly one matching input")
    args = parser.parse_args()
    if args.benchmark_notebook:
        if args.package or args.diagnostic or args.accept_gpu_pilot or args.resume_output:
            parser.error("--benchmark-notebook must be used separately")
        benchmark_notebook(args.package_dir)
        return 0
    if args.diagnostic and not args.package:
        parser.error("--diagnostic requires --package")
    if args.package:
        package(args.package_dir, diagnostic=args.diagnostic)
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
