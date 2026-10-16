import json
import hashlib
from pathlib import Path
import sys
import subprocess
import shutil
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import train_stage2_quantum_gpu as gpu
import kaggle_stage2_pilot as batch


def test_exact_package_identity_survives_relocation_and_rejects_mixed_inputs(tmp_path, monkeypatch):
    archive = tmp_path/"project.zip"
    with zipfile.ZipFile(archive,"w") as z:
        for name in batch.RUNTIME_FILES:
            z.writestr(name,b"{}")
        z.writestr("outputs/2107_stage2_correction/ridge.joblib",b"preprocessing")
        z.writestr("outputs/2107_stage2_correction/hpo/settings.json",b"portable manifest")
    monkeypatch.setattr(batch,"EXPECTED_INPUT_HASHES",batch.input_hashes(archive))
    identities = []
    for name in ("checkout", "kaggle/input/datasets/owner/project"):
        root = tmp_path/name
        batch.extract(archive,root)
        identities.append(batch.verify_inputs(root))
        source = root/"outputs/2107_stage2_correction/ridge.joblib"
        source.write_bytes(b"changed preprocessing")
        with pytest.raises(ValueError,match="SHA-256 mismatch"):
            batch.verify_inputs(root)
        source.unlink()
        with pytest.raises(ValueError,match="missing="):
            batch.verify_inputs(root)
        source.write_bytes(b"preprocessing")
        extra = root/"ridge.joblib"
        extra.write_bytes(b"preprocessing")
        with pytest.raises(ValueError,match="unexpected="):
            batch.verify_inputs(root)
        extra.unlink()
        (root/"outputs/2107_stage2_correction/hpo/settings.json").write_bytes(b"stale manifest")
        with pytest.raises(ValueError,match="hpo/settings.json"):
            batch.verify_inputs(root)
    assert identities[0]==identities[1]


def test_duplicate_archive_members_and_literal_manifest_keys_fail(tmp_path):
    archive = tmp_path/"project.zip"
    with zipfile.ZipFile(archive,"w") as z:
        z.writestr("ridge.joblib",b"first")
        with pytest.warns(UserWarning,match="Duplicate"):
            z.writestr("ridge.joblib",b"second")
    with pytest.raises(ValueError,match="Duplicate"):
        batch.input_hashes(archive)
    with pytest.raises(ValueError,match="Duplicate"):
        batch.extract(archive,tmp_path/"unpacked")
    manifest=tmp_path/"outputs/2107_stage2_correction/hpo/settings.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"plan":{"baseline_hashes":{},"baseline_hashes":{}}}')
    with pytest.raises(ValueError,match="Duplicate"):
        batch.portable_manifest(tmp_path)


def test_wrong_package_fails_before_environment_or_pilot(tmp_path,monkeypatch):
    project,work,stage=(tmp_path/name for name in ("project","working","stage"))
    project.mkdir()
    work.mkdir()
    (project/"manifest").write_bytes(b"old")
    expected = {"manifest":hashlib.sha256(b"new").hexdigest()}
    for name in batch.RUNTIME_FILES:
        (project/name).write_bytes(b"{}")
        expected[name] = hashlib.sha256(b"{}").hexdigest()
    monkeypatch.setattr(batch,"EXPECTED_INPUT_HASHES",expected)
    def forbidden(*args,**kwargs):
        raise AssertionError("Wrong inputs must not install or execute")
    monkeypatch.setattr(batch.subprocess,"run",forbidden)
    monkeypatch.setattr(batch.venv.EnvBuilder,"create",forbidden)
    assert batch.run(project,work,stage)==1
    status=json.loads((work/"quantum_gpu_batch_status.json").read_text())
    assert status["phase"]=="inputs" and "SHA-256 mismatch" in status["error"]
    assert not stage.exists()


@pytest.mark.parametrize("change", ["missing", "altered", "unlisted"])
def test_reference_preflight_stops_before_installation(tmp_path,monkeypatch,change):
    project,work,stage=(tmp_path/name for name in ("project","working","stage"))
    project.mkdir()
    work.mkdir()
    expected = {}
    for name in batch.RUNTIME_FILES:
        (project/name).write_bytes(b"{}")
        expected[name] = hashlib.sha256(b"{}").hexdigest()
    reference = project/"classical_reference_settings.json"
    if change=="missing":
        reference.unlink()
    elif change=="altered":
        reference.write_bytes(b"changed")
    else:
        del expected[reference.name]
    monkeypatch.setattr(batch,"EXPECTED_INPUT_HASHES",expected)
    def forbidden(*args,**kwargs):
        raise AssertionError("Invalid reference must not install or run")
    monkeypatch.setattr(batch.subprocess,"run",forbidden)
    monkeypatch.setattr(batch.venv.EnvBuilder,"create",forbidden)
    assert batch.run(project,work,stage)==1
    status=json.loads((work/"quantum_gpu_batch_status.json").read_text())
    assert status["phase"]=="inputs" and reference.name in status["error"]
    assert not stage.exists()


def test_packaged_fingerprints_are_root_independent_and_reject_changed_content(tmp_path):
    key = "outputs/2107_stage2_correction/ridge.joblib"
    sha = hashlib.sha256(b"original preprocessing").hexdigest()
    original = dict(plan=dict(baseline_hashes={key.replace("/", "\\"):sha}, frame_fingerprints={"training":"unchanged"}))
    for root in (tmp_path/"checkout", tmp_path/"kaggle/input/datasets/owner/dataset"):
        source = root/key
        source.parent.mkdir(parents=True)
        source.write_bytes(b"original preprocessing")
        manifest = source.parent/"hpo/settings.json"
        manifest.parent.mkdir()
        manifest.write_text(json.dumps(original))
        batch.portable_manifest(root)
        saved = json.loads(manifest.read_text())
        assert saved==dict(original,plan=dict(original["plan"],baseline_hashes={key:sha}))
        # This is the unchanged prepare_data lookup, including resolved mount paths.
        expected = {str((root/Path(p)).resolve()):h for p,h in saved["plan"]["baseline_hashes"].items()}
        assert hashlib.sha256(source.read_bytes()).hexdigest()==expected[str(source.resolve())]
        before = manifest.read_bytes()
        source.write_bytes(b"changed preprocessing")
        with pytest.raises(ValueError,match="fingerprint"):
            batch.portable_manifest(root)
        assert manifest.read_bytes()==before


@pytest.mark.parametrize("keys", [
    ["elsewhere/ridge.joblib"],
    ["outputs/another_experiment/ridge.joblib"],
    ["C:/old-checkout/outputs/2107_stage2_correction/ridge.joblib"],
    ["outputs/2107_stage2_correction/../ridge.joblib"],
    ["outputs/2107_stage2_correction/forest.joblib"],
    ["outputs/2107_stage2_correction/ridge.joblib", "outputs\\2107_stage2_correction\\ridge.joblib"],
])
def test_packaged_manifest_rejects_wrong_missing_or_ambiguous_identity(tmp_path,keys):
    directory = tmp_path/"outputs/2107_stage2_correction/hpo"
    directory.mkdir(parents=True)
    (directory.parent/"ridge.joblib").write_bytes(b"content")
    manifest = directory/"settings.json"
    manifest.write_text(json.dumps(dict(plan=dict(baseline_hashes={key:hashlib.sha256(b"content").hexdigest() for key in keys}))))
    before = manifest.read_bytes()
    with pytest.raises(ValueError):
        batch.portable_manifest(tmp_path)
    assert manifest.read_bytes()==before


def test_reused_cpu_circuit_float64_gradients_and_matched_initialization():
    torch = pytest.importorskip("torch")
    pytest.importorskip("pennylane")
    old = torch.get_default_dtype()
    try:
        torch.set_num_threads(4)
        for seed in gpu.cpu.SEEDS:
            torch.set_default_dtype(torch.float32)
            classical = gpu.CPU_FACTORY("classical", 14, seed)
            torch.set_default_dtype(torch.float64)
            quantum = gpu.make_model(14, seed, "lightning.qubit")
            assert sum(p.numel() for p in quantum.parameters())==89
            for name in ("projection", "head"):
                for a, b in zip(getattr(classical,name).parameters(), getattr(quantum,name).parameters()):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
        first = gpu.make_model(14, 2107, "lightning.qubit")
        second = gpu.make_model(14, 2107, "lightning.qubit")
        x = torch.randn(3,14,generator=torch.Generator().manual_seed(11),dtype=torch.float64)
        result = gpu.compare(first,second,x,torch.tensor([0.4,-0.3,1.2]))
        assert result["passed"] and result["expectation_dtype"]=="float64"
        with pytest.raises(ValueError, match="GPU statevector"):
            gpu.gpu_evidence(first)
    finally:
        torch.set_default_dtype(old)


@pytest.mark.parametrize("size", [1, 128])
@pytest.mark.parametrize("backend", ["lightning.qubit", "default.qubit"])
def test_broadcast_qnode_preserves_circuit_outputs_and_all_gradients(size,backend):
    torch = pytest.importorskip("torch")
    qml = pytest.importorskip("pennylane")
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(4)
        current = gpu.make_model(14,2107,"lightning.qubit")
        candidate = gpu.make_model(14,2107,backend,batched=True)
        assert sum(p.numel() for p in candidate.parameters())==89
        x = torch.randn(size,14,generator=torch.Generator().manual_seed(11))
        y = torch.linspace(-1,1,size)
        result = gpu.compare(current,candidate,x,y)
        assert result["passed"] and result["loss_max_abs"] < gpu.TOLERANCES["output_atol"]
        if backend=="default.qubit":
            evidence = gpu.broadcast_evidence(candidate,x)
            assert evidence["tracker"]["simulations"]==1 and evidence["state_dtype"]=="torch.complex128"
        angles = candidate.projection(x).tanh()*torch.pi
        tape = candidate.circuit.construct((angles.T,candidate.weights),{})
        assert candidate.circuit.device.supports_derivatives(qml.devices.ExecutionConfig(gradient_method=candidate.circuit.diff_method),tape)
        expanded,_ = qml.transforms.broadcast_expand(tape)
        for i, item in enumerate(expanded):
            row = current.circuit.construct((angles[i],current.weights),{})
            qml.assert_equal(item,row)
    finally:
        torch.set_default_dtype(old)


@pytest.mark.parametrize("failure", ["version", "unsupported", "mismatch"])
def test_diagnostic_never_times_unsupported_or_inequivalent_candidate(monkeypatch,failure):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pennylane")
    monkeypatch.setattr(gpu.metadata,"version",lambda name:"0.44.0" if failure=="version" else "0.45.0")
    current = gpu.make_model(14,2107,"lightning.qubit")
    candidate = gpu.make_model(14,2107,"lightning.qubit",batched=True)
    monkeypatch.setattr(candidate.circuit.device,"supports_derivatives",lambda *args:failure!="unsupported")
    def forbidden(*args,**kwargs):
        raise AssertionError("Must not time failed compatibility")
    monkeypatch.setattr(gpu,"timing",forbidden)
    def compare(*args):
        raise ValueError("synthetic gradient mismatch")
    monkeypatch.setattr(gpu,"compare",compare)
    with pytest.raises(ValueError,match="pinned|unsupported|gradient mismatch"):
        gpu.batch_diagnostic(current,current,candidate,torch.zeros(3,14,dtype=torch.float64),torch.zeros(3),3,3)


def test_diagnostic_packaging_option_is_local_and_explicit(monkeypatch,tmp_path):
    calls = []
    monkeypatch.setattr(batch,"package",lambda destination,diagnostic:calls.append((destination,diagnostic)))
    monkeypatch.setattr(sys,"argv",["batch","--package","--diagnostic","--package-dir",str(tmp_path)])
    assert batch.main()==0 and calls==[(tmp_path,True)]
    monkeypatch.setattr(sys,"argv",["batch","--diagnostic"])
    with pytest.raises(SystemExit) as error:
        batch.main()
    assert error.value.code==2 and len(calls)==1


@pytest.mark.parametrize("size", [64,128,256])
def test_timing_waits_for_completed_work_and_sync_failure_propagates(monkeypatch,size):
    torch = pytest.importorskip("torch")
    events = []
    mode = dict(training=True)
    model = SimpleNamespace(train=lambda:mode.update(training=True),eval=lambda:mode.update(training=False),
                            parameters=lambda:iter([torch.nn.Parameter(torch.zeros(1))]),
                            circuit=SimpleNamespace(device=SimpleNamespace(name="lightning.gpu")))
    monkeypatch.setattr(torch.optim,"Adam",lambda *args,**kwargs:SimpleNamespace(
        zero_grad=lambda **kwargs:None,step=lambda:events.append("update")))
    monkeypatch.setattr(torch.nn,"HuberLoss",lambda **kwargs:lambda *args:SimpleNamespace(backward=lambda:None,item=lambda:0.))
    # Callable proxy keeps this a timing-control test, without fitting an estimator.
    class Model:
        def __getattr__(self,name): return getattr(model,name)
        def __call__(self,x):
            if not mode["training"]: events.append("inference")
            return x
    monkeypatch.setattr(gpu.time,"perf_counter",lambda:events.append("clock") or float(len(events)))
    def sync(): events.append("sync")
    pool = SimpleNamespace(getTotalDevices=lambda:1,setDeviceID=lambda i:None)
    pool_type = lambda:SimpleNamespace(syncDevice=sync)
    pool_type.getTotalDevices = pool.getTotalDevices
    pool_type.setDeviceID = pool.setDeviceID
    monkeypatch.setitem(sys.modules,"pennylane_lightning.lightning_gpu_ops",SimpleNamespace(DevPool=pool_type))
    x = torch.zeros(1024,14,dtype=torch.float64)
    result = gpu.timing(Model(),x,x[:,0],221057,70555,size)
    assert events==["update","sync","clock"]+["update"]*(1024//size)+["sync","clock","inference","sync","clock"]+["inference"]*(1024//size)+["sync","clock"]
    assert np.isclose(result["rough_three_seeds_100_epochs_hours"],.6*result["rough_five_seeds_100_epochs_hours"])
    pool_type.getTotalDevices = lambda:2
    with pytest.raises(ValueError,match="exactly one"):
        gpu.synchronize(model)
    pool_type.getTotalDevices = lambda:1
    def broken(): raise RuntimeError("CUDA synchronization failed")
    pool_type.setDeviceID = lambda i:broken()
    with pytest.raises(RuntimeError,match="synchronization failed"):
        gpu.synchronize(model)


def test_native_benchmark_failure_saves_separate_evidence_without_timing(monkeypatch,tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pennylane")
    import pandas as pd
    monkeypatch.setattr(gpu,"OUT",tmp_path/"reference")
    x = dict(training=np.zeros((1024,14)),validation=np.zeros((10,14)))
    tables = dict(training=pd.DataFrame(dict(correction_kw=np.ones(1024))))
    spec = dict(dimension=14,target_scale_kw=2.)
    def mismatch(*args): raise ValueError("gradient mismatch")
    def forbidden(*args): raise AssertionError("Failed correctness must not time or train")
    monkeypatch.setattr(gpu,"compare",mismatch)
    monkeypatch.setattr(gpu,"timing",forbidden)
    monkeypatch.setattr(gpu,"train_all",forbidden)
    with pytest.raises(ValueError,match="gradient mismatch"):
        gpu.broadcast_benchmark(x,tables,spec,"cpu")
    path = tmp_path/"2107_stage2_quantum_broadcast/benchmark_cpu.json"
    saved = json.loads(path.read_text())
    assert saved["status"]=="failed" and "gradient mismatch" in saved["error"]
    with pytest.raises(ValueError,match="successfully"):
        gpu.require_pilot(path,spec,True)
    before = path.read_bytes()
    with pytest.raises(ValueError,match="not be overwritten"):
        gpu.broadcast_benchmark(x,tables,spec,"cpu")
    assert path.read_bytes()==before


def test_native_benchmark_rejects_unavailable_cuda_before_data_loading(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    def forbidden(): raise AssertionError("Unavailable CUDA must fail before loading data")
    monkeypatch.setattr(gpu.cpu,"prepare_data",forbidden)
    monkeypatch.setattr(sys,"argv",["gpu","--broadcast-benchmark","--device","cuda"])
    with pytest.raises(SystemExit) as error:
        gpu.main()
    assert error.value.code==2


def test_native_cuda_sync_uses_torch_device(monkeypatch):
    torch = pytest.importorskip("torch")
    device = torch.device("cuda:0")
    calls = []
    model = SimpleNamespace(parameters=lambda:iter([SimpleNamespace(device=device)]),
                            circuit=SimpleNamespace(device=SimpleNamespace(name="default.qubit")))
    monkeypatch.setattr(torch.cuda,"synchronize",calls.append)
    gpu.synchronize(model)
    assert calls==[device]


@pytest.mark.parametrize("changed", [False, True])
def test_saved_cpu_comparison_only_times_requested_batches(monkeypatch,tmp_path,changed):
    pytest.importorskip("torch")
    pytest.importorskip("pennylane")
    import pandas as pd
    monkeypatch.setattr(gpu,"OUT",tmp_path/"reference")
    spec = {key:{} for key in ("architecture","settings","split","features","counts","daylight_n",
                              "target_scale_ddof","target_centered","frame_fingerprints","preprocessing_sha256","versions")}
    spec.update(dimension=14,target_scale_kw=2.)
    indices = np.random.default_rng(2107).choice(1024,1024,replace=False)
    metric = "rough_three_seeds_100_epochs_hours"
    saved = dict(status="complete",device="cpu",seed=2107,repetitions=5,parameters=89,
                 sample_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),reference_specification=spec,
                 benchmark_source_sha256="old timing implementation",equivalence={str(n):dict(passed=True) for n in (128,256)},
                 execution=dict(backend="default.qubit",differentiation="backprop",shots=None,
                                torch_dtype="float64",state_dtype="complex128",torch_device="cpu"),
                 summary={f"broadcast_{n}":{metric:dict(median=20.)} for n in (128,256)})
    path = tmp_path/"saved_cpu.json"
    path.write_text(json.dumps(saved))
    before = path.read_bytes()
    if changed:
        spec["frame_fingerprints"] = dict(training="different data")
    calls = []
    monkeypatch.setattr(gpu,"compare",lambda *args:dict(passed=True))
    monkeypatch.setattr(gpu,"broadcast_evidence",lambda *args:dict(verified=True))
    def measured(model,x,y,training_n,validation_n,size):
        calls.append(size)
        assert len(x)==1024 and training_n==1024 and validation_n==10
        return {metric:10.}
    monkeypatch.setattr(gpu,"timing",measured)
    x = dict(training=np.zeros((1024,14)),validation=np.zeros((10,14)))
    tables = dict(training=pd.DataFrame(dict(correction_kw=np.ones(1024))))
    if changed:
        with pytest.raises(ValueError,match="frame_fingerprints"):
            gpu.broadcast_benchmark(x,tables,spec,"cpu",cpu_reference=path)
        assert not calls
    else:
        gpu.broadcast_benchmark(x,tables,spec,"cpu",cpu_reference=path)
        assert len(calls)==10 and calls.count(128)==calls.count(256)==5
        result = json.loads((tmp_path/"2107_stage2_quantum_broadcast/benchmark_cpu.json").read_text())
        assert result["cpu_to_gpu_speed_ratios"]==dict(broadcast_128=2.,broadcast_256=2.)
        assert result["status"]=="complete" and "reference_row_128" not in result["timings"]
    assert path.read_bytes()==before


def test_benchmark_torch_exception_cannot_change_production(monkeypatch):
    pytest.importorskip("torch")
    def forbidden(): raise AssertionError("Must reject before loading data")
    monkeypatch.setattr(gpu.cpu,"prepare_data",forbidden)
    monkeypatch.setattr(sys,"argv",["gpu","--prepare","--benchmark-torch-release","2.11.0"])
    with pytest.raises(SystemExit) as error:
        gpu.main()
    assert error.value.code==2


def test_notebook_preparation_checks_archive_without_setup_or_benchmark(monkeypatch,tmp_path):
    import ast
    reference = {key:{} for key in ("settings","split","features","counts","dimension","daylight_n","target_scale_kw",
                                    "frame_fingerprints","preprocessing_sha256","versions","source_hashes")}
    saved = dict(status="complete",reference_specification=reference,
                 equivalence={str(n):dict(passed=True) for n in (128,256)})
    monkeypatch.chdir(tmp_path)
    directory = tmp_path/"outputs/2107_stage2_quantum_broadcast"
    directory.mkdir(parents=True)
    (directory/"benchmark_cpu.json").write_text(json.dumps(saved))
    (tmp_path/"scripts").mkdir()
    (tmp_path/"scripts/train_stage2_quantum_gpu.py").write_text("# synthetic overlay\n")
    package = tmp_path/"prepared"
    (package/"dataset").mkdir(parents=True)
    with zipfile.ZipFile(package/"dataset/project.zip","w") as archive:
        for name in batch.RUNTIME_FILES:
            archive.writestr(name,json.dumps(reference) if name=="classical_reference_settings.json" else "")
    def forbidden(*args,**kwargs): raise AssertionError("Preparation must not install, compute or rebuild inputs")
    monkeypatch.setattr(batch.subprocess,"run",forbidden)
    monkeypatch.setattr(batch.venv.EnvBuilder,"create",forbidden)
    monkeypatch.setattr(batch,"package",forbidden)
    before = (package/"dataset/project.zip").read_bytes()
    batch.benchmark_notebook(package)
    notebook = json.loads((directory/"kaggle_benchmark.ipynb").read_text())
    code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"]=="code"]
    for cell in code:
        ast.parse(cell)
    assert "verify_inputs(project)" in code[0] and "subprocess.run(" not in code[0]
    assert "--cpu-reference" in code[-1] and "--train-all" not in code[-1]
    assert "torch==2.11.0+cu128" in code[1] and "with_pip=False" in code[1]
    assert (package/"dataset/project.zip").read_bytes()==before
    with pytest.raises(FileExistsError):
        batch.benchmark_notebook(package)


@pytest.mark.parametrize("mode,version,release,allowed", [
    ("--broadcast-benchmark","2.14.1+cu128",None,True),
    ("--broadcast-benchmark","2.15.0+cu128",None,False),
    ("--broadcast-benchmark","2.11.0+cu128","2.11.0",True),
    ("--broadcast-benchmark","2.10.0+cu128","2.11.0",False),
    ("--prepare","2.14.1+cu128",None,False)])
def test_torch_build_exception_is_limited_to_benchmark(monkeypatch,tmp_path,mode,version,release,allowed):
    torch = pytest.importorskip("torch")
    reference = {name:{} for name in ("settings","split","seeds","features","dimension","counts","daylight_n",
                                     "target_scale_kw","frame_fingerprints","preprocessing_sha256","source_hashes")}
    reference.update(dimension=14,versions=dict(torch="2.14.1+cpu"),source_commit="completed-classical")
    (tmp_path/"classical_reference_settings.json").write_text(json.dumps(reference))
    monkeypatch.chdir(tmp_path)
    original = dict(reference,dimension=14,versions=dict(torch=version))
    monkeypatch.setattr(gpu.cpu,"prepare_data",lambda:({}, {}, None, original, {}))
    calls = []
    monkeypatch.setattr(gpu,"broadcast_benchmark",lambda *args:calls.append(args))
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    options = ["--device","cuda" if release else "cpu"] if mode=="--broadcast-benchmark" else []
    if release:
        options += ["--benchmark-torch-release",release]
    monkeypatch.setattr(sys,"argv",["gpu",mode]+options)
    old = torch.get_default_dtype()
    try:
        if allowed:
            gpu.main()
            assert len(calls)==1 and calls[0][2]["versions"]["torch"]==version
        else:
            with pytest.raises(ValueError,match="versions"):
                gpu.main()
            assert not calls
    finally:
        torch.set_default_dtype(old)


def test_gpu_pilot_requires_acceptance_success_and_exact_compatibility(tmp_path):
    spec = dict(features=["past_only"], seed=2107, settings=dict(backend="lightning.gpu"))
    path = tmp_path/"gpu_pilot.json"
    saved = dict(status="complete", specification=spec, correctness=dict(passed=True), gpu=dict(verified=True))
    path.write_text(json.dumps(saved))
    assert gpu.require_pilot(path,spec,True)==saved
    with pytest.raises(ValueError, match="explicit"):
        gpu.require_pilot(path,spec,False)
    with pytest.raises(ValueError, match="incompatible"):
        gpu.require_pilot(path,dict(spec,features=["changed"]),True)
    saved["gpu"]["verified"] = False
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="successfully"):
        gpu.require_pilot(path,spec,True)
    path.write_text(json.dumps(dict(status="complete",specification=spec,equivalence=dict(passed=True),gpu=dict(verified=True))))
    with pytest.raises(ValueError,match="successfully"):
        gpu.require_pilot(path,spec,True)
    with pytest.raises(ValueError,match="unsuccessful"):
        batch.accept_pilot(path,tmp_path)


def test_safe_recovery_extract_rejects_traversal_and_overwriting(tmp_path):
    archive = tmp_path/"resume.zip"
    target = tmp_path/"working"
    target.mkdir()
    with zipfile.ZipFile(archive,"w") as z:
        z.writestr("../outside", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        batch.extract(archive,target)
    assert not (tmp_path/"outside").exists()
    with zipfile.ZipFile(archive,"w") as z:
        z.writestr("last.pt", "checkpoint")
    batch.extract(archive,target)
    with pytest.raises(ValueError, match="overwrite"):
        batch.extract(archive,target)
    assert (target/"last.pt").read_text()=="checkpoint"


def test_restricted_checkpoint_roundtrip_and_rejection(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pennylane")
    model = gpu.make_model(14,2107,"lightning.qubit")
    path = tmp_path/"last.pt"
    spec = dict(backend="lightning.gpu", seed=2107)
    gpu.cpu.checkpoint(path,dict(specification=spec,model=model.state_dict(),numpy_rng=np.random.get_state()))
    state = gpu.read_checkpoint(path,spec)
    torch.testing.assert_close(state["model"]["weights"],model.weights,rtol=0,atol=0)
    with pytest.raises(ValueError, match="Incompatible"):
        gpu.read_checkpoint(path,dict(spec,seed=2108))


@pytest.mark.parametrize("phase, exit_code", [("pilot", 0), ("pilot", 7), ("torch", 9), ("dependencies", 11)])
@pytest.mark.parametrize("workload", ["pilot", "diagnostic"])
def test_finite_gpu_pilot_has_no_transition_retry_or_timeout(monkeypatch, tmp_path, phase, exit_code, workload):
    monkeypatch.setattr(batch,"JOB_MODE",workload)
    monkeypatch.setattr(batch,"verify_inputs",lambda project:"synthetic verified package")
    project, work, stage = (tmp_path/name for name in ("input", "working", "stage"))
    work.mkdir()
    for name in ("scripts", "src", "data", "outputs/2107_stage2_weather", "outputs/2107_stage2_correction"):
        (project/name).mkdir(parents=True)
    (project/"pyproject.toml").write_text('[project.optional-dependencies]\nneural-quantum=[]\nquantum-gpu=[]\n')
    (project/"classical_reference_settings.json").write_text(json.dumps(dict(versions=dict(torch="2.14.1+cpu"))))
    def link(target,source,target_is_directory=False):
        if not target_is_directory:
            shutil.copyfile(source,target)
    monkeypatch.setattr(Path, "symlink_to", link)
    created = []
    def create(builder, environment):
        assert builder.with_pip is False
        created.append(environment)
    monkeypatch.setattr(batch.venv.EnvBuilder, "create", create)
    commands = []
    def child(command, **kwargs):
        assert "timeout" not in kwargs
        assert (kwargs["cwd"]/"classical_reference_settings.json").read_bytes()==(project/"classical_reference_settings.json").read_bytes()
        commands.append(command)
        current = ("torch", "dependencies", "pilot")[len(commands)-1]
        if current!="pilot":
            assert command[:6]==[sys.executable,"-m","pip","--python",str(stage.resolve()/"venv/bin/python"),"install"]
        return subprocess.CompletedProcess(command, exit_code if current==phase else 0)
    monkeypatch.setattr(batch.subprocess, "run", child)
    assert batch.run(project,work,stage)==exit_code
    assert created==[stage.resolve()/"venv"]
    assert len(commands)==("torch", "dependencies", "pilot").index(phase)+1
    if phase=="pilot":
        assert commands[-1][-1]=="--"+workload
    assert not any("--train-all" in command for command in commands)
    status = json.loads((work/"quantum_gpu_batch_status.json").read_text())
    assert status["status"]==("complete" if exit_code==0 else "failed")
    assert status["phase"]==(workload if phase=="pilot" else phase)
    assert (work/"quantum_gpu_batch.log").exists()


def test_full_batch_requires_manual_acceptance_before_setup(monkeypatch, tmp_path):
    monkeypatch.setattr(batch,"verify_inputs",lambda project:"synthetic verified package")
    project, work, stage = (tmp_path/name for name in ("input", "working", "stage"))
    work.mkdir()
    for name in ("scripts", "src", "data", "outputs/2107_stage2_weather", "outputs/2107_stage2_correction"):
        (project/name).mkdir(parents=True)
    monkeypatch.setattr(batch, "JOB_MODE", "training")
    monkeypatch.setattr(batch, "ACCEPTED_PILOT_SHA256", None)
    monkeypatch.setattr(Path, "symlink_to", lambda *args, **kwargs: None)
    def forbidden(*args, **kwargs):
        raise AssertionError("Unaccepted job must not install or train")
    monkeypatch.setattr(batch.subprocess, "run", forbidden)
    assert batch.run(project,work,stage)==1
    assert "gated" in json.loads((work/"quantum_gpu_batch_status.json").read_text())["error"]


@pytest.mark.parametrize("fail_seed", [None, 2109])
def test_five_seed_driver_preserves_examples_resumes_and_stops_on_failure(monkeypatch, tmp_path, fail_seed):
    monkeypatch.setattr(gpu, "OUT", tmp_path)
    monkeypatch.setattr(gpu.cpu, "summary", lambda: None)
    prior = gpu.cpu.OUT, gpu.cpu.make_model, gpu.cpu.read_checkpoint
    directory = tmp_path/"quantum/seed_2108"
    directory.mkdir(parents=True)
    (directory/"last.pt").write_bytes(b"synthetic pending checkpoint")
    x, tables, day, spec, preprocessing = object(), object(), object(), {}, object()
    calls = []
    def training(kind, seed, resume, actual_x, actual_tables, actual_day, actual_spec, actual_preprocessing):
        assert kind=="quantum" and actual_x is x and actual_tables is tables
        assert actual_day is day and actual_preprocessing is preprocessing
        assert actual_spec is spec and resume==(seed==2108)
        calls.append(seed)
        if seed==fail_seed:
            raise RuntimeError("synthetic interruption")
    monkeypatch.setattr(gpu.cpu, "train", training)
    if fail_seed is None:
        gpu.train_all(x,tables,day,spec,preprocessing)
        assert calls==gpu.cpu.SEEDS
    else:
        with pytest.raises(RuntimeError, match="interruption"):
            gpu.train_all(x,tables,day,spec,preprocessing)
        assert calls==[2107,2108,2109]
    assert (gpu.cpu.OUT,gpu.cpu.make_model,gpu.cpu.read_checkpoint)==prior
