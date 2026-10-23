"""Separate three-seed native-batched CPU correction experiment (batch 256)."""
import argparse
import json
from pathlib import Path
import random
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import train_stage2_neural as neural
from stage2_shared import make_model, read_checkpoint, compare, broadcast_evidence, preserve_plan

OUT = Path("outputs/2107_stage2_quantum_cpu")
SEEDS = [2107,2108,2109]
SETTINGS = dict(neural.SETTINGS,batch_size=256,backend="default.qubit",differentiation="backprop")


def prepare_data():
    reference_path = Path("classical_reference_settings.json")
    if not reference_path.exists():
        reference_path = neural.OUT/"settings.json"
    reference = json.loads(reference_path.read_text())
    for name, expected in reference["source_hashes"].items():
        source = Path("scripts")/name.replace("\\","/").rsplit("/",1)[-1]
        if neural.digest(source)!=expected:
            raise ValueError(f"Classical source provenance differs: {source}")
    x,tables,day,spec,preprocessing = neural.prepare_data()
    for key in ("settings","split","features","dimension","counts","daylight_n","target_scale_kw",
                "frame_fingerprints","preprocessing_sha256","versions"):
        if neural.canonical(spec[key])!=neural.canonical(reference[key]):
            raise ValueError(f"Completed classical reference differs: {key}")
    # Relative source identities allow the friend to resume after moving the checkout.
    sources = [Path("scripts")/n.replace("\\","/").rsplit("/",1)[-1] for n in reference["source_hashes"]]
    sources += [Path(__file__),Path("scripts/stage2_shared.py")]
    spec.update(architecture="stage2-native-batched-cpu-v1",settings=SETTINGS,seeds=SEEDS,
                classical_reference_commit=reference["source_commit"],
                source_hashes={"scripts/"+p.name:neural.digest(p) for p in sources},
                classical_reference_sha256=neural.digest(reference_path),
                change="Batch 256 replaces 128; default.qubit/backprop replaces row/adjoint; accuracy equivalence is not assumed")
    return x,tables,day,spec,preprocessing


def inference(model,x):
    import torch
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(x[i:i+256])).numpy() for i in range(0,len(x),256)])


def train(seed,resume,x,tables,day,spec,preprocessing):
    import torch
    torch.set_default_dtype(torch.float64)
    spec = preserve_plan(OUT/"settings.json", spec)
    run_spec = dict(spec,model="quantum",seed=seed)
    directory = OUT/"quantum"/f"seed_{seed}"
    directory.mkdir(parents=True,exist_ok=True)
    neural.freeze(OUT/"settings.json",spec)
    neural.freeze(directory/"settings.json",run_spec)
    last,best = directory/"last.pt",directory/"best.pt"
    if resume!=last.exists():
        raise ValueError("Use --resume for an existing checkpoint; omit it for a new seed")
    if not (OUT/"preprocessing.joblib").exists():
        joblib.dump(preprocessing,OUT/"preprocessing.joblib")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = make_model(spec["dimension"],seed,"default.qubit",batched=True)
    neural.gradient_check(model,spec["dimension"])
    optimizer = torch.optim.Adam(model.parameters(),lr=spec["settings"]["learning_rate"])
    generator = torch.Generator().manual_seed(seed)
    history,epoch,best_mae,stale = [],0,float("inf"),0
    if resume:
        state = read_checkpoint(last,run_spec)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history,epoch,best_mae,stale = state["history"],state["epoch"],state["best_mae"],state["stale"]
        generator.set_state(state["shuffle_rng"])
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if (directory/"metrics.json").exists():
            result = json.loads((directory/"metrics.json").read_text())
            best_state = read_checkpoint(best,run_spec)
            if (not (epoch>=spec["settings"]["epochs"] or stale>=spec["settings"]["patience"])
                    or result["stopping_epoch"]!=epoch or result["best_epoch"]!=best_state["epoch"]
                    or not np.isclose(result["daylight_mae_kw"],best_state["best_mae"],rtol=1e-12,atol=1e-9)
                    or not (directory/"validation_predictions.parquet").is_file()):
                raise ValueError("Inconsistent completed seed")
            print(seed,"already complete; skipped",flush=True)
            return
    xt = torch.from_numpy(x["training"])
    yt = torch.from_numpy(tables["training"].correction_kw.to_numpy()/spec["target_scale_kw"])
    actual = tables["validation"].actual_target_power_kw.to_numpy()
    physics = tables["validation"].physics_forecast_kw.to_numpy()
    loss_fn = torch.nn.HuberLoss(delta=spec["settings"]["huber_delta"])
    while epoch<spec["settings"]["epochs"] and stale<spec["settings"]["patience"]:
        begin = time.perf_counter()
        model.train()
        order = torch.randperm(len(xt),generator=generator)
        total_loss = 0.
        for batch in order.split(spec["settings"]["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xt[batch]),yt[batch])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*len(batch)
        prediction = neural.final_power(physics,inference(model,x["validation"]),spec["target_scale_kw"])
        scores = neural.metrics(actual,prediction,day)
        if not all(np.isfinite(v) for v in scores.values()):
            raise ValueError("Nonfinite validation metric")
        improved = best_mae-scores["daylight_mae_kw"]>=spec["settings"]["min_delta_kw"]
        best_mae = scores["daylight_mae_kw"] if improved else best_mae
        stale = 0 if improved else stale+1
        epoch += 1
        history.append(dict(epoch=epoch,normalized_huber=total_loss/len(xt),**scores,seconds=time.perf_counter()-begin))
        state = dict(specification=run_spec,model=model.state_dict(),optimizer=optimizer.state_dict(),
                     epoch=epoch,best_mae=best_mae,stale=stale,history=history,
                     shuffle_rng=generator.get_state(),torch_rng=torch.get_rng_state(),
                     numpy_rng=np.random.get_state(),python_rng=random.getstate())
        if improved:
            neural.checkpoint(best,state)
        neural.checkpoint(last,state)
        pd.DataFrame(history).to_csv(directory/"history.csv",index=False)
        print(seed,history[-1],flush=True)
    best_state = read_checkpoint(best,run_spec)
    model.load_state_dict(best_state["model"])
    correction = inference(model,x["validation"])*spec["target_scale_kw"]
    saved = tables["validation"][["target_timestamp","actual_target_power_kw"]].copy()
    saved["predicted_correction_kw"] = correction
    saved["final_power_kw"] = physics+correction
    saved.to_parquet(directory/"validation_predictions.parquet")
    pd.DataFrame(history).to_csv(directory/"history.csv",index=False)
    result = dict(**neural.metrics(actual,physics+correction,day),model="quantum",seed=seed,stopping_epoch=epoch,
                  best_epoch=best_state["epoch"],training_validation_seconds=sum(h["seconds"] for h in history),
                  label="Validation results; batch 256; no seed selection or ensemble; no test evaluation")
    (directory/"metrics.json").write_text(json.dumps(result,indent=2))


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare","train","summary"):
        mode.add_argument("--"+name,action="store_true")
    parser.add_argument("--seed",type=int,choices=SEEDS)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--accept-runtime",action="store_true",help="Manual approval after ML/DL test review and runtime review")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float64)
    if args.summary:
        previous = neural.OUT
        try:
            neural.OUT = OUT
            neural.summary()
        finally:
            neural.OUT = previous
        return
    if args.train and (args.seed is None or not args.accept_runtime):
        parser.error("Training requires --seed and explicit --accept-runtime; first complete/review ML/DL testing")
    x,tables,day,spec,preprocessing = prepare_data()
    if args.prepare:
        sample = torch.from_numpy(x["training"][:3])
        target = torch.from_numpy(tables["training"].correction_kw.to_numpy()[:3]/spec["target_scale_kw"])
        candidate = make_model(spec["dimension"],2107,"default.qubit",batched=True)
        check = compare(make_model(spec["dimension"],2107,"lightning.qubit"),candidate,sample,target)
        evidence = broadcast_evidence(candidate,sample)
        print(json.dumps(dict(dimension=spec["dimension"],parameters=89,counts=spec["counts"],daylight_n=spec["daylight_n"],
                              seeds=SEEDS,settings=SETTINGS,equivalence=check,native_broadcast=evidence,change=spec["change"]),indent=2))
        return
    train(args.seed,args.resume,x,tables,day,spec,preprocessing)


if __name__=="__main__":
    with threadpool_limits(limits=4):
        main()
