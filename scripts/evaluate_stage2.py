"""Freeze selected correction artifacts, then evaluate the shared test split once."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

import train_stage2_neural as neural
from train_stage2_quantum_gpu import read_checkpoint
from compare_stage2_correction import FEATURES, OUT as CORRECTION, canonical, examples, metrics
from forecast_stage2_weather import SPLIT, DATA, OUT as WEATHER, VARIABLES, split_masks
from calculate_stage2_physics import complete_fleet_power
from audit_stage2_readiness import digest, indexed, solar_elevation
from oedi2107.real_data import discover_channels

OUT = Path("outputs/2107_stage2_test")


def frozen_artifacts():
    reference = json.loads((neural.OUT/"settings.json").read_text())
    if reference["split"]!=SPLIT or reference["features"]!=FEATURES:
        raise ValueError("Classical reference split/features differ")
    for name, expected in reference["source_hashes"].items():
        source = Path("scripts")/name.replace("\\","/").rsplit("/",1)[-1]
        if digest(source)!=expected:
            raise ValueError(f"Completed classical source changed: {source}")
    if neural.dependencies()!=reference["versions"]:
        raise ValueError("Use the completed classical environment versions")
    hpo = json.loads((CORRECTION/"hpo/settings.json").read_text())
    if hpo["selected_trial"]!="forest_05" or hpo["plan"]["split"]!=SPLIT:
        raise ValueError("Expected the frozen forest_05 selection")
    candidate = next(c for c in hpo["plan"]["candidates"] if c["id"]=="forest_05")
    paths = [neural.OUT/"settings.json",CORRECTION/"hpo/settings.json",CORRECTION/"hpo/best_forest.joblib",
             CORRECTION/"ridge.joblib",WEATHER/"validation_models.joblib",WEATHER/"validation_settings.json",
             WEATHER/"training_1_settings.json",Path(__file__),Path("scripts/train_stage2_quantum_gpu.py")]
    if digest(CORRECTION/"ridge.joblib")!=reference["preprocessing_sha256"]:
        raise ValueError("Training-fitted neural preprocessing differs")
    forest = joblib.load(CORRECTION/"hpo/best_forest.joblib")
    if (forest["trial"]["id"]!="forest_05" or forest["features"]!=FEATURES
            or canonical(forest["model"].get_params())!=canonical(candidate["effective_settings"])):
        raise ValueError("Winning forest bundle/configuration differs")
    weather = joblib.load(WEATHER/"validation_models.joblib")
    if set(weather)!=set(VARIABLES):
        raise ValueError("Incomplete weather models: testing must never fit missing variables")
    for seed in neural.SEEDS:
        directory = neural.OUT/"classical"/f"seed_{seed}"
        spec = dict(reference,model="classical",seed=seed)
        if canonical(json.loads((directory/"settings.json").read_text()))!=canonical(spec):
            raise ValueError(f"Incompatible classical seed {seed}")
        best = read_checkpoint(directory/"best.pt",spec,parameters=85)
        last = read_checkpoint(directory/"last.pt",spec,parameters=85)
        result = json.loads((directory/"metrics.json").read_text())
        if (not (last["epoch"]>=100 or last["stale"]>=10)
                or result["stopping_epoch"]!=last["epoch"] or result["best_epoch"]!=best["epoch"]
                or not np.isclose(result["daylight_mae_kw"],best["best_mae"],rtol=1e-12,atol=1e-9)):
            raise ValueError(f"Classical seed {seed} did not finish consistently")
        paths.extend(directory/n for n in ("settings.json","metrics.json","best.pt","last.pt"))
    paths.extend(DATA/f"2107_{n}.parquet" for n in ("electrical_clean","electrical_qc_events","irradiance_clean","irradiance_qc_events"))
    return dict(protocol="stage2-frozen-test-v1",split=SPLIT,features=FEATURES,selected_trial="forest_05",
                seeds=neural.SEEDS,neural_reference=reference,
                source_hashes={p.as_posix():digest(p) for p in paths},
                selection="Frozen validation selections; no fitting, tuning, seed selection or test-driven changes",
                scoring="Identical eligible rows for all models; target elevation>0 daylight; bias=prediction-actual; no clipping",
                disclosure="Parts of this chronological test period appeared in earlier exploratory work; held out from these model fits, not entirely unseen")


def load_testing_examples():
    weather = pd.read_parquet(WEATHER/"forecasts_testing.parquet")
    table = pd.read_parquet(WEATHER/"physics_testing.parquet")
    for step in ("testing_settings.json","physics_testing_settings.json"):
        saved = json.loads((WEATHER/step).read_text())
        spec = saved.get("specification",saved)
        if spec["split"]!=SPLIT or spec["protocol"]!="elapsed-70-15-15-v1" or spec["step"]!="testing":
            raise ValueError("Incompatible test input protocol")
        for name, expected in spec["source_hashes"].items():
            if digest(Path(name))!=expected:
                raise ValueError(f"Test input provenance differs: {name}")
    if not table.index.equals(weather.index) or not table.index.is_unique:
        raise ValueError("Physics/weather origins differ or repeat")
    shared = ["target_timestamp","training_cutoff","forecast_poa","forecast_ambient","forecast_wind"]
    pd.testing.assert_frame_equal(table[shared],weather[shared],check_dtype=False)
    if (not split_masks(table.index)["testing"].all() or not table["split"].eq("testing").all()
            or not table.training_cutoff.eq(pd.Timestamp(SPLIT["validation"])).all()):
        raise ValueError("Test targets or frozen weather cutoff differ")
    lower = str(table.index.min()-pd.Timedelta(minutes=15))
    upper = str(pd.Timestamp(table.target_timestamp.max())+pd.Timedelta(minutes=5))
    filters = [("measured_on",">=",lower),("measured_on","<",upper)]
    source = DATA/"2107_electrical_clean.parquet"
    mapping = discover_channels(pq.ParquetFile(source).schema_arrow.names)
    ac = indexed(source,["measured_on",*mapping.ac_power],filters=filters)
    flags = indexed(DATA/"2107_electrical_qc_events.parquet",["measured_on"],filters=filters).index
    plant = complete_fleet_power(ac,flags)
    np.testing.assert_allclose(table.actual_target_power_kw,plant.reindex(table.target_timestamp).to_numpy(),equal_nan=True,rtol=0,atol=1e-9)
    poa = indexed(DATA/"2107_irradiance_clean.parquet",filters=filters).iloc[:,0]
    flags = indexed(DATA/"2107_irradiance_qc_events.parquet",["measured_on"],filters=filters).index
    local = poa.index.tz_localize("America/Los_Angeles",ambiguous="NaT",nonexistent="NaT")
    poa = poa.where(np.isfinite(poa)&poa.ge(0)&~poa.index.isin(flags)&~local.isna())
    features,table,counts = examples(table,plant,poa)
    day = solar_elevation(pd.DatetimeIndex(table.target_timestamp)).to_numpy()>0
    return features,table,day,counts


def evaluate(plan):
    features,table,day,counts = load_testing_examples()
    predictions = table[["target_timestamp","actual_target_power_kw","physics_forecast_kw"]].copy()
    predictions["persistence_kw"] = features.origin_power_kw
    forest = joblib.load(CORRECTION/"hpo/best_forest.joblib")
    predictions["forest_05"] = table.physics_forecast_kw+forest["model"].predict(forest["processor"].transform(features))
    bundle = joblib.load(CORRECTION/"ridge.joblib")
    transformed = bundle["processor"].transform(features)
    x = np.ascontiguousarray(np.column_stack([bundle["scaler"].transform(transformed)[:,:len(FEATURES)],
                                              transformed[:,len(FEATURES):]]),dtype=np.float64)
    reference = plan["neural_reference"]
    if x.shape[1]!=reference["dimension"] or not np.isfinite(x).all():
        raise ValueError("Invalid transformed neural inputs")
    for seed in neural.SEEDS:
        spec = dict(reference,model="classical",seed=seed)
        state = read_checkpoint(neural.OUT/"classical"/f"seed_{seed}"/"best.pt",spec,parameters=85)
        model = neural.make_model("classical",reference["dimension"],seed)
        model.load_state_dict(state["model"])
        predictions[f"classical_{seed}"] = neural.final_power(table.physics_forecast_kw,
            neural.inference(model,x),reference["target_scale_kw"])
    columns = ["physics_forecast_kw","persistence_kw","forest_05",*[f"classical_{s}" for s in neural.SEEDS]]
    scores = score_predictions(predictions,columns,day)
    # Recheck frozen inputs before publishing any evaluation artifacts.
    if frozen_artifacts()!=plan:
        raise ValueError("Frozen evaluation artifacts changed during inference")
    predictions.to_parquet(OUT/"predictions.parquet")
    scores.to_csv(OUT/"metrics.csv")
    print(json.dumps(counts,indent=2),flush=True)
    print(scores.to_string(),flush=True)
    print(scores.loc[[f"classical_{s}" for s in neural.SEEDS]].agg(["mean","std"]).to_string())
    print(plan["disclosure"],flush=True)


def score_predictions(predictions,columns,day):
    actual = predictions.actual_target_power_kw.to_numpy()
    if not np.isfinite(predictions[["actual_target_power_kw",*columns]]).all().all() or not np.any(day):
        raise ValueError("All models must have finite predictions on the same eligible rows, including daylight")
    scores = {}
    for name in columns:
        error = predictions[name].to_numpy()-actual
        scores[name] = dict(**metrics(actual,predictions[name].to_numpy(),day),
                           all_day_rmse_kw=float(np.sqrt(np.mean(error**2))),all_day_bias_kw=float(error.mean()))
    return pd.DataFrame.from_dict(scores,orient="index").rename_axis("model")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare",action="store_true")
    mode.add_argument("--evaluate",action="store_true")
    parser.add_argument("--allow-test",action="store_true")
    args = parser.parse_args()
    if args.evaluate and not args.allow_test:
        parser.error("Evaluation requires --allow-test after reviewing the frozen plan")
    plan = frozen_artifacts()
    OUT.mkdir(exist_ok=True)
    settings = OUT/"settings.json"
    if args.prepare:
        neural.freeze(settings,plan)
        print("Frozen forest_05 + five completed classical seeds; no test rows read or predictions made.")
        print("Pending test inputs:",*[n for n in ("forecasts_testing.parquet","physics_testing.parquet") if not (WEATHER/n).exists()])
        print(plan["disclosure"])
        return
    if not settings.exists() or canonical(json.loads(settings.read_text()))!=canonical(plan):
        raise ValueError("Run --prepare first; frozen models/settings must match exactly")
    if (OUT/"predictions.parquet").exists() or (OUT/"metrics.csv").exists():
        raise ValueError("Existing test results; refusing automatic repeated evaluation or overwrite")
    evaluate(plan)


if __name__=="__main__":
    with threadpool_limits(limits=4):
        main()
