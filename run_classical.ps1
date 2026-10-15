foreach ($seed in @(2107, 2108, 2109, 2110, 2111)) {
    & .\.venv\Scripts\python -B scripts/train_stage2_neural.py --train --model classical --seed $seed --accept-pilot
    if ($LASTEXITCODE -ne 0) {
        throw "Classical training failed for seed $seed."
    }
}

& .\.venv\Scripts\python -B scripts/train_stage2_neural.py --summary
