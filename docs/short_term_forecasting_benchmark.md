# Short-term complete-fleet AC forecasting

**Verdict:** one modest boosted model reliably beats persistence on the supported daytime evaluation record at **5, 15 and 30 minutes**, including each season and rapid changes. It also improves on regularized linear regression. **Zero-output/shutdown forecasting is not reliably improved.** This is one site benchmark, with no novelty or deployment claim. The inconclusive decline findings are preserved.

## Frozen benchmark and availability

[Specification](../outputs/2107_short_term_forecasting/benchmark_specification.json) was written before fitting. Train on **2018–2020**, report **2021 validation without tuning**, evaluate **2022–2023**, through the available weather overlap on **2023-10-31**. Later years were already explored in readiness/decline work: they are **held out from forecasting model fitting, not an entirely unseen dataset**.

Target: strict simultaneous sum of **all 24 inverter AC readings**, in kW. Missing/negative/previously flagged electrical readings are excluded; zero readings are retained. All inverter observations are combined before splitting; units are not independent plants. No missing targets are filled.

Origin-only predictors: AC/POA at t, t-5, t-15, t-30; their 5/15/30-minute changes; origin fraction of units recording zero; historical ambient/wind; cyclic hour/annual phase; deterministic origin/target solar elevation. Require all seven five-minute AC/POA labels in the 30-minute lookback to be valid. Electrical targets and entire lookbacks must remain in the same split. No future measured POA/weather/electrical values, physics power or calculated temperature enter predictors. Future actual power only supplies labels and reporting slices.

Persistence uses origin AC. Ridge uses training-only StandardScaler and alpha=100. Histogram boosting uses <=15 leaves, depth <=4, minimum leaf 100, learning rate .06, L2=10, 150 iterations and seed 2107. One configuration per family is fitted once per horizon; both learn future-minus-origin power and bound final forecasts below at zero. No refitting or model collection. Identical valid origin/target pairs compare all three models at each horizon.

**Availability assumption:** t-labelled AC/POA are available at origin t. Environment uses the latest label <=t-15 minutes, at most 30 minutes old, assumed available 15 minutes after its label. This is a backward lookup, not the earlier forward weather alignment. Actual logger delay and interval-label semantics remain unverified; especially at five minutes, these are retrospective label-time forecasts, not proven operational lead times.

Daytime means **origin and target solar elevation >10 degrees**, with no future measured-sunlight filter. Cloudy/low-POA and eligible zero periods remain; nighttime zeros cannot dominate. Training has 126,789/124,699/121,567 observations for 5/15/30 minutes; validation 43,152/42,441/41,378; evaluation **79,227/77,955/76,046 on 641 dates**. Missing-history/weather episodes remain outside this supported cohort.

## Evaluation results

Bias is prediction minus actual; improvement is MAE reduction versus persistence.

| Ahead | Model | MAE kW | RMSE kW | Bias kW | MAE improvement |
|---|---|---:|---:|---:|---:|
| 5 min | Persistence | 28.46 | 63.37 | +0.19 | — |
| | Ridge | 27.10 | 57.57 | -0.47 | 4.8% |
| | Boosting | **25.38** | **56.29** | -0.70 | **10.8% [8.4, 13.3]** |
| 15 min | Persistence | 49.05 | 85.32 | +0.64 | — |
| | Ridge | 38.86 | 72.59 | -1.05 | 20.8% |
| | Boosting | **35.50** | **70.43** | -1.40 | **27.6% [23.9, 31.5]** |
| 30 min | Persistence | 73.65 | 108.37 | +1.41 | — |
| | Ridge | 49.15 | 83.82 | -1.89 | 33.3% |
| | Boosting | **43.04** | **79.47** | -2.19 | **41.6% [37.3, 45.8]** |

Brackets are 95% paired bootstrap intervals: **2,000 draws of 95 whole calendar-week date blocks**, retaining within-date/site dependence and pairing the same observations for every comparison. Boosting RMSE reductions are **11.2% [10.4,11.9]**, **17.5% [16.4,18.7]**, **26.7% [24.8,29.0]**. Its MAE reductions versus Ridge are **6.4% [5.4,7.5]**, **8.7% [7.2,10.3]**, **12.4% [10.7,14.5]**. [All pairwise comparisons](../outputs/2107_short_term_forecasting/pairwise_comparisons.csv) include Ridge's intervals. Conditional uncertainty excludes fitting/measurement-latency uncertainty and external-site variation; these are not 24 independent plants.

Validation boosting MAE reductions are **10.0%, 26.8%, 39.5%**; evaluation gains appear separately in both 2022 and 2023.

**Rapid changes:** reporting-only slice |future-origin AC|>=100 kW. There are 4,632/7,877/14,269 observations. Persistence/Ridge/boosting MAE is respectively **215.5/181.4/174.7**, **215.4/172.1/163.6**, **194.1/132.5/119.5 kW**. Boosting improvements are **18.9% [18.1,19.7]**, **24.0% [23.3,24.9]**, **38.4% [36.5,40.8]**. Absolute ramp errors remain large; these future-defined slices cannot be known from origin data.

| Season | Boosting MAE improvement, 5 min | 15 min | 30 min |
|---|---:|---:|---:|
| DJF | 5.1% [2.2,8.5] | 16.4% [11.2,22.5] | 27.9% [21.4,35.0] |
| MAM | 6.3% [3.0,10.4] | 22.1% [16.8,28.6] | 35.7% [29.1,43.2] |
| JJA | 18.9% [15.2,23.0] | 40.2% [34.1,46.2] | 54.9% [48.7,60.5] |
| SON | 14.4% [10.3,20.5] | 31.8% [25.0,41.0] | 45.7% [37.7,55.4] |

Ridge is weaker than persistence in five-minute winter forecasts (-5.8%, interval -9.2 to -2.3%). [Full metrics](../outputs/2107_short_term_forecasting/metrics.csv) give MAE, RMSE, bias, counts and improvement for **every model**, season, rapid-change, year and zero/generating slice.

## Zero versus generating operation

Generating here means **fleet target >0**, including periods when individual units record zero. All-fleet recorded-zero means all 24 readings are zero; it is not missingness or a confirmed fault label.

| Target state / ahead | Persistence MAE | Ridge MAE | Boosting MAE | Boosting RMSE / bias, kW |
|---|---:|---:|---:|---:|
| Generating / 5 min | 28.41 | 27.04 | 25.32 | 56.00 / -0.77 |
| Generating / 15 min | 48.91 | 38.73 | 35.36 | 69.83 / -1.56 |
| Generating / 30 min | 73.42 | 48.91 | 42.80 | 78.57 / -2.45 |
| Recorded zero / 5 min | 307.99 | 359.25 | 357.59 | 420.98 / +357.59 |
| Recorded zero / 15 min | 462.55 | 472.16 | 476.82 | 515.52 / +476.82 |
| Recorded zero / 30 min | 513.88 | 508.25 | 509.16 | 534.71 / +509.16 |

Evaluation has only **15/25/39 zero targets on nine dates**; these largely test abrupt transitions, not sustained shutdowns. Boosting's zero-target MAE changes are **-16.1% [-31.6,-3.5]**, **-3.1% [-12.7,7.4]**, **+0.9% [-6.6,8.8]**. There are only seven zero origins on three evaluation dates: too little evidence for reliable comparison. Undefined relative bootstrap gains when persistence error is zero are omitted and their fraction reported; absolute error-saving intervals remain available.

The 2021 validation record supplies a stronger limitation: **3,899/3,923/3,953 zero targets**. Persistence versus boosting MAE is **5.35/12.50**, **15.95/28.71**, **29.12/52.49 kW**. Boosting also loses to persistence on sustained zero origins. Consequently the positive overall benchmark does **not** establish useful shutdown/restart forecasting, and weather-missing common-zero episodes cannot be scored without inventing inputs.

## Artifacts and reproduction

- [Daytime MAE/RMSE plot](../outputs/2107_short_term_forecasting/daytime_errors.png).
- [Season/ramp/zero improvement plot](../outputs/2107_short_term_forecasting/slice_improvements.png).
- [Code](../scripts/benchmark_short_term_forecasting.py), [focused tests](../tests/test_short_term_forecasting.py), [support](../outputs/2107_short_term_forecasting/support.csv), [provenance](../outputs/2107_short_term_forecasting/provenance.json).

```powershell
.\.venv\Scripts\python -B scripts/benchmark_short_term_forecasting.py
.\.venv\Scripts\python -B -m pytest tests/test_short_term_forecasting.py -q -p no:cacheprovider
```

Uses the repository environment with scikit-learn **1.7.2**; exact library versions are in provenance. Saved forecasts/models allow scoring without fitting again. Five tests pass: split/gap safeguards, future-measurement invariance, paired-block arithmetic, undefined zero-baseline percentages and correct pairwise references. The scoring correction reuses unchanged forecasts/models. Protected hashes verify used data, all 21 frozen baseline files, and every saved decline-experiment file/report unchanged. Equipment assumptions were not revisited.
