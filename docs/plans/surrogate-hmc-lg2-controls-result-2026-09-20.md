# LG2 control selection and measured inference cost

LG2 selected protection multiplier **0.01** on separate calibration and
validation data. Retained inference was deferred: the measured forecast for
one B arm alone is **4.035 GPU hours and 3.744 CPU-worker hours**, exceeding the
initial allocation of 2 GPU hours and 0.5 CPU-worker hours. This is a budget
finding, not evidence against surrogate HMC. No LG2 tuning or posterior look
was run, and no claim data were used to select controls.

The target remains the two-parameter LG2 model at T=20, N=K=1024, attempt 1,
replicate 0. The actual consumer evaluates the full canonical analytical DCTR
force in FP32/TF32 GPU/XLA and the independent matrix Kalman endpoint in FP64.
The [plan](surrogate-hmc-lg2-controls-2026-09-20.md) states the selection rule,
diagnostic roles, assumptions, timing screen and budget before execution.

All eight control arms preserved valid original statuses and finite nominal
and boundary-stress rows. The table gives each adjacent pair's largest nominal
change across value, force and filter/reset moments, normalized by
`1 + abs(weaker)`. The frozen non-harm tolerance is 0.001; stress checks also
passed for every pair.

| Stronger / weaker multiplier | Calibration | Validation | Shared pass |
|---|---:|---:|---|
| 1 / 0.1 | 0.00626175 | 0.00770317 | No |
| 0.1 / 0.01 | 0.00216958 | 0.00063674 | No |
| 0.01 / 0.001 | 0.00044640 | 0.00043052 | Yes |

The first shared pair selects 0.01. Its reset ridge is 1e-7, correction damping
1e-5, scale floor 1e-8, trust radius 10, pairwise cap 200 and coordinate cap
1600. The selection grants experimental use as an imperfect proposal field.
It does not establish exact-score admission, posterior recovery, or a default
for LG2 or any other model. The remaining fixed controls are still hypotheses.

The complete four-chain cost probe used one leapfrog step, step size 0.02 and
identity mass on calibration data. Compilation took 27.280 wall seconds;
the four subsequent transitions took 3.102, 3.224, 2.980 and 3.741 seconds.
All transitions were finite, with no divergence, force fallback or endpoint
failure. All chains accepted these tiny mechanics steps; acceptance says
nothing about convergence or useful sampling efficiency.

The forecast uses the fastest observed wall and CPU costs, 3250 transitions
(250 fresh warm-up, 1000 verification, 2000 retention), and a factor of 1.5.
It is already over allocation at 14525.543 GPU seconds and 13478.532 CPU-worker
seconds. It excludes preparation, nomination/adaptation, recompilation,
analysis, A/C arms and finite-reference construction. This optimistic timing
screen is neither a complete forecast nor a proven hardware lower bound.
Substantial batching or compilation savings could change it; they would need
an unchanged-target parity check and a new complete measurement.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Select 0.01 for proposal experiments | Shared non-harm pair passed | No numerical/status veto | Finite design points; inherited tolerance and other controls | Preserve selection and give other ready families a turn | Exact force, posterior accuracy or default readiness |
| Defer LG2 inference | Initial allocation fails measured cost screen | Budget veto only | Timing variability and omitted costs | Revisit allocation or verified cost repair within the total campaign | Rejection of surrogate HMC |

| Inference status | Evidence |
|---|---|
| Hard veto screen | Calibration and mechanics healthy; inference forecast exceeds initial budget |
| Statistically supported ranking | None |
| Descriptive-only differences | Control changes, transition times and acceptance |
| Default-readiness | Not established |
| Next evidence needed | Complete affordable forecast, public scoped tuning, retained B/R; separately qualified finite target and C/D, D/R, C/R; conditional heuristics |

All four posterior comparisons and heuristic dominance remain `NOT_RUN`.
The strongest alternative explanation for the budget finding is avoidable
consumer/controller overhead. This stage measures the current implementation
and does not rule out a faster equivalent implementation. The weakest
calibration evidence is finite point coverage: healthy rows cannot establish
non-harm throughout the prior. Downstream inference remains necessary.

Execution used the tftwogpu environment, GPU
`GPU-68251639-fe82-8f81-3ccc-2953c32e805b`, verified memory growth and XLA. Peak
TensorFlow allocator usage was 151249664 bytes. The worker exited 0; master
exit 2 records incomplete all-model implementation. Command:

```bash
CUDA_VISIBLE_DEVICES=GPU-68251639-fe82-8f81-3ccc-2953c32e805b TF_FORCE_GPU_ALLOW_GROWTH=true BAYESFILTER_PRELOAD_CUSTOM_OP=0 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage P3-LG2-controls --scope LG2 --reference-root docs/plans/artifacts/surrogate-hmc-all-models-20260920-lg2-reference-01/reference --output docs/plans/artifacts/surrogate-hmc-all-models-20260920-lg2-controls-01
```

The stage consumed 255.179038 GPU seconds and 310.790024 CPU-worker seconds,
including the master. The two focused test runs consumed another 20.164785
CPU seconds and passed 50 tests after a stale test expectation was repaired.
Whole-campaign charges through this calibration are 8115.387729 GPU seconds
and 8439.626857 CPU-worker seconds, within the original 63/15-hour ceiling.
Later master regression and reporting charges are recorded separately.

Evidence: [report](artifacts/surrogate-hmc-all-models-20260920-lg2-controls-01/P3-LG2-controls/report.json),
[selection](artifacts/surrogate-hmc-all-models-20260920-lg2-controls-01/P3-LG2-controls/selection.json),
[cost traces](artifacts/surrogate-hmc-all-models-20260920-lg2-controls-01/P3-LG2-controls/cost-transitions.json),
[run manifest](artifacts/surrogate-hmc-all-models-20260920-lg2-controls-01/run-manifest.json),
[accounting](artifacts/surrogate-hmc-all-models-20260920-lg2-controls-01/accounting.json).
These preserve source snapshots, data identities, seeds, command, device,
status curves and both successful and failed pair checks.
