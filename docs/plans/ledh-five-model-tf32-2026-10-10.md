# Five-model TF32 likelihood and score evaluation

The owner requests a 1% covariance-restoration tolerance and evaluation of the
final likelihood and analytical score for LGSSM, KSC SV, predator–prey, SIR and
range bearing. This is a precision evaluation of the current covariance-guided
defensive-mixture filter, a prerequisite to the original surrogate-HMC question.

## Evidence contract and research intent

Compare FP64 (TF32 off), ordinary FP32 (TF32 off), and FP32 with TF32 enabled.
Use identical FP32-representable observations, parameters, mixture weights and
random designs within each comparison. FP64 is the comparator for arithmetic
error, not an exact nonlinear filtering oracle. Preserve the complete final
log-likelihood and every analytical score coordinate, validity and step count.

Use N1008/T50 for the first four models, their two saved independent datasets
26100831/26100832, and eight designs per dataset. LGSSM is the saved P44 d3
model, KSC d1, predator–prey d2, and SIR d18. For range bearing use T20/d4,
N1008/N2016/N4032, data seeds 1704/18020/18021, truth and the existing difficult
parameter point, and eight designs per case. These are the previous measured
scopes; no cross-model or cross-N tuning transfer is permitted. Preserve saved
scope-specific controls as frozen diagnostic comparators. They do not become
new FP32/TF32 tuning-admission artifacts.

Change only the relative Frobenius covariance-restoration check for float32
from 100*eps*d to 0.01. FP64 retains its original stricter check. The mean,
finite-value, full-horizon, factorization, rank and correction-cap guards remain
active. Record the selected covariance tolerance and observed residuals. This
changes rejection policy, not the value or score arithmetic on accepted paths.
Test the actual consumer with an injected covariance perturbation below and
above 1%, including the surviving finite/rank guards. Run the existing focused
recursive-score and reset regressions before the campaign.

The primary numerical result is the paired final arithmetic discrepancy, by
model and dataset/parameter point. Report absolute differences, RMS and maxima,
coordinate differences, relative score RMS, and RMS discrepancy relative to
FP64 design spread. For an explicit diagnostic materiality screen require
likelihood RMS <= max(0.01, 0.1*FP64 likelihood SD) and score RMS <=
max(0.01, 0.1*FP64 RMS score spread). The 0.01 absolute floors prevent division
by negligible design variation; score units differ by model, so the floors
are declared screening conventions, not posterior-error guarantees. Keep all
numerators and denominators available. Failure triggers a reported precision
weakness; it does not justify further tolerance relaxation or stop other models.

Range-bearing additionally reports discrepancy relative to the earlier matched
N1008-to-N2016 change. Saved independent Kalman/grid/particle references may
provide explanatory context only; do not call a stale rounded-input reference
an exact oracle. This study cannot rank the filtering method against other
algorithms or establish HMC convergence, posterior agreement or default readiness.

## Diagnostic roles and stop conditions

- Promotion criterion for this bounded precision screen: complete final
  likelihood/scores and the two declared RMS materiality bounds in every case.
- Promotion veto: any invalid/incomplete evaluation, nonfinite output, changed
  inputs/controls, or material final discrepancy. A rejected precision remains
  a research result; continue independent models and other arithmetic modes.
- Continuation veto: corrupted inputs, source changes during execution, broken
  pairing, unavailable GPU, budget exhaustion, or persistent infrastructure
  failure. Repair local harness faults within the unchanged budget and preserve
  each attempt in a fresh output directory.
- Explanatory only: intermediate covariance residuals below 1%, ESS, moments,
  cold/steady timing, maximum errors and the limited-design spread estimates.
- Repair trigger: localized harness/serialization failure; TF32 numerical
  failures are reported separately, with no silent algorithm replacement.

## Default and assumption audit; skeptical review

| Choice | Provenance and status | Failure mode and earliest check |
|---|---|---|
| 1% covariance tolerance | Explicit owner choice; rejection-policy change | Accumulation could materially distort final scores; measure complete paired outputs |
| Saved controls and beta | Per-scope FP64 tuning; diagnostic baseline | Precision changes best controls; no retuning/default claim |
| Common FP32 inputs | Previous precision diagnostic; reviewed pairing rule | RNG/dtype changes masquerade as arithmetic error; hash every input |
| Two datasets/eight designs, range six cases | Saved campaign coverage | Missed parameter regions/tails; conditional tables, descriptive claims only |
| Materiality bounds and absolute floors | Predeclared practical hypothesis | Floors hide small-unit effects; publish raw coordinates and scales |
| GPU/XLA/chunks | Repository policy and existing checked kernel | Wrong backend, TF32 flag or incomplete fast exit; manifest and full-step checks |
| Existing guards | Checked factorization/mean/cap validity | Other guards can still reject; retain per-step diagnostics and do not mislabel rejection as convergence failure |

Skeptical audit passes after addressing two risks: covariance acceptance is not
treated as final accuracy, and saved FP64 tuning is not relabelled as TF32
tuning. Every model calls the same make_filter implementation. Known cheap
filtering adversaries (prior/bootstrap, Gaussian guide, exact Kalman where
available) remain in earlier conditional studies; they are not rerun for an
arithmetic comparison. Their previous weaknesses continue to veto broad
algorithm promotion. No reference/heuristic outcome is a tuning target here.

Pre-mortem: the command could succeed with inaccurate accumulated scores,
different random numbers, unequal parameter rounding, or timings of failed
prefixes. Complete paired records, serialized-input hashes and step checks
address those risks before interpretation. A final failure can reflect
precision sensitivity without refuting the filtering algorithm.

## Execution and artifacts

Use /home/chakwong/anaconda3/envs/tftwogpu/bin/python, trusted GPU/XLA on the
RTX4080 SUPER, verified memory growth and stable signatures. Input generation
may use CPU as an explicitly recorded preparation step; numerical filtering
uses GPU. Ordinary FP32 and FP64 are precision-reference exceptions to the
repository TF32 direction. No package changes, network use or external agents.

Budget: at most 7200 seconds of serial GPU worker wall time, 21 planned
workers (seven model/N scopes times three precisions), with up to two localized
infrastructure retries within the same budget. Focused CPU engineering tests
have a separate 600-second cap. Each launch has a unique output root.

Command: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/five_model_precision_diagnostic.py run --output docs/plans/artifacts/ledh-five-model-tf32-20261010-01`.
The same script provides `status` and `report` without GPU imports. Run manifests
record commit, dirty-source hashes, command, environment, GPU/memory/TF32/XLA,
input/tuning hashes, seeds, timings, plan and result file. Structured summaries
preserve a decision and inference-status table. Result note:
`docs/plans/ledh-five-model-tf32-results-2026-10-10.md`.
