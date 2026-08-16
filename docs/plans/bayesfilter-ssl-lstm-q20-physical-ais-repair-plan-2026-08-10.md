# SSL-LSTM q=20 physical AIS weight-repair plan (2026-08-10)

Status: `COMPLETE; SPARSE_AIS_REJECTED; ANNEALED_SMC_TRIGGERED`

## Execution amendment after measured canary

The immutable `r1` canary passed all mechanics checks: XLA compiled, all four
paths and terminal target statuses were valid, mean HMC acceptance was `0.984375`,
and receipt hashes verified.  Its worker runtime was `777.999 s` for 16 bridge
steps, however.  The original material design would require 36 sequential canary
equivalents (`8 * 64/16 + 2 * 32/16`), or approximately `28,008 s`.  This violates
the predeclared `7,200 s` continuation cap.  Full per-bridge rejuvenation is
therefore stopped for compute invalidity; no material samples were opened.

The bounded repair preserves all 64/32 weight bridges but applies HMC every eighth
bridge.  Intermediate bridges use the identity transition, which is exactly
invariant for each bridge law; this changes mixing efficiency, not the AIS target
or weight identity.  Proposal and target values are carried across identity steps,
and every scheduled HMC move still bootstraps fresh results under its actual beta.
Known-law XLA tests pass for identical targets and unequal-weight/unequal-scale
mixtures under sparse rejuvenation.

The `r2` canary uses four paths, 16 bridges, and two HMC moves.  It has the same
mechanics-only gates as `r1`.  The material design remains unopened unless its
measured extrapolation (`8 * t64 + 2 * t32`, using HMC-move count plus measured
fixed overhead) fits `7,200 s`.  Sparse rejuvenation is a new reviewed hypothesis,
not a promoted default; ESS, maximum weight, independent-batch interval, schedule
sensitivity, and terminal movement remain material vetoes.

The `r2` canary subsequently passed in `107.908 s` worker time with all four paths
and terminal statuses valid, mean acceptance `1.0`, and verified receipts.  Scaling
by HMC move count yields approximately `432 s` for each 64-bridge wave and `216 s`
for each 32-bridge wave, or `3,888 s` for all ten waves before aggregation.  This
fits the frozen cap.  The material output root is versioned separately as `r3`.
Each independent batch uses a fresh 25-worker spawn wave, CPUs `0--99` in disjoint
four-core groups, so framework initialization and XLA compilation cannot leak
state between batches.  The measured canary already includes that overhead.

Exact material command/environment:

```bash
systemd-run --user \
  --unit=bayesfilter-ssl-q20-physical-ais-material-20260810-r3 \
  --collect \
  --property=WorkingDirectory=/home/ubuntu/python/BayesFilter \
  --property=CPUQuota=10000% \
  --property=RuntimeMaxSec=7200s \
  --property=TimeoutStopSec=60s \
  --property=StandardOutput=append:/home/ubuntu/python/BayesFilter/docs/plans/artifacts/ssl-lstm-q20-physical-ais-repair-2026-08-10/r3/material.log \
  --property=StandardError=append:/home/ubuntu/python/BayesFilter/docs/plans/artifacts/ssl-lstm-q20-physical-ais-repair-2026-08-10/r3/material.log \
  --setenv=CUDA_VISIBLE_DEVICES=-1 \
  --setenv=TF_CPP_MIN_LOG_LEVEL=3 \
  --setenv=OMP_NUM_THREADS=4 \
  --setenv=OPENBLAS_NUM_THREADS=4 \
  --setenv=MKL_NUM_THREADS=4 \
  --setenv=NUMEXPR_NUM_THREADS=4 \
  --setenv=TF_NUM_INTRAOP_THREADS=4 \
  --setenv=TF_NUM_INTEROP_THREADS=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_physical_ais_material_2026_08_10.py
```

## Research intent ledger

| Field | Declaration |
|---|---|
| Main question | Can annealed importance sampling from the normalized two-local-Gaussian proposal to the exact physical SSL-LSTM target repair the severe direct-importance weight concentration and produce stable evidence for the relative mass of the two known sign regions? |
| Candidate | TensorFlow/TFP linear-schedule AIS with fixed, Metropolis-corrected HMC rejuvenation in the reviewed physical affine chart.  Every bridge re-bootstraps HMC under the new bridge target.  No NUTS. |
| Expected failure | Too few bridge steps or weak HMC movement may leave heavy weights; schedule-specific estimates may disagree; the two-mode proposal may omit another region. |
| Promotion criterion | Every target state valid and finite; known unequal-weight/unequal-scale analytic AIS test passes; central pooled weight ESS fraction `>=0.30`; maximum normalized weight `<=0.02`; eight-batch 95% interval half-width `<=0.08`; 32/64-step schedule estimates differ by `<=0.08`; and at least one terminal sign change from the sampled proposal label in the central run. |
| Promotion veto | Target/status invalidity, non-finite log weights/states/log acceptance, analytic known-law failure, or any primary reliability gate failure. |
| Continuation veto | XLA/harness invalidity after one localized repair, target/source identity drift, receipt failure, or 7,200-second campaign cap.  AIS weight failure triggers SMC rather than a posterior claim. |
| Repair trigger | Low ESS/high max weight/schedule sensitivity triggers annealed SMC with resampling and ancestry diagnostics.  Low HMC final-stage acceptance triggers smaller step before material seeds open. |
| Explanatory diagnostics | Final HMC acceptance, log-weight quantiles/range, per-batch estimates, terminal sign change, runtime, and log normalizer ratio. |
| Must not be concluded | Exhaustive mode discovery, posterior correctness beyond the two known sign regions, transition-chain stationarity, NeuTra repair, default readiness, or predictive validity. |

## Evidence contract

The exact target and physical chart are those validated in the physical global repair
result.  The proposal is the normalized equal-component mixture of the two checked
local Gaussian approximations.  Equal proposal probabilities are sampling design;
AIS weights perform the exact correction.

The primary comparator is the failed direct-importance artifact:

`docs/plans/artifacts/ssl-lstm-q20-physical-global-repair-2026-08-10/r1/weights.json`

AIS must reduce weight concentration and schedule dependence; a point estimate near
the direct-IS mean is not a pass criterion.  The structured output root is:

`docs/plans/artifacts/ssl-lstm-q20-physical-ais-repair-2026-08-10/r1/`

## Numeric/default audit

| Choice | Provenance/status | Justification | Failure mode / early diagnostic |
|---|---|---|---|
| Linear beta schedule | AIS derivation and inspected installed TFP weight formula | Smallest exact AIS baseline; uses incremental `(target-proposal)/num_steps` weights | Linear schedule may allocate steps poorly; 32/64 sensitivity |
| Scheduled HMC bootstrap | Required local correctness repair after installed-source audit | Installed TFP 0.25 driver passes cached `inner_results` between changing bridge targets; every scheduled HMC move must evaluate fresh bootstrap results under its current bridge | Extra target evaluation per move; analytic normalizer/mixture tests and per-step finite telemetry |
| Rejuvenation every 8 bridges | Measured repair after `r1` falsified the full-rejuvenation compute assumption | Identity transitions are exactly bridge-invariant; eight HMC moves at 64 bridges and four at 32 bridges may fit the frozen campaign cap | Less mixing may leave concentrated weights; `r2` timing canary, then the unchanged material weight and movement vetoes |
| Physical affine chart | Measured stage-2 warm start | Both known modes locally usable; hot HMC crossed signs | Residual precision condition about 732; AIS acceptance and weight diagnostics |
| Proposal | Measured two local MAP/Hessian Gaussians, equal component draw probabilities | Normalized, covers both known modes, direct IS baseline available | Heavy tails/missed modes; ESS/max weight/schedule sensitivity |
| HMC step `0.03`, `L=4` | Conservative hypothesis below stage-2 cold step `0.05`; fixed across beta | Both proposal and target are expressed in common chart; avoids region-specific step tuning inside AIS | Too small gives weak rejuvenation; too large lowers acceptance; 4-particle canary |
| Canary `4 particles x 16 steps` | Convenience mechanics/timing minimum | Exercises full AIS graph cheaply | Cannot establish weight stability; no mass claim |
| Central `8 x 100 particles x 64 steps` | Reviewed uncertainty design | Matches prior eight-batch comparison and gives 800 independent weighted paths | Expensive; worker-parallel execution and 7,200 s cap |
| Sensitivity `2 x 100 x 32 steps` | Reviewed lower-budget schedule comparator | Detects gross schedule dependence | Two batches are descriptive; only the predeclared mean difference is a gate |
| 25 workers x 4 particles, 4 cores/worker | Revised repeated-target hypothesis; CPUs `0--99` in disjoint groups of four | AIS repeatedly evaluates one batch-4 target inside each persistent worker; four local cores reduce the serial bridge cost while 25 workers preserve 100 independent concurrent paths | This differs from the one-core training-data pool; 4-path canary measures cost before material seeds, and worker/thread affinity is archived |
| ESS `>=0.30`, max weight `<=0.02` | Stricter reviewed thresholds than failed direct IS | Requires clear repair over ESS `0.226` and max `0.572` | Finite heavy tail can remain unseen; independent batches and sensitivity also required |
| Interval half-width `<=0.08` | Reviewed precision goal | Improves materially on failed `0.167` | Not an equivalence margin |
| Schedule difference `<=0.08` | Reviewed stability goal | Detects substantial discretization/rejuvenation sensitivity | Passing two schedules does not prove all schedules agree |

## Execution plan

1. Add a diagnostic TensorFlow/TFP AIS helper with a fresh HMC bootstrap at every
   beta, and analytic tests for exact normalized
   equal/unequal-weight, unequal-scale mixtures, replay, finite traces, and weight
   correction.  Inspect and bind installed TFP 0.25 source semantics.
2. Implement 25 persistent pinned single-process workers.  Each worker owns the exact
   batch-4 target and runs four independent AIS paths under XLA.  Preserve initial
   proposal labels/signs, terminal states/signs, log weights, and final HMC telemetry.
3. Run the detached 4-particle, 16-step full-rejuvenation canary.  Require exact
   target parity, finite states/weights, zero invalid status, and final binary
   acceptance at least `0.5`.  If measured extrapolation breaches the campaign cap,
   run one sparse-rejuvenation timing canary using identity transitions between
   HMC moves every eighth bridge.
4. If the sparse canary passes and its measured extrapolation fits the cap, run
   eight independent 64-step batches and two independent
   32-step batches.  Write progress atomically after every batch and refuse overwrite.
5. Aggregate only from verified receipts.  Apply all primary gates.  If any fails,
   classify AIS as failed and write an SMC reset plan; do not quote the mode estimate
   as resolved.

## Compute and attempt budget

- Helper/tests: `300 s`.
- Canary: one scientific attempt plus one localized harness retry, `1,800 s`.
- Material: 1,000 AIS paths total, at most 57,600 bridge transitions, `7,200 s`.
- No transition-chain continuation, NeuTra training, posterior issuance, or predictive
  run is authorized here.

## Skeptical plan audit

| Risk | Resolution |
|---|---|
| Wrong baseline | Failed direct IS and exact analytic AIS fixtures are explicit. |
| Proxy promoted | Loss, final acceptance, runtime, and point-estimate agreement are explanatory only; weight reliability gates are primary. |
| Hidden proposal weight assumption | Proposal probabilities enter normalized `log q`; corrected weights determine target mass. |
| Missing mode | Two-mode scope is explicit; AIS cannot certify discovery outside proposal support observed in finite paths. |
| Schedule overfitting | Central and sensitivity seeds are disjoint; numeric gates freeze before execution. |
| Few effective weights | ESS, maximum weight, independent batches, and schedule sensitivity jointly veto. |
| TFP API semantics | Installed source inspected.  Public TFP 0.25 AIS is not used with HMC because it forwards cached prior-bridge HMC results.  Local driver re-bootstraps at each beta; helper tests normalized known targets and log normalizer ratio. |
| Long stream failure | Runs use bounded user services, atomic progress, logs, and terminal receipts. |
| Failed candidate stopping direction | AIS failure triggers SMC, exactly the planned repair; it does not reject physical multimodal inference. |

The audit passes for AIS as the next discriminating repair.  It does not authorize
using AIS output when any reliability gate fails.

## Pre-mortem

AIS may appear stable because every path begins in one of the two known modes while a
third mode remains absent.  It may also produce a stable sign fraction but unstable
normalizing weights.  Both remain explicit limitations.  A failed canary can reflect
step size or XLA integration rather than AIS mathematics; only localized repair is
allowed before material seeds.  A failed material weight gate triggers SMC, not a
conclusion about the true mode fraction.
