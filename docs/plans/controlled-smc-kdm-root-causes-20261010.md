# Replication root causes and bounded repairs — 10 October 2026

Parent: `controlled-smc-kdm-replication-master-program-20260928.md`.
This replaces the next-action checklist, not the preserved qualification results.

## Intent and evidence contract

Question: which remaining differences are translation defects, shared source
conditioning, or insufficient statistical evidence? The target remains author
replication, with no materially worse numerical results. Shared author failures
are acceptable. No author-method enhancement, Task 3/E0, HMC or default-quality
claim is part of this work.

Comparators are the pinned Heng source (2696f6d611aae172d10102cb7bfbfd314bb6b88f)
and Younis source (b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18), using the preserved
source-controlled fixtures from `gap-closure-20261010/localization-01` and
`c0-source-fit-20261010`. The cSMC fit fixture has exact result/RNG controls;
KDM has 64 exact native/replayed-source controls. Hash the inputs again.

Replication promotion criteria remain the existing cSMC likelihood 1% / variance
5% author-relative criteria and KDM 0.01-nat NLL / 1% RMSE criteria, with their
uncertainty requirements and no additional TF failures on source-valid cases.
These diagnostics cannot themselves qualify either full replication.

| Observation | Role and consequence |
|---|---|
| Source controls, fixture hashes, shapes or GPU/XLA placement wrong | Continuation veto; repair harness |
| Extra nonfinite values or invalid state | Promotion veto and localized repair trigger |
| Strict coefficient/gradient mismatch | Explanatory and repair trigger, never a bitwise prerequisite |
| Matched-input solver/optimizer error reduced by source-matching repair | Engineering repair evidence; requires endpoint regression |
| Shared ill-conditioning or source tails | Explanation, not permission to loosen statistical margins |
| Interval too wide | Inconclusive; feasibility calculation, no degradation claim |
| Budget/attempt cap reached | Continuation veto for this allocation |

The heuristic adversary is the unchanged author algorithm on identical inputs;
the question is translation, not improvement over practitioners' methods. No
ranking, superiority or general accuracy claim will be made from these fixtures.

## Phases and hypotheses

1. **C-SOLVE.** On all 101 saved backward regression rows, hold the actual
   design, target, rounded Gram matrix and RHS fixed. Compare source coefficients
   to Cholesky, general dense solve, rank-truncated Gram least squares and direct
   design least squares. Record source branch, rank/conditioning and prediction
   errors. Run the actual TF `regress -> _regress` endpoints on the same inputs.
   Hypotheses: source conditional solver semantics; Gram squaring of condition
   number; independent rounding of Gram/RHS; propagation through backward
   corrections. Distinguish first backward time (100) from chronological time 0.
   A better fit residual is not evidence of more faithful author replication.
2. **C-GRAPH.** Reuse one mutable forcing variable and two fixed-signature
   solver instances in the existing fit/forward diagnostic. Check every source
   forward control and both solver results against the preserved run; verify
   each wrapper traces once. This is harness repair only. No XLA-off/eager
   fallback and no algorithm change.
3. **K-RESET.** Reset TF parameters/moments to the exact source pre-update state
   before each of four updates, compare with the evolving-run evidence. This
   separates instantaneous arithmetic sensitivity from accumulated divergence.
   Preserve FP32 and FP64 results, all finite/commit checks and endpoint calls.
4. **K-ADAM.** Feed identical clipped source gradients and prior optimizer states
   through the actual TF optimizer endpoint and a diagnostic source-scalar
   implementation. PyTorch's CPU Adam computes beta powers/step size in Python
   binary64; current TF computes these in parameter dtype. Test both against
   actual source parameters/moments. This cannot explain update-0 raw gradients.
   If isolated evidence confirms a scalar-arithmetic defect, repair only that
   arithmetic and verify the actual endpoint plus coupled replay. Preserve old
   results and explicitly identify checkpoint/learning-rate semantics.
5. **K-ANGLE.** At identical observed raw angular vectors, verify
   `atan2(v/r,u/r)` derivatives against `(-v,u)/(u*u+v*v)` in FP32/FP64. Measure
   conditioning and perturbation sensitivity; normalization is unchanged.
   A radius close to zero amplifies angular error by approximately `1/r` and
   derivative error by order `1/r^2`. This diagnoses shared source sensitivity;
   it cannot by itself bound the complete network gradient.
6. **PRECISION/NEXT.** Inspect the frozen Q2 uncertainty calculations and
   completed training cost/protocol. Calculate a descriptive precision forecast,
   including its heavy-tail limitation, and specify the smallest justified
   prospective qualification design. Do not append samples to Q2, train on
   qualification data or promote a single training unit to population evidence.

## Defaults and assumption audit

| Choice and provenance | Risk; early check; status |
|---|---|
| Existing development fixtures/seeds | Selected diagnostics cannot generalize; preserve source controls; diagnostic only |
| FP64 cSMC / FP32 and FP64 KDM, TF32 off | Not TF32 production evidence; exact device/dtype manifest; frozen reference scope |
| SciPy/NumPy independent solver analysis | Different LAPACK driver may differ from Octave; source-beta control and explicit branch tests; reference only |
| Native TF/XLA regression/update endpoints | Synthetic subcomponent alone could miss wiring; endpoint comparison required; unchanged candidate |
| 128 ulps-scaled FP32 and 1e-9 FP64 legacy screens | These are diagnostic thresholds, not downstream margins; retain raw errors |
| Source-scalar Adam candidate | Can change accumulated trajectory; isolate identical gradient/state first, then coupled replay; hypothesis |
| No angular clipping/ridge/new solver default | Owner's author-replication target; source itself uses unprotected normalization/normal equations; source behavior retained |
| sqrt(n) precision forecast | Heavy tails can invalidate forecast; label descriptive, inspect influential pairs, do not promise qualification |

## Execution, budget and artifacts

Reserve at most **600 CPU / 1800 GPU wall seconds**, two scientific attempts plus one serialization-only recovery attempt,
from the existing campaign balance (verify before launch). Failed attempts and
GPU preflight count. This is a bounded diagnostic/repair allocation; a new
training/qualification allocation must have its own fixed sample/design and
budget within the already authorized campaign. Routine syntax/unit checks are
recorded separately. Fresh outputs:
`artifacts/controlled-smc-kdm-replication-20260928/root-causes-20261010/root-causes-NN`.

The existing approved supervisor is extended with `--phase root-causes`:

```text
/home/chakwong/anaconda3/envs/tftwogpu/bin/python /home/chakwong/BayesFilter/docs/benchmarks/run_replication_gap_localization.py --phase root-causes --output /home/chakwong/BayesFilter/docs/plans/artifacts/controlled-smc-kdm-replication-20260928/root-causes-20261010/root-causes-01
```

Use escalated GPU access, TF memory growth verified before initialization,
TF32 off, static signatures, JIT on. CPU subprocesses hide GPUs before imports.
Preserve commands, source/input hashes, Git commit, environment, seeds, wall
times, complete logs, per-check results and terminal budget charges. The script
must enforce caps and terminate only its own process groups. No rule expansion
is needed: the literal supervisor interpreter/script prefix is already approved.

## Skeptical pre-execution review

Reviewed the call chain, source controls, stale acceptance labels, scope and
budget. Three misleading outcomes are explicitly prevented: treating solver
coefficients as the replication goal; blaming Adam for a pre-optimizer gradient
error; and interpreting wide variance intervals as worse numerical behavior.
The six graph instances in the previous harness each traced once: the warning
is not yet evidence of input-driven retracing. Reuse will test that explanation.
The source-scalar candidate must be evaluated before runtime changes. A proposed
larger run without a plausible precision/cost calculation is rejected.

**Verdict: execute the bounded phases.** No material unexamined default is being
promoted. Terminal review must state confirmed causes, unproved explanations,
repair non-harm, remaining uncertainty and the next justified action.


## First execution and localized repair review

Attempt `root-causes-01` completed, costing 5.3773 CPU and 165.2156 GPU
wall seconds. Resetting source pre-update states reduced the failing FP32 raw
parameter-gradient comparisons from 18 to 3; FP64 passed. Holding gradients
identical isolated a separate Adam scalar mismatch: the source-scalar candidate
reduced maximum FP32 parameter error from about 1.2e-7 to below 7.5e-9.
The actual runtime now computes Adam bias corrections and learning-rate scalar
arithmetic in binary64, casting at source tensor-operation boundaries. Neural
training checkpoints use v2 with the scalar policy recorded; old v1 optimizer
states are rejected explicitly, while preserved evidence remains untouched.

The second attempt must repeat source-controlled replay, compare isolated
optimizer outputs, check exact learning-rate checkpoint round trips and rejected
incompatible checkpoints, and verify direct optimizer calls from replay, native,
full-training and weight-pretraining consumers. It also replays each saved
Gram/RHS through Octave to distinguish source branch behavior from independently
recomputed matrices. The qualification-size calculation reports zero-effect
and observed-effect scenarios separately: neither is a power guarantee, and
neither establishes heavy-tail adequacy. The variance rule is one-sided.

Skeptical repair review: scalar precision is a translation correction supported
by the actual pinned source optimizer and matched-input evidence. It cannot
explain the initial raw-gradient discrepancy. No loss, clipping, optimizer
hyperparameter, angle protection or acceptance margin changes. Second-run
non-harm checks remain required; passing four updates does not qualify full
training. The existing two-attempt, 600 CPU / 1800 GPU-second allocation holds.


### Serialization repair before recovery attempt

Attempt 02 stopped after 1.7179 CPU seconds because installed Octave does not
implement jsonencode. The numerical replay completed; export failed before GPU
work. Replace this exporter with Octave fprintf TSV plus standard-library JSON
serialization, validate row count, and preserve attempt 02 unchanged. Permit
one additional launch (attempt 03) within the unchanged 600 CPU / 1800 GPU
seconds. This is a localized infrastructure retry under the active campaign
policy; no scientific criterion, source, data or numerical target changes.
