# KSC discrepancy analysis — 2026-09-29

## Question and configuration

The owner requests a reviewed plan and execution to distinguish an incorrect
finite-program derivative, particle-design variability, and persistent
particle/numerical approximation error against the full seven-component KSC
likelihood and score. Earlier comparison completion did not establish a cause.

**FP64 TensorFlow GPU/XLA diagnostic variants, TF32 off; UNTUNED diagnostic
cells.** Contract E and dual-cap safeguards remain enabled. Production uses
FP32/TF32. The old route-specific T120 controls are frozen experimental
baselines; changed particle counts/settings do not inherit tuning admission.
No default, production, HMC, native-SV exactness or method-ranking claim.
Target: theta=(gamma_raw,log_beta)=(1.5,0), Q=1, h0~N(0,1), prediction before
observation, all seven observation components. Three SQMC routes share Halton
inputs at matched seeds; IID uses Gaussian draws. All call the shared executor.

Baseline: docs/benchmarks/sqmc-ksc-full-mixture-corrected-results-20260929.md.
The corrected Gaussian-sum reference agrees with an independent density grid.
The old single-Gaussian Kalman calculation is a heuristic, not the target.

## Evidence contract and research intent

The primary outcome is a supported explanation or an explicit unresolved
classification for each proposed source of error. There is no promotion.
The target is grad(log L_7); the particle score differentiates its finite
likelihood program. Equality is not assumed. FD checks the derivative of that
program; reference agreement checks its approximation to the target.

| Diagnostic | Role and consequence |
|---|---|
| Same-input replay | Engineering validity; unexplained drift stops attribution pending localization. |
| Stable branch-matched FD disagreement | Derivative defect and repair trigger; veto score interpretation until repaired. |
| Branch changes or unstable FD | Unresolved nonsmooth diagnostic, not automatic implementation failure. |
| Fixed-data replication and N ladder | Conditional variability/persistent-error evidence; not population ranking. |
| Flow/transport refinement | Explanatory convergence/sensitivity, not tuning on old claim data. |
| Unresolved reference mismatch | Continuation veto. |
| Nonfinite/inconsistent particle output | Candidate invalidity; preserve it and continue healthy diagnostic arms. |
| Source drift, corrupted evidence, budget/deadline exhaustion | Continuation veto; preserve partial results. |

Report actual log likelihood and both score coordinates, signed/absolute
coordinate errors, vector norms, controls and validity for every evaluation.
A finite-program FD pass is not likelihood-score accuracy. A lower observed
error is not method superiority. No implication for other regimes or HMC.
Construct and report cheap adversaries separately per dataset/horizon: zero
score (no local information), exact first-observation mixture score (discard
later information), and moment-matched Gaussian Kalman (collapse mixture).
Their errors against the full-mixture reference are sanity-check promotion
vetoes only; losses cannot stop an investigation designed to explain them.

## Data and phases

Read immutable inputs at Git 023e106102c89ee9d4787df3a55fbf5f68b88ecf via
`git show` (sparse checkout omits old evidence). Snapshot exact bytes/hashes.
Use T120 dataset 213001 by seed order and the different dataset with largest
archived mean score-vector error across all four routes. This retrospective
selection ensures a large-error case and cannot estimate population accuracy.
Use T10/T50 prefixes of these observations for localization; compute checked
mixture references for these prefixes, which differ from old T10/T50 datasets.

1. **Replay and bounded checks.** Verify checkout, applicable policy, writable
   roots and budget. Use trusted NVIDIA/TF probes and verified memory growth.
   Replay both T120 datasets, original design seeds and all four old controls.
   Require agreement <=1e-8*(1+abs(saved value or score)). A diagnostic wrapper
   calls the same executor with existing `return_trace=True`; return only
   ancestry indices plus value/gamma derivative, allowing XLA to eliminate
   unused dense trace histories. Check normal-endpoint parity at N16/T2 and
   N1008/T120, and bound memory before using this wrapper for FD.
2. **Derivative checks.** Both datasets, all routes, N1008, original design
   seeds, T10/50/120, both coordinates. Centered FD h=1e-4,1e-5,1e-6, extending
   to 1e-7,1e-8 only if unresolved. Keep initial cloud, noise and uniforms fixed
   at theta and theta±h. Save one-sided slopes and ancestry differences/first
   changed time. Two adjacent steps must have unchanged ancestry, FD agreement
   <=1e-4*(1+abs(score)), and analytical agreement <=2e-4*(1+abs(score)). Stable
   branch-matched disagreement is a derivative defect; crossings/instability
   are unresolved. Localize a defect at the shortest failing prefix before a
   narrow repair. Trace parity failure invalidates this diagnostic wrapper.
3. **Fixed-data replication.** Both T120 datasets, four routes, N1008/2016/4032,
   design seeds 231001..231008. Freeze controls across N to isolate resolution.
   Repository chunk rule gives K1008/2016/2016. Report conditional mean values
   and scores, coordinate error mean/SD/SE, mean norm error with SD/SE, norm of
   mean error, paired changes across N and all draws. Exploratory Student-t
   intervals for coordinate mean errors use df7; state small-sample assumptions
   and avoid population inference. Eight replications/two cases cannot prove
   unbiasedness or asymptotic convergence.
4. **Numerical controls.** Both T120 datasets, seeds231001/231002, all routes,
   N1008. Compare baseline, flow16/32/64, Sinkhorn/balance96/48 and384/192,
   epsilon25.6/409.6 with96/48, and combined flow64/384/192 at original epsilon.
   One-factor comparisons isolate resolution; epsilon changes regularization,
   not merely solver accuracy. Preserve paired inputs and actual oracle errors.
   Do not select/promote settings. A derivative defect blocks score attribution
   but still permits value diagnostics and its planned repair.

Localized implementation defects may be repaired after reproduction with a
focused regression and fresh attempt under the same budget. Preserve failed
results. No model, baseline, safeguards, package or environment changes.

## Assumption audit and pre-mortem

| Choice/provenance | Reason; failure mode; early diagnostic; status |
|---|---|
| Prior FP64 GPU/XLA configuration | Reduce precision confounding; not production validation; replay/device metadata; diagnostic baseline. |
| Same theta/prior/data | Isolate existing errors; narrow regime; matched-input hashes; frozen baseline. |
| Old numerical controls | Isolate N; may not converge; separate refinement ladder; experimental warm start, not tuned new scope. |
| Positive ridge/damping/trust controls | Preserve compared program and safeguards; old sensitivity did not prove non-harm; retain and disclose unresolved calibration; no safeguard retuning. |
| Two retrospective cases | Include the observed failure; selection bias; report separately; explanatory choice. |
| Eight random designs | Estimate conditional variability; skew/outliers weaken intervals; preserve all draws/SD/SE; exploratory. |
| FD ladder/ancestry checks | Detect truncation and resampling crossings; not every nonsmooth branch is observed; require stable slopes; diagnostic thresholds. |
| Shared trace interface | Avoid a filter fork; tracing may affect numerics/memory; endpoint parity/bounded first call; engineering hypothesis. |

A passing run could mislead by confusing finite-program and target derivatives,
optimizing old holdouts, interpreting branch crossings as missing terms, or
mixing data and design variance. This plan separates those questions. Residual
error may still involve reset approximation or untested settings; eliminating
one explanation does not prove another. A repair in later phases remains
allowed after a failed candidate; only the stated continuation vetoes stop it.

## Budget and execution

Immutable prior ledger:
docs/plans/artifacts/sqmc-ksc-full-mixture-20260929/budget.json
SHA256 eedf51ff268c027ed73ede76b5f25b74676de35d8d0a25ea58a51314744842cc.
It includes all previous work and the unchanged300s old-hook reserve:
24462.190031178063s charged of43200s;18737.809968821937s remain (5.204947h).
Do not add its predecessor again. Allocate <=14400s GPU-owning worker wall
time, including probes, compilation, failures and repairs. Same deadline:
2026-09-30T16:15:40.010888+00:00. Maximum two infrastructure retries per unit.
Supervisor enforces the smaller of allocation, aggregate remainder and elapsed
window, retaining shutdown margin. One GPU worker at a time. CPU-only tests
and reporting hide GPUs before framework import and consume no GPU budget.

Root: docs/plans/artifacts/sqmc-ksc-discrepancy-20260929/attempt-01/.
Fresh attempt directories; full logs, manifest with commit/source snapshots,
command/environment/device/allocator metadata, input/design hashes, seeds,
wall time and plan/result paths. Link prior ledger; record attempts before
launch and reconcile any interruption before retrying.
Interpreter: /home/chakwong/anaconda3/envs/tftwogpu/bin/python.
Runner: docs/benchmarks/run_sqmc_ksc_discrepancy.py; `--mode prepare`,
`--mode supervise --phase replay|derivative|particles|numerics` plus internal
worker. Syntax/focused harness checks precede trusted GPU supervisors.
Finish a result note, complete tables, decision/inference-status tables,
budget reconciliation and terminal engineering/numerical/interpretive review.
Update concise checkpoint. No merge/push or promotion in this follow-up.

## Skeptical review before implementation

Codex self-review checked the current call chain and archived evidence. It
found and corrected three material risks: full-scale FD crosses ancestry
branches; changed N cannot inherit tuning admission; arbitrary fixed datasets
might omit the discrepancy. The revised plan adds branch-aware FD, explicit
UNTUNED status and transparent case selection. Comparator, every coordinate,
input pairing, budgets/stops, failure classes and non-ranking interpretation
are explicit. **Proceed with bounded replay first.** Review is supported by
executable checks; no independent reviewer is used. Revisit these limitations
at terminal review. No mandatory review chain is introduced.
