# SSL-LSTM q=20 Phase 9B Sequential Validation Plan

Date: 2026-09-06  
Status: `P0_PASSED_P1_INCOMPLETE_HARNESS_REPAIR_REQUIRED_P2_BLOCKED`  
Parent master: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`  
Executable-readiness Phase 0: `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-executable-readiness-phase0-plan-2026-09-06.md`  
Recovery memo: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-reset-memo-2026-09-06.md`  
Latest factor result: `docs/plans/bayesfilter-ssl-lstm-q20-factor-route-fresh-tuning-admission-result-2026-09-04.md`

## Purpose and boundary

This is the new reviewed Phase 9B subplan requested after the narrow factor
eigensystem admission. It governs sequential posterior-validation work for the
q=20 SSL-LSTM target. It does not promote the factor backend to a repository
default, and it does not treat a finite mechanics run as posterior evidence.

The first execution under this plan is P0, a read-only source, artifact, and
route-authority preflight. P0 may write a new structured receipt, but it must
not launch HMC, consume posterior draws, retune parameters, or alter any prior
artifact. A claim-bearing Phase 9B run remains closed until P0 passes and the
P1 canary entry gate is separately closed.

The recovery memo and factor result are dated 2026-09-06, the current session
date. The source-synchronized receipt carries the attempt namespace
`source-sync-20260905T203000Z`; P0 independently verifies its bytes and hashes.
The factor result supplies the narrow numerical-backend admission only. The
factor route is eligible for a fresh P1 candidate canary but is not a promoted
sampler, transport, or repository default.

The source-authority P0 is complete. A distinct M4-P0 executable-readiness
phase now sits between that preflight and P1. It must close the repaired runner
contract, stable TensorFlow graph contract, route classification, durable
budget, compile/steady-state forecast, source/seed/artifact provenance, and GPU
runtime receipt. Until then, P1 remains a reviewed plan but not a launchable
command.

## Research-intent ledger

| Item | Binding definition |
|---|---|
| Main question | Can target-specific tempered reverse-KL charts with fixed, state-independent HMC kernels provide reliable q=20 posterior exploration after fresh scope-bound tuning? |
| Candidate mechanism | Proposed factor-route fixed multi-chart HMC with exact adjacent replica exchange, using `tensorflow_eigh_strict_factor_cached` only after the source receipt and current-dated admission boundary are valid. |
| Primary comparator | Strict square-root backend with its own fresh scope-bound tuning artifacts; factor and strict artifacts may not be mixed. |
| Secondary comparators | Physical-coordinate HMC, strict single-chart HMC, physical replica exchange under the same proper bridge, and single-chart tempering where the route is implemented and independently tuned. |
| Expected failure modes | Chart collapse, poor bridge overlap, invalid inverse or log determinant, extreme acceptance with little movement, low bulk/tail ESS, seed sensitivity, mode locking, retracing, stale handoffs, source drift, or resource exhaustion. |
| Promotion unit | A declared arm and scope set, never an unqualified repository-wide route. A passing arm remains a viable candidate until downstream posterior and uncertainty gates pass. |
| Must not conclude | No IID Gaussian whitening, exhaustive mode discovery, posterior correctness, convergence, sampler superiority, high-dimensional scaling, production readiness, or repository-wide default change from P0, P1, or a short mechanics receipt. |

## Fixed target and execution identities

| Choice | Binding value and provenance |
|---|---|
| Sampling measure | q=20 SSL-LSTM posterior in `theta in R^4`; target signature `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`, inherited from the completed factor receipt and checked by P0. |
| Bridge | Proper likelihood-tempered bridge with beta slots `(0, 0.5, 1)`, same target signature, and a repository-issued bridge identity. |
| Candidate backend | `tensorflow_eigh_strict_factor_cached`, admitted only for the q=20 Phase 9B candidate lane by the current factor receipt/reconciliation; this is not a repository default or sampler admission. |
| Strict comparator | `tensorflow_eigh_strict`; it requires fresh tuning under the same Phase 9B scope and may not consume factor handoffs. |
| Transport protocol | C5 `phase8-k2-compact-high-l3-pure`, two `(16,16)` tanh charts, fixed `gamma=(0.5,0.5)`, retained as a target-specific hypothesis rather than a universal default. |
| Tuning interface | `tune_fixed_transport_hmc_kernel` with `measured_joint_grid_v1`; a chain runner or mechanics helper cannot issue a canonical tuning handoff. |
| Sequential policy | `bayesfilter_neutra_sequential_hmc_v1` through the shared repository controller; warmup is archived but excluded from posterior estimates. |
| Runtime | TensorFlow/TFP, GPU0, float64 for this validation lane, XLA enabled, TF32 enabled where applicable, and memory growth configured and verified before device initialization. |
| Forbidden shortcuts | No stale or caller-stamped handoff, no cross-scope transfer, no old M3/M3P/M3Q replay launcher, no fixed terminal burn-in in a claim-bearing route, no pfor or row-mapped scalar target, and no NumPy numerical path. |

## Evidence contract

### M4-P0 executable readiness

The detailed technical entry gate is governed by
`docs/plans/bayesfilter-ssl-lstm-q20-phase9b-executable-readiness-phase0-plan-2026-09-06.md`.
It is not a posterior-validation phase. Its exit receipt must show that the P1
runner can use the shared controller without the repaired callback/axis/archive
defects, that `_build_batched_hmc_program` has an explicit stable input
signature or a reviewed exception, that all new qualifying routes are
classified, and that a complete factor-plus-strict P1 schedule fits a durable
campaign budget. The full route-policy residuals for unrelated legacy paths
must remain visible rather than being silently relabeled.

### P0 source and authority preflight

The P0 question is whether the inputs needed for a fresh Phase 9B canary are
durable, parseable, identity-consistent, and governed by supported interfaces.

Primary pass criterion:

- the source-synchronized factor manifest and independent audit parse with
  strict JSON and match their recorded SHA-256 values;
- all six factor chart checkpoints and all six selected tuning artifacts are
  present, parseable, target-bound, factor-backend-bound, and marked as using
  `measured_joint_grid_v1`;
- the q=20 target signature and factor backend identities agree across the
  receipt, checkpoints, tuning artifacts, and plan;
- the shared sequential controller and the supported fixed-transport tuner are
  present in the current source tree; and
- the current worktree, Git revision, date discrepancy, and changed-source
  boundary are recorded in a new P0 receipt.

P0 hard vetoes are missing or modified durable inputs, invalid JSON, hash
mismatch, target/backend identity mismatch, a stale or unsupported tuning
interface, or an artifact collision. P0 does not test HMC convergence and
cannot open posterior admission by itself. The factor result and reset memo are
current authority for the narrow backend state, while the P0 receipt verifies
the underlying source artifacts.

### P1 sequential canary

P1 is the smallest real sequential execution. It must use four chains, the
factor candidate, and the strict comparator only after each has a fresh
same-scope tuning handoff. It must use distinct calibration, held-out, and
confirmation seed namespaces and a new output root. P1 must execute the shared
sequential policy with:

- warmup chunks of 500 transitions per chain;
- at least 2,000 warmup transitions per chain;
- a recent 1,000-transition warmup diagnostic window;
- warmup R-hat threshold `<= 1.05`;
- retained chunks of 500 transitions per chain;
- at least 1,000 retained transitions per chain for the canary;
- retained R-hat threshold `<= 1.01`;
- minimum bulk and tail ESS of 400 where the selected controller exposes these
  diagnostics;
- finite state, target value, target score, target status, movement, and
  declared energy-error checks on every chunk; and
- native divergence telemetry recorded as unavailable when the TFP kernel does
  not expose it, never silently reported as zero.

The 500/2,000/1,000 sizes and 1.05/1.01/400 thresholds are inherited from the
repository sequential-controller policy and are treated as reviewed policy
values, not evidence that this target has converged. The P1 canary is a route
health and gate diagnostic; its draws cannot be used as final posterior
confirmation data.

### P2 untouched sequential validation

P2 may start only after P1 closes with a durable result and a refreshed
subplan. It must use fresh confirmation seeds and a new output root. It must
retain every warmup chunk while excluding warmup from posterior estimates and
grow retained sampling cumulatively until the declared retained gate or the
10,000-transition-per-chain cap. P2 must report modern split/folded R-hat,
bulk/tail ESS, MCSE, finite-state/status, movement, energy-error, and
divergence telemetry for every arm and scope.

P2's primary promotion criterion is joint validity: every declared arm and
scope passes identity, finite-health, warmup, retained, and downstream
posterior/reference checks under the predeclared uncertainty analysis. Runtime,
acceptance, ESS/gradient, and residual differences are descriptive unless the
plan's paired uncertainty analysis supports a comparison.

## Chart-quality decision

The factor admission receipt reports centered log-density RMS values of
`535.98` and `661.94`, and largest pullback-score RMS/coordinate values of
`1438.64` and `2551.14`. These are already above any plausible numerical
roundoff tolerance and are a hard veto on an IID-Gaussian whitening or NeuTra
quality claim for the current charts.

For Phase 9B, chart quality has two separate roles:

1. Non-finite round-trip, log-determinant, score, or conditioning values are
   hard vetoes.
2. Finite centered-density and pullback-score residuals are a candidate
   promotion veto until an independent calibration bank establishes the
   target-specific acceptance thresholds. The current large residuals are not
   waived, averaged away, or treated as a continuation veto; they trigger a
   fresh chart-training/calibration repair before posterior interpretation.

The threshold-calibration receipt must precede P2 and must state numeric
thresholds, their provenance, the calibration/validation split, and the exact
decision rule. Until that receipt exists, the allowed outcome is “mechanics
candidate only,” not posterior admission. This avoids converting an arbitrary
residual cutoff into a scientific default.

## Phase sequence and budgets

| Phase | Scope | Material budget | Entry and exit |
|---|---|---:|---|
| P0 | Source, artifact, route, and date-boundary preflight | 60 seconds wall-time planning allowance; no TensorFlow or HMC | Pass only if all authority and identity checks pass; write one immutable receipt. |
| P1 | Four-chain factor candidate and fresh strict-comparator canary | New budget required in the P1 subplan; no inherited factor remainder | Requires P0 receipt, fresh same-scope handoffs, fresh seeds, and a canary result/closeout. |
| P2 | Untouched sequential warmup and retained sampling | New budget required in the P2 subplan; maximum 10,000 retained transitions per chain | Requires P1 closeout, chart-threshold receipt, and a complete uncertainty plan. |
| P3 | Downstream posterior/reference, mode-travel, and comparator review | New budget required in the P3 subplan | Closes Phase 9B or records the exact candidate repair; does not silently promote a default. |

The approximately 575.59 seconds remaining in the factor-tuning campaign is
not transferred to P1/P2 and is not authorization for a material run. Every
Phase 9B material run must declare a fresh budget and versioned output root.

## Default and assumption audit

| Choice | Provenance | Failure mode | Earliest diagnostic | Promotion status |
|---|---|---|---|---|
| Factor eigensystem backend | Source-synchronized receipt and factor admission result | Backend parity may hold on fixtures but fail in longer cold sampling | P1 strict comparison and sequential canary | Candidate only |
| C5 chart architecture | Frozen calibration representative | Charts may be poorly Gaussianizing, as current residuals suggest | Independent chart-quality calibration before P2 | Hypothesis, not default |
| Four chains | Shared controller minimum and existing mechanics convention | Too few chains can hide multimodality or inflate uncertainty | Modern split/folded R-hat, ESS, MCSE, and mode travel | Validation minimum, not evidence |
| Warmup/retained policy | Repository `bayesfilter_neutra_sequential_hmc_v1` policy | Short or discarded fixed burn-in could bias the result | Cumulative warmup readiness and retained gates | Reviewed policy |
| GPU0/XLA/TF32/memory growth | Repository execution policy and prior receipt provenance | Device initialization or retracing can distort cost or invalidate launch | P0 record, pre-launch memory-growth check, retracing telemetry | Execution requirement |
| Chart residual threshold | Not yet target-specific; intentionally withheld from arbitrary choice | An arbitrary cutoff could accept a non-Gaussian map or reject a useful candidate for the wrong reason | Independent calibration/validation threshold receipt | Unresolved; blocks P2 |

## Pre-mortem and skeptical plan audit

The plan was audited before execution for wrong baselines, proxy promotion,
missing stop conditions, unfair comparisons, stale context, environment
mismatch, silent defaults, and whether the planned artifacts answer the stated
question.

| Audit question | Finding and repair |
|---|---|
| Is the baseline fair? | Yes for the declared candidate/comparator boundary: factor and strict must each have fresh scope-bound handoffs; no old strict artifact is reused. |
| Could a proxy become a promotion criterion? | Prevented: acceptance, runtime, ESS/gradient, residuals, and short mechanics results are descriptive or veto diagnostics; downstream posterior/reference agreement is required later. |
| Are stop conditions explicit? | Yes: artifact/hash failure, identity drift, unsupported tuner, source closure change, budget exhaustion, missing threshold receipt, and target/data/hardware/privacy/contract change stop or require a new plan. |
| Could the run pass while misleading us? | Yes: short chains may miss cold modes and large chart residuals may invalidate Gaussianization. P1 is therefore mechanics-only; P2 requires modern diagnostics, fresh confirmation seeds, and reference/mode checks. |
| Could it fail for infrastructure rather than science? | Yes: retracing, resource caps, or serialization can fail a candidate without rejecting the research direction. Each must be preserved and classified in its own receipt. |
| Does the first artifact answer the question? | P0 answers only whether a governed fresh run is technically admissible. It cannot answer posterior correctness or transport quality. |

Audit verdict: `PASS_P0_PLAN_REVIEW_P1_PLANNING_OPEN_P2_BLOCKED`.
The unresolved chart-threshold choice is deliberately recorded as a P2 entry
veto rather than silently selected as a convenience default. The factor result
and reset memo remain narrow-scope authority and do not open P2 by themselves.

## Artifact contract

P0 writes under:

`docs/plans/artifacts/ssl-lstm-q20-phase9b-sequential-validation-2026-09-05/p0-source-preflight/`

Required P0 files are `run_manifest.json`, `p0-preflight.json`, and a Markdown
result note. Every later phase must use a new attempt directory and include a
manifest with the Git revision and dirty-worktree hash, actual command,
environment, device and memory policy, seeds, wall time, source and plan
hashes, artifact paths, and explicit claim boundaries.

## P0 execution result

P0 completed with status `PASS_PHASE9B_P0_SOURCE_PREFLIGHT` and no failures.
It verified the source-synchronized factor manifest and audit hashes, all six
chart/beta checkpoint and tuning pairs, the q=20 target signature, the proposed
factor backend, the supported sequential controller, and the supported fixed
transport tuner. It did not import TensorFlow, launch HMC, consume posterior
draws, or retune any parameter.

The authoritative P0 manifest is:

`docs/plans/artifacts/ssl-lstm-q20-phase9b-sequential-validation-2026-09-05/p0-source-preflight/run_manifest.json`

Its SHA-256 is
`d29501a7d258fca4baec503a1449880109cccf632e532dcf3824e3926b2b67bd`.
The P0 claim boundary remains
`phase9b_p0_source_authority_preflight_only`. The factor result and reset memo
provide the narrow backend state; no posterior or sampler claim follows.

## Execution command

The authorized P0 command was executed successfully:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/audit_ssl_lstm_q20_phase9b_plan_2026_09_05.py \
  --output-dir \
  docs/plans/artifacts/ssl-lstm-q20-phase9b-sequential-validation-2026-09-05/p0-source-preflight
```

No P1/P2 command is authorized by this document. P1 planning is now open, but a
P1 command requires the passing M4-P0 executable-readiness closeout, its own
refreshed subplan/audit, a fresh material budget, a fresh output root, and the
passing source-authority P0 result. P2 remains blocked by chart-threshold and
downstream posterior gates.

## P1 execution update — 2026-09-06

The reviewed P1 subplan was executed three times under fresh output roots. The
first two attempts failed in the harness while selecting the returned chart
object. The third repaired that indexing defect, completed factor chart
construction and tuning, and reached the first 500-transition report before
failing because a telemetry evaluator was supplied where the shared controller
expects a status-mapping summarizer. No sequential chunk archive, posterior
draw, strict comparator result, or P1 pass exists.

The current decision is `P1_HARNESS_REPAIR_BUDGET_REASSESSMENT_REQUIRED`, and
M4-P0 is the active executable-readiness gate.
The runner must be repaired and tested for its callback contract, shared
controller `[draw, chain, parameter]` to diagnostic `[chain, draw, parameter]`
axis conversion, MCSE reporting, role/arm/attempt seed disjointness, failure
provenance, and pre-chunk budget enforcement. The observed first sequential
chunk took approximately 1,423 seconds after tuning, so the declared minimum
schedule is not supported by the remaining nominal P1 budget. No further
material GPU launch is valid until M4-P0 passes, compile and steady-state costs
are measured, and a refreshed budget passes the skeptical audit. P2 remains
blocked.
