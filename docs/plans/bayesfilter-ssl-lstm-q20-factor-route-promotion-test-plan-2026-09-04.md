# q=20 Factor-Route Promotion Test Plan

Date: 2026-09-04  
Status: `COMPLETE_DEFAULT_ELIGIBLE_PENDING_FRESH_TUNING`  
Owner: BayesFilter research workspace  
Parent authority: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`  
Related diagnostic: `docs/plans/bayesfilter-ssl-lstm-q20-gpu-replay-batching-eigh-reuse-result-2026-09-04.md`

## Purpose

This plan tests whether `tensorflow_eigh_strict_factor_cached` can replace
`tensorflow_eigh_strict` in the q=20 SSL-LSTM target and its fixed-beta HMC
mechanics.  The candidate removes a second eigendecomposition by reusing the
factor basis for the Sylvester derivative solve.  It is not the same as the
historical raw-covariance reuse candidate, which has already failed score
parity and remains vetoed.

The target question is deliberately narrower than “is the sampler better?”:

> Does the factor route compute the same finite q=20 value, score, status, and
> one-step/full-chain mechanics as the strict route, with an unambiguous
> backend-bound identity and no XLA, retracing, or allocator violation?

A passing answer permits a reviewed factor-route candidate/default decision;
it does not establish posterior correctness, convergence, whitening, mode
discovery, or high-dimensional scaling.

## Current blocker and repair

The component target includes `principal_sqrt_backend` in its adapter
signature, but the bridge and `FixedBetaBridgeAdapter` signatures previously
did not carry that identity.  Consequently a strict tuning artifact could be
mistaken for a factor artifact even though the numerical program differed.
No promotion or fresh tuning is valid until the bridge and fixed-beta adapter
bind the component adapter signature and a regression test proves that strict
and factor constructions have distinct signatures while retaining the same
mathematical target signature.

## Research-intent ledger

| Item | Binding definition |
|---|---|
| Question | Can the factor-basis eigensystem route replace strict eigensystem evaluation for q=20 mechanics? |
| Baseline | `tensorflow_eigh_strict`, q=20, float64, TensorFlow/TFP, XLA on, GPU0; strict is the current numerical authority. |
| Candidate | `tensorflow_eigh_strict_factor_cached`, identical target data, bridge, seeds, state bank, and HMC settings. |
| Expected failure | Score drift in near-degenerate spectra, status-class changes, HMC transition divergence, stale tuning identity, retracing, or memory-policy violation. |
| Primary promotion gates | Backend identity separation; finite/status parity; value and score parity on actual q=20 batches and spectral stress fixtures; paired one-step transition parity; no hard numerical/runtime veto. |
| Promotion veto | Any identity collision, non-finite result, status/class mismatch, declared tolerance failure, transition mismatch beyond declared tolerance, missing telemetry, XLA/memory-growth violation, stale scope, or unmeasured arm. |
| Continuation veto | The target/bridge contract or required artifacts cannot be made durable, the harness is invalid, the fixed target changes, the campaign budget is exhausted, or three localized repairs fail to make progress. A candidate parity failure alone rejects the candidate and triggers its declared repair; it does not reject the research direction. |
| Repair triggers | Identity collision, spectral parity failure, transition sensitivity, retracing, allocator pressure, or insufficient stress coverage. |
| Nonclaims | No posterior correctness, convergence, R-hat/ESS claim, whitening, mode-discovery guarantee, sampler superiority, production readiness, or broad dimension-scaling claim. |

## Evidence contract

The exact comparison is a paired comparison: each candidate receives the same
TensorFlow tensors, state bank, beta, step size, leapfrog count, and stateless
seed.  The strict route is evaluated first only for reproducible artifact
ordering; ordering is not a scientific advantage.  Every attempt writes a new
directory below:

`docs/plans/artifacts/ssl-lstm-q20-factor-route-promotion-2026-09-04/`

Each serious artifact records the Git revision, command, Python/conda
environment, visible and logical GPUs, memory-growth verification, TF32/XLA
settings, target/bridge/fixed-adapter signatures, seeds, timings, trace counts,
and the plan path.  Existing strict/factor eigensystem benchmark artifacts are
historical inputs for diagnosis only; this plan does not overwrite or relabel
them.

The reusable eigensystem benchmark also contains the previously rejected raw
cache arm.  For this campaign, `--candidate-backend
tensorflow_eigh_strict_factor_cached` is mandatory: the process status is then
determined only by the factor candidate.  An all-candidate invocation remains
useful as a diagnostic, but its intentional raw-arm veto must not be confused
with a factor promotion result.

Hard evidence is limited to implementation and numerical validity:

* signatures must differ exactly where the numerical backend differs and match
  where the mathematical target is intentionally shared;
* all required values, scores, statuses, repair counts, and invalid-row
  classifications must be finite/equal or within the predeclared tolerances;
* paired HMC transitions must preserve state/target/score/acceptance/status
  behavior within the declared comparison tolerance and must have no divergence
  or non-finite telemetry;
* all repeated TensorFlow calls must use stable signatures and satisfy the
  no-pfor policy; and
* GPU runs must configure memory growth before TensorFlow device initialization.

Runtime, compile time, eigensystem-node count, acceptance, displacement, and
short-chain summaries are explanatory diagnostics.  They may nominate a useful
candidate or trigger repair, but cannot by themselves promote the route.

## Phases and budgets

The campaign uses a bounded 3,600-second material envelope, excluding ordinary
document edits and focused CPU tests.  A phase may stop early on a hard veto;
unused time is not silently reassigned to a different scientific question.

### P0: identity repair and contract fixture (<= 300 s)

1. Add the component adapter signature to the bridge identity payload and to
   the fixed-beta adapter payload.  Keep generic component targets without an
   adapter signature explicitly represented as unbound rather than inventing a
   backend identity.
2. Add a CPU-hidden regression that constructs strict and factor q=20 bridges
   and fixed-beta adapters, checks the expected equality/inequality relations,
   and verifies deterministic repeatability.
3. Run the focused bridge and fixed-transport identity tests.

Exit: identity collision is repaired and the focused tests pass.  Failure is an
implementation repair trigger; no numerical campaign may start.

### R0: repair/refresh closeout

Preserve the test output, record changed files and hashes, classify the issue,
and refresh this plan's status and exact P1 command.  If the fixture reveals a
generic-target compatibility issue, repair it before proceeding.

### P1: expanded numerical parity (<= 1,200 s)

Run the existing q=20 batch-native benchmark with strict and factor routes on
GPU0/XLA/float64 and on a CPU-hidden reference fixture.  Use at least the
center and varied row-offset batches, plus the existing near-degenerate
covariance/derivative fixtures.  Compare value, score, status code, row class,
repair/invalid counts, reconstructed covariance, and minimum spectral-gap
telemetry.  Use the existing tolerances in the benchmark as the initial
reviewed baseline (`value rtol 1e-10`, `score rtol 1e-9`, `atol 1e-10`); any
tolerance change is a repair and requires a written numerical justification.
Record compile/steady timings and graph eigensystem-node counts separately.

Exit: strict and factor pass every required parity/status check on every
fixture and device class.  A failure rejects the candidate for this attempt and
triggers a fresh stress fixture or code repair; it does not authorize relaxing
the tolerance.

### R1: repair/refresh closeout

Audit the parity artifact for finite values, complete status fields, exact
backend identities, trace counts, and device policy.  Refresh P2 inputs from
the observed weakest spectral case; preserve failed candidates in their own
directory.

### P2: paired one-step HMC transition (<= 700 s)

Construct separate strict and factor q=20 fixed-beta adapters with the repaired
identities.  Use the shared `FixedTransportReusableRunner`/one-step mechanics
route, four-chain state bank, intermediate and cold beta levels, one declared
positive step size and leapfrog count, and at least two fresh stateless seeds.
Compare next state, accepted state, target value, score, log acceptance, energy
error, accept bits, divergence/status telemetry, and tracing count.  The
declared hard energy-error veto is `abs(delta_h) <= 1.0` for every reported
transition; this is a mechanics safety bound, not a claim about a universal
HMC divergence threshold.  When the installed TFP kernel exposes native
divergence flags they must be false and equal.  When it does not, the artifact
must say `native_divergence_status=unavailable` and the energy-error veto is the
only available hard screen; absence is never reported as zero divergences.
This is a paired mechanics equivalence test, not a convergence run.

The initial transition comparison tolerance is `atol=1e-8` and `rtol=1e-7`,
recorded as a float64 accumulation hypothesis rather than a universal theorem;
it is frozen for this campaign and may not be relaxed after seeing results.

Exit: no hard veto and no unexplained backend-dependent transition difference.
Any difference beyond the declared numerical tolerance is a candidate failure
and triggers a localized transition diagnostic.

### R2: repair/refresh closeout

Record whether a discrepancy is target math, derivative precision, HMC
randomness/fixture mismatch, or harness failure.  Re-run only a localized
repair under the remaining budget and refresh P3 seeds/configuration.  Do not
change the target or silently substitute a different kernel.

### P3: short full-chain sensitivity (<= 900 s)

Run the shared full-chain runner for four chains, at least two fresh seeds, and
a short declared result count (mechanics diagnostic only).  Use the same state
bank and fixed kernel for strict and factor.  Check finite/status/divergence
vetoes, transition-level paired differences, movement and acceptance telemetry,
trace count, and allocator peak.  Report chain summaries as descriptive only;
do not calculate or interpret them as posterior convergence evidence.

Exit: all hard vetoes pass and no backend-sensitive transition failure remains.

### R3: repair/refresh closeout

Write the phase result and a decision/inference-status table.  If the candidate
fails, preserve it as rejected and refresh the next smallest repair.  If it
passes, refresh the exact scope-specific tuning plan and identity-bound artifact
requirements for P4.  A passing short chain does not itself authorize a
default change.

### P4: fresh scope identity and tuning admission fixture (<= 300 s)

Build strict and factor fixed-beta adapters from fresh constructions and verify
that a tuning/handoff payload made for strict is rejected by the factor
adapter's expected signature.  The positive measured-grid handoff path is
covered by the repository tuner regression; the standalone fixture must not
pretend that an incomplete synthetic payload is a valid tuning result.  Do not
tune on claim data.  This phase checks the provenance path that the previous
implementation omitted; it is not a full eight-pair tuning campaign.

Exit: no cross-backend artifact can pass identity validation, and the existing
measured-grid handoff regression remains green for a valid same-scope payload.

### R4: repair/refresh closeout and promotion decision

Combine P0--P4 into a terminal result note with separate engineering,
numerical, and scientific ledgers.  Promotion levels are:

* `KERNEL_CANDIDATE`: P0/P1 pass; factor may be used in further diagnostics,
  but strict remains the authority for HMC claims.
* `MECHANICS_CANDIDATE`: P0--P2 pass; factor may enter a reviewed short-chain
  sensitivity run, still not a default.
* `DEFAULT_ELIGIBLE_PENDING_FRESH_TUNING`: P0--P4 pass and P3 has no veto;
  a new exact-scope tuning artifact and untouched replay are still required.
* `PROMOTED`: only after a separately reviewed fresh tuning/untouched claim
  run under the factor identity and the parent program's default-readiness
  gates.  This plan alone does not grant that status.

If a gate is not completed because the budget ends, state `NOT_DECIDED`, not
pass.  Refresh the master program and write a reset memo before any replay or
default edit.

## Skeptical pre-run audit

The plan was audited before execution against the repository policy:

* Baseline: strict is the existing authority; the rejected raw-cache route is
  excluded and cannot contaminate the comparator.
* Target: both arms use the same q=20 bridge/data and differ only in the named
  eigensystem backend; backend identity is itself a hard gate.
* Metrics: parity and transition validity are promotion gates; speed,
  acceptance, displacement, and short-chain summaries are explicitly
  explanatory, so no proxy is silently promoted.
* Stopping: each phase has a cap, hard veto, repair action, and continuation
  veto; no failed candidate causes an unjustified research-direction stop.
* Statistical scope: the short chain has no convergence or ranking claim;
  multiple seeds are sensitivity evidence only.
* Environment: GPU0, memory growth before TensorFlow initialization, XLA, TF32,
  float64, no pfor, stable rank-2 batch signatures, and fresh output roots are
  explicit. CPU-hidden runs are reference diagnostics only.
* Staleness/provenance: historical M3 and raw-cache artifacts are not resumed;
  every promotion-level artifact must bind the repaired backend identity and a
  fresh scope.
* Pre-mortem: the run could appear successful while using a stale strict
  tuning artifact, a too-easy non-degenerate fixture, or identical random
  streams with no transition sensitivity. P0 identity checks, near-degenerate
  P1 fixtures, paired P2 telemetry, and fresh P4 scope checks expose those
  cases. It could also fail from compilation/resource limits rather than math;
  phase-specific manifests and CPU reference checks distinguish those causes.

This audit passes for the bounded campaign.  Execution begins with P0 only;
later phases require the stated closeout receipt.

## Exact commands

Focused CPU contract tests:

```bash
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=3 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_q20_bridge_identity.py \
  tests/test_tempered_transport_ensemble.py \
  tests/test_fixed_transport_hmc_step_cap.py
```

The positive same-scope handoff path is checked with the measured-grid tuner
regressions:

```bash
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=3 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_fixed_transport_hmc_tuning.py::test_replicated_efficiency_traverses_all_l_values_and_tunes_epsilon_independently \
  tests/test_fixed_transport_hmc_tuning.py::test_replicated_efficiency_policy_screens_every_ladder_nominee_then_holds_out_winner \
  tests/test_fixed_transport_hmc_tuning.py::test_verified_handoff_rejects_target_scope_substitution \
  tests/test_fixed_transport_hmc_tuning.py::test_verified_handoff_rejects_transport_substitution
```

The expanded parity command is (the benchmark enforces the plan-bound GPU0
selection and rejects a multi-device override):

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=3 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/benchmark_ssl_lstm_q20_eigh_reuse_2026_09_04.py \
  --output docs/plans/artifacts/ssl-lstm-q20-factor-route-promotion-2026-09-04/p1-gpu-attempt3/center.json \
  --batch-size 4 --repeats 2 \
  --candidate-backend tensorflow_eigh_strict_factor_cached
```

The varied GPU run uses `--row-offset-scale 0.02` and writes
`p1-gpu-attempt3/varied.json`; the CPU-hidden command adds `--cpu` and uses a separate `p1-cpu` directory.
the CPU-hidden command adds `--cpu` and uses a separate `p1-cpu` directory.
P2/P3 use the repository fixed-transport mechanics runner and write structured
JSON under `p2-transition/` and `p3-chain/`.  No command may reuse an existing
output path.

## Mandatory closeout fields

Every phase closeout records the actual command, wall time, Git revision,
environment/device policy, changed files, artifact hashes, failure
classification, remaining budget, focused regression result, refreshed
assumptions/defaults, next exact command, and whether the phase rejected only
the candidate or fired a continuation veto.

## Execution closeout

P0 through P4 completed on 2026-09-04.  The terminal result is
`docs/plans/bayesfilter-ssl-lstm-q20-factor-route-promotion-test-result-2026-09-04.md`.
The first P1 all-candidate invocation was preserved as a harness failure because
the unrelated raw-cache arm failed; the candidate-selectable benchmark repaired
that ambiguity and the fresh GPU0 artifacts then passed.  An additional P1
rerun enforced `CUDA_VISIBLE_DEVICES=0` after the first repaired GPU artifact
showed both devices visible; that rerun also passed and is the authoritative P1
receipt.  P2 and P3 passed with
native divergence telemetry explicitly unavailable in the installed TFP build,
using the frozen finite energy-error veto.  P4 proved stale strict handoffs are
rejected by the factor identity; its authoritative receipt is
`p4-identity-attempt3/admission.json`.  No continuation veto fired.  The required
follow-on plan for fresh factor tuning is
`docs/plans/bayesfilter-ssl-lstm-q20-factor-route-fresh-tuning-admission-plan-2026-09-04.md`.
