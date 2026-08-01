# BayesFilter LGSSM NeuTra Scientific Gap-Closure Plan

> **2026-07-15 sequential-controller amendment:** The fixed 1,000 burn-in /
> 1,000 retained Phase 4 contract below is superseded by
> `docs/plans/bayesfilter-lgssm-neutra-sequential-hmc-repair-plan-2026-07-15.md`.
> Historical runs remain evidence, but their terminal no-admission decision is
> not valid under the corrected sequential warm-up and retained-sampling policy.

Date: 2026-07-15  
Status: `COMPLETE_EXACT_FIXTURE_PASS_ONE_OF_TWO_CANDIDATES`  
Supervisor/executor: Codex  
Active output root: `docs/plans/artifacts/lgssm-neutra-gap-closure-2026-07-15/`  
Training output root: `docs/plans/artifacts/neutra-batch-native-training-2026-07-14/long-training-attempt-01/`

## Decision And Supersession

The repository had a current plan for two fresh 5,000-step training jobs and an
older end-to-end scientific plan. It did not have one executable plan that
joined the current graph-native training artifacts to a policy-compliant HMC
campaign. This plan supplies that join and is the active continuation plan.

It preserves the scientific target, thresholds, seeds, selected recipe, and
plain-HMC comparator from:

- `docs/plans/bayesfilter-neutra-batch-native-training-fresh-5000-step-handoff-2026-07-14.md`;
- `docs/plans/bayesfilter-lgssm-neutra-target-specific-training-protocol-amendment-2026-07-14.md`; and
- `docs/plans/bayesfilter-lgssm-neutra-knowledge-transfer-and-serious-validation-plan-2026-07-13.md`.

It supersedes their active execution mechanics where they assume the old Phase
4 schema/root or import NumPy-backed HMC orchestration. Historical artifacts are
preserved and are not silently upgraded.

## Research Intent Ledger

| Field | Predeclared value |
| --- | --- |
| Main question | Can a frozen learned dense-IAF NeuTra transport for the exact 18-dimensional, T=120 LGSSM support independently tuned HMC that passes modern convergence, health, posterior-agreement, and recovery gates? |
| Candidate | Two fresh independent 5,000-step fits of the already nominated `wide_2x_lr5e3` recipe, seeds `(20260713,1201)` and `(20260713,1202)` |
| Mechanism under test | Reverse-KL dense-IAF NeuTra, composed as `T(z)=affine(dense_iaf_stack(z))`, followed by fixed-trajectory HMC in the frozen latent coordinates |
| Exact comparator | Immutable tuned plain-HMC result `docs/benchmarks/artifacts/multidim_lgssm_full_estimation_rerun_2026_07_13/final_recovery_result.json`, with its file SHA-256 frozen by Phase 0 |
| Promotion criterion | At least one independently trained candidate passes all frozen transport, tuning admission, confirmatory HMC, agreement, and recovery gates |
| Promotion vetoes | Identity drift; failed frozen reload/score parity; nonfinite target, score, sample, log acceptance, or diagnostic; target-status failure; predeclared energy-error divergence; tuning rank/folded R-hat above 1.01; confirmatory R-hat above 1.01; bulk ESS below 1000; tail ESS below 400; posterior mean disagreement above 4 combined MCSE; or any recovery distance above 3 posterior SD |
| Continuation vetoes | Invalid target/fixture, broken frozen transport semantics, corrupted artifacts, zero admitted candidates after the one declared tuning repair, unavailable trusted GPU for both training jobs, or exhausted campaign compute budget |
| Repair triggers | Localized schema, serialization, process, XLA, or infrastructure failure; a tuning candidate that is finite but fails its acceptance nomination or modern R-hat verification may use the one predeclared expanded grid |
| Explanatory diagnostics | Training/heldout reverse KL, gradient and clipping summaries, acceptance, energy-error magnitude, runtime, ESS, R-hat rows, posterior means/SDs/quantiles, and between-seed differences |
| Forbidden conclusions | Training loss or acceptance alone proves nothing about posterior validity; one fixture does not prove calibration, robustness, or generality; descriptive runtime/ESS differences do not rank methods; a passing candidate does not establish sampler superiority or a repository default |

## Evidence Contract

The exact baseline is the tuned plain-HMC result above, not an untuned or weak
HMC arm. Tuning admission requires 4 chains with 1,000 retained results per
chain after 1,000 burn-in steps and the maximum of rank-normalized split R-hat
and folded rank-normalized split R-hat no greater than 1.01 for all 18
parameters. Acceptance nominates a step size inside a finite grid but cannot
admit it without the modern R-hat verification.

Confirmation uses a fresh seed, 4 chains, 4,000 retained results per chain,
and 1,000 burn-in steps. It requires:

- maximum modern R-hat `<= 1.01`;
- minimum bulk ESS `>= 1000`;
- minimum tail ESS `>= 400`;
- no nonfinite or target-status failure;
- no predeclared HMC energy-error divergence (`log_accept_ratio < -1000`);
- all posterior-mean differences from tuned plain HMC `<= 4` combined MCSE;
- all truth-recovery distances `<= 3` candidate posterior SD.

The energy-error rule is an explicit TensorFlow/TFP HMC numerical-divergence
screen because this TFP HMC result does not expose a native Stan-style
divergence flag. It must be reported as that screen, not as a claim that a
native divergence diagnostic was available.

The terminal result is
`docs/plans/bayesfilter-lgssm-neutra-scientific-gap-closure-result-2026-07-15.md`.
Raw retained tensors use TensorFlow serialization in candidate-private
versioned directories. JSON summaries contain no raw sample arrays.

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Earliest diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| `wide_2x_lr5e3` | Four-arm 500-step screen, selected recipe artifact SHA-256 `1984c33142496ecbbd77ecaea17b1d3dc3320caa45a1b08aa947439ca7088c97` | Predeclared proxy nomination from four engineering-valid recipes | Short proxy selects a transport with poor HMC geometry | Frozen HMC tuning and confirmation | nominated hypothesis, not a proven default |
| 5,000 steps, batch 128 | Source NeuTra program and target-specific amendment | Serious source-grounded baseline after short-budget migration | Still undertrains or wastes budget on this target | two independent seeds and downstream HMC | transferred baseline requiring target evidence |
| Two training seeds | Parent campaign | Detects gross seed-specific failure within bounded compute | Too few seeds for population ranking | report both; forbid superiority ranking | bounded validation choice |
| Fixed affine-last composition | Prior mathematical/parity work | Preserves the trained finite program and checked convention | Restore or order drift changes the target | compiled forward/logdet/score parity | reviewed fixed choice |
| HMC leapfrog count 10 | Older target-specific HMC plan | Keeps tuning one-dimensional and bounded | Fixed trajectory family misses useful geometry | primary and repair grids plus modern R-hat | inherited hypothesis, not a default |
| Step-size grid | Older plan: base 0.1, primary scales `(0.25,0.5,1,2,4)`; repair `(0.125,0.25,0.5,1,2,4,8)` | Bounded grid spans a 64x step range | None passes or boundary candidate indicates insufficient range | finite 64-result probes, then 1,000-result R-hat verification | predeclared tuning hypothesis |
| Acceptance band 0.60-0.90 | Older plan | Rejects obviously poor finite proposals before expensive verification | Acceptance is near 1 while movement is poor | rank/folded R-hat is the admission gate | nomination-only diagnostic |
| Modern diagnostic thresholds | Older plan and plain-HMC campaign | Conventional strict screen and exact continuity with comparator | Short chains pass by chance | fresh 4,000-draw confirmation | promotion gates |
| Truth-centered affine geometry | Exact fixture and mass artifact | Preserves same-target continuity | Makes the problem unusually favorable | terminal limitation/nonclaim | fixed favorable-fixture limitation |
| CPU/XLA batched four-chain HMC | Repository NeuTra execution policy | HMC sample generation remains batched and uses TensorFlow CPU parallelism while training stays GPU | CPU/XLA incompatibility or thread oversubscription | Gaussian XLA fixture and one frozen short smoke | implementation hypothesis to prove in Phase 0 |
| Plain-HMC JSON summary comparator | Immutable final recovery result | Avoids importing legacy NumPy archives while preserving checked means/SD/MCSE | Summary identity or parameter ordering drifts | file hash, schema, 18 names, and finite-field checks | exact historical comparator summary |

## Skeptical Plan Audit

Audit date: 2026-07-15. The initial idea of simply resuming the old Phase 5/6
harness failed this audit for two material reasons:

1. the current strict trainer emits a new graph-native result schema and a new
   output root that the old `finalize_phase4` cannot consume; and
2. the old HMC tuner, sampler, archive, and posterior summary import and execute
   NumPy in active inference and admission paths, contrary to the owner policy.

This revision fixes those problems before expensive execution. It does not use
heldout loss as a posterior promotion metric, does not treat acceptance as
convergence, retains a tuned plain-HMC comparator, preserves distinct tuning
and confirmatory seeds, and states actual continuation vetoes. A command can
answer the stated question only after Phase 0 proves schema consumption,
TensorFlow-only import closure, Gaussian sampler mechanics, rank/folded R-hat
behavior, tensor archive round-trip, and a short frozen-candidate XLA smoke.

Pre-mortem: a command could succeed while using the wrong restored transport,
producing immobile high-acceptance chains, comparing parameters in the wrong
order, hiding a nonfinite/status event, or reading a stale comparator. The
earliest checks are exact artifact hashes, deterministic compiled score parity,
Gaussian movement/R-hat fixtures, per-transition health reductions, parameter
name equality, and versioned no-overwrite roots.

Audit verdict: `PASS_AFTER_PHASE0_ENGINEERING_GATE`; expensive training is not
admitted until Phase 0 closes.

## Compute And Attempt Budget

- Phase 0: local CPU tests and smokes, at most 20 minutes wall time; one trusted
  GPU frozen-mechanics smoke of at most 8 transitions only if a compatible
  existing payload is available.
- Training: two sequential 5,000-step trusted GPU/XLA jobs, aggregate compiled
  time ceiling 45 minutes and aggregate wall ceiling 60 minutes; one localized
  infrastructure retry per seed into a new attempt root, with no checkpoint or
  screen-weight reuse.
- Tuning: per frozen candidate, primary five-point 64-result/128-burn-in probes,
  one selected 1,000-result/1,000-burn-in modern verification, and at most one
  predeclared seven-point repair grid. Aggregate CPU wall ceiling 6 hours.
- Confirmation: one fresh 4,000-result/1,000-burn-in run per admitted candidate,
  aggregate CPU wall ceiling 6 hours. No same-candidate confirmatory rerun.
- No package/environment mutation, network fetch, paid compute, or public act.

Localized infrastructure repairs remain inside this authorized campaign when
the target, recipe, seeds, thresholds, hardware class, and total budget do not
change. Every attempt writes a new versioned directory and preserves failures.

## Phases

### Phase 0 - Compatibility And Policy Repair

Implement a narrow LGSSM NeuTra campaign route that:

- validates and consumes the strict training result schema;
- performs frozen GPU/XLA forward, logdet, target-value, and score parity;
- performs batched four-chain CPU/XLA HMC directly with TensorFlow/TFP;
- tunes on a fixed finite grid and admits only after fresh modern rank/folded
  R-hat verification;
- serializes retained tensors with TensorFlow and constructs JSON with the
  standard library;
- computes health, convergence, posterior agreement, recovery, and MCSE with
  TensorFlow/TFP; and
- never imports the legacy NumPy-backed HMC/tuning/orchestration modules.

Exit: focused tests, import-closure audit, Gaussian XLA sampler diagnostic, and
strict-result adapter fixture pass. The frozen-candidate smoke may be deferred
until the first fresh payload exists; in that case Phase 1 may train but Phase 3
cannot close until the smoke passes.

### Phase 1 - Fresh Seed 1201 Training

Run the exact command in the active 5,000-step handoff for
`dense_seed1201`. Verify selected recipe identity, GPU/memory-growth/XLA state,
one compiled training invocation, all-finite and target-status records, terminal
checkpoint/payload hashes, and frozen reload/score parity.

Exit: write the seed close record and refresh Phase 2 from observed timing and
remaining aggregate budget.

### Phase 2 - Fresh Seed 1202 Training

Run the corresponding fresh `dense_seed1202` job with no state reuse. Apply the
same gates and report between-seed training quantities as descriptive only.

Exit: both strict training results are immutable and at least one is eligible
for frozen validation. A single rejected seed does not stop the other seed or
the research direction.

### Phase 3 - Frozen Transport Validation And Handoff

Consume both strict result schemas, validate hashes and target identity, reload
each transport, and run deterministic GPU/XLA and CPU/XLA fixed-objective
probes. Freeze candidate records and all HMC seed families before tuning.

Exit: at least one candidate has exact reload and cross-device value/score
parity within the established tolerances and no identity/status failure.

### Phase 4 - NeuTra-HMC Tuning Admission

For every Phase 3 candidate, run finite-grid tuning in batched four-chain
CPU/XLA mode. Use acceptance only to nominate a finite step size. Admit only a
fresh 1,000-result verification for which every parameter's maximum rank/folded
split R-hat is `<=1.01`, health/status gates pass, and the energy-error
divergence count is zero. Use the one repair grid only for a finite
acceptance/R-hat failure.

Exit: freeze all admission decisions and selected kernel hashes before any
confirmation. Zero admitted candidates is a real campaign stop.

### Phase 5 - Confirmatory HMC

Run each admitted candidate once with its immutable kernel and fresh serious
seed. Preserve raw and transformed retained draws as TensorFlow tensor shards.
Compute modern R-hat, bulk/tail ESS, health/status/divergence, posterior
agreement, recovery, and uncertainty diagnostics.

Exit: all pre-admitted candidates have confirmatory results; at least one must
pass every promotion gate for a positive NeuTra-on-this-LGSSM verdict.

### Phase 6 - Scientific Closeout

Write the decision and inference-status tables, run manifest, uncertainty-aware
interpretation, post-run red team, and reset memo. Clearly separate candidate
rejection from research-direction rejection.

Permitted positive claim: a specific frozen dense-IAF seed supports tuned HMC
on this exact favorable 18D LGSSM fixture under the recorded gates. Forbidden:
superiority, calibration, robustness, generality, production readiness, or new
default claims.

## Mandatory Phase Procedure

At the end of every phase:

1. run the required local checks;
2. write a phase result/close record;
3. draft or refresh the next subplan from actual evidence and remaining budget;
4. review that subplan for scientific suitability, schema/artifact coverage,
   feasibility, stale assumptions, and boundary safety; and
5. continue when no real target, numerical, artifact, hardware, privacy, or
   budget blocker exists.

A phase subplan must state objective, inherited entry conditions, artifacts,
checks/reviews, evidence contract, forbidden claims/actions, exact handoff, and
stop conditions. Reviewer availability is advisory and cannot turn a
procedural issue into a scientific blocker.

## Stop Conditions

Stop only for a continuation veto, total budget exhaustion, a request that
changes the scientific contract, or an external/irreversible boundary. A failed
candidate, proxy loss, acceptance miss, or localized infrastructure defect is
not by itself a reason to abandon the next planned repair or the other seed.
