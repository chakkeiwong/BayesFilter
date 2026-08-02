# NeuTra Remaining-Models Broad-Grid Continuation Plan

Date: 2026-07-30

Status: `SUPERSEDED_SIR_UKF_OWNER_EXCLUDED`

Supersession, 2026-07-31: the owner determined that UKF does not work for SIR
and removed `SIR-UKF` from testing. The historical admission work below is
preserved as provenance only. It must not be executed or used as a reentry plan.
The generic broad-grid machinery remains available for eligible models.

## Objective

Continue the BayesFilter NeuTra experiment without rerunning completed cells.
First make the approved broad fixed-mass `L`/epsilon procedure a reusable
master-runner path. Then re-evaluate the smallest unresolved target-admission
gap, `SIR-UKF` CPU versus trusted GPU/XLA score parity. Serious `SIR-UKF`
training begins only after both gates pass and a phase-specific compute budget
is recorded.

## Research Intent Ledger

| Field | Binding decision |
| --- | --- |
| Main question | Can the reusable NeuTra campaign advance a currently unresolved model using the approved non-directional broad-grid tuning protocol without rerunning completed cells or transferring another model's settings as defaults? |
| Candidate mechanism | Current graph-native `SIR-UKF` observed-data posterior adapter, followed only after admission by target-specific batched GPU/XLA NeuTra training and fixed-identity NeuTra-coordinate HMC |
| Expected failure mode | The historical `SIR-UKF` GPU/CPU score gap persists; the generic broad-grid wrapper drifts from independent per-primary epsilon tuning; target/status telemetry is incomplete; or later target-specific training/tuning fails locally |
| Promotion criterion | Phase 1: generic broad-grid contract tests pass. Phase 2: current `SIR-UKF` value/status parity passes and score scale-normalized gap is `<=1e-7` on the frozen audit points. Later phases require separately recorded training, HMC, convergence, health, and same-target comparator gates. |
| Promotion veto | Shared/frozen-epsilon primary screen, missing primary barrier, recursive coverage, coverage failure vetoing a viable parent, parity gap `>1e-7`, nonfinite values/scores, status mismatch, wrong device, missing XLA/memory-growth provenance, or target-identity drift |
| Continuation veto | Generic harness invalidity, current parity failure with no localized repair, output collision, GPU/XLA/memory-growth failure, or phase budget exhaustion |
| Repair trigger | Local wrapper, telemetry, serialization, device-parity, or numerical-order failure under the unchanged target and threshold |
| Explanatory diagnostics | Exact parity gap magnitude, runtime, acceptance estimates, tuned epsilon values in private tuning artifacts, training loss, and later ESS/R-hat values outside their declared gate role |
| Must not be concluded | Exact-filter correctness, epidemiological calibration, cross-model superiority, universal NeuTra validity, default readiness, or that a blocked model is a failed NeuTra candidate |

## Current Inventory

| Cell | Current state | Action in this campaign |
| --- | --- | --- |
| `LGSSM-EXACT` | completed two-seed sampler diagnostic | do not rerun |
| `PP-UKF` | completed HMC and same-target distributional compatibility | do not rerun |
| `PP-SGQF` | completed narrow same-target mean confirmation | do not rerun unless a later broader claim is explicitly selected |
| `SIR-SGQF` | completed narrow same-target mean confirmation | do not rerun unless a later broader claim is explicitly selected |
| `STR-UKF` | qualified one-seed truth-tail pass | do not rerun in this campaign |
| `SIR-UKF` | historical GPU score-parity blocker | first unresolved admission lane |
| other blocked cells | filter, source-route, or observed-data-score blockers | remain blocked inventory |

## Evidence Contract

### Generic tuning integration

- Route: `operational_broad_fixed_mass_l_epsilon_grid_v1`.
- Primary grid: `L=(3,5,9,13,18,25)`.
- Every primary receives an independent fixed-mass dual-averaging epsilon tune.
- Each primary is screened on three fresh replications and classified using the
  frozen 90% working interval over replication means.
- Every viable primary contributes nonrecursive `L-1` and `L+1` coverage
  requests within the reviewed bounds. Coverage inherits the exact parent
  epsilon and does not retune.
- A failed coverage pair never removes a viable primary.
- The next-round result is the complete unranked union of viable primaries and
  viable coverage pairs after both barriers complete.
- All tuning draws are discarded. Broad-grid survival is not convergence or
  retained-sampling evidence.

### `SIR-UKF` admission

- Frozen observed data and audit points must match the existing target-design
  fixture.
- CPU reference and trusted GPU/XLA calculation must use the same current
  callable and float64 inputs.
- Value scale-normalized gap must be `<=1e-8`.
- Score scale-normalized gap must be `<=1e-7`.
- Status codes must match exactly; values and scores must be finite and GPU
  resident in the trusted run.
- TensorFlow memory growth must be configured and verified before logical GPU
  initialization.
- The parity artifact is target admission evidence only. It is not training,
  HMC, or posterior evidence.

## Phases

1. **Inventory and stale-contract audit**
   - Record completed cells and current blockers.
   - Confirm the master currently calls the wrong anchor-only public route for
     this campaign question.
2. **Generic broad-grid integration**
   - Extract the existing PP-UKF TensorFlow/TFP callback mechanics into a
     target-agnostic repository module.
   - Add a master CLI action for preserved-transport broad-grid tuning.
   - Keep the existing public-tuner replay path readable for historical
     artifacts, but do not use it as the new-model campaign route.
   - Add contract tests for independent epsilon, exact grid, one-hop coverage,
     complete barriers, parent preservation, fixed identity mass, and no local
     sampler/diagnostic reimplementation.
3. **Focused `SIR-UKF` parity recheck**
   - Produce a fresh CPU reference using deliberate GPU hiding.
   - Produce a fresh trusted GPU/XLA canary against that exact reference.
   - Preserve the historical `1e-7` score gate unchanged.
   - Attempt 01 reproduced the historical blocker: value normalized gap
     `2.1697897501161396e-09` passed, but score normalized gap
     `5.966165369939377e-07` failed. Status, finiteness, and GPU residence
     passed.
   - A CPU execution-mode discriminator isolated the failure to XLA rather
     than GPU hardware: eager versus non-XLA graph was
     `2.3859923510794413e-14`, while eager versus CPU XLA was
     `5.96616501950473e-07`, essentially identical to GPU XLA.
   - Localized repair rung: record eager/non-XLA/CPU-XLA results for every
     observation prefix and score component; identify the first divergent
     recursion step; then test the smallest mathematically equivalent
     cancellation- or accumulation-order repair. Do not disable XLA, change
     the UKF target, relax the threshold, or rely on cancellation at the final
     horizon.
   - Prefix localization completed over all 20 horizons. The first score-gate
     failure occurred at `T=5`; the maximum eager/non-XLA normalized score gap
     was `1.0036416142611415e-13`, while the maximum eager/CPU-XLA gap was
     `7.6010498553102934e-06` at `T=17`. Values remained within
     `3.4287957729570304e-09`. This rejects terminal score accumulation as the
     sole cause.
   - Isolated SIR RK4 value, state Jacobian, and source Jacobian parity was
     within `8.6e-16`; an independent reverse-cotangent UKF score reproduced
     the forward-sensitivity XLA gap. These diagnostics localize the drift to
     the shared XLA principal-root forward path rather than the model
     derivatives or analytic score direction.
   - Selected repair candidate: an opt-in, fixed 24-iteration TensorFlow/XLA
     Newton-Schulz principal root with the existing Sylvester derivative. It
     computes the same strict-SPD principal root and remains subject to the
     existing reconstruction and derivative residual gates. A CPU diagnostic
     changed the eager score by at most `3.188579969711777e-14` normalized and
     reduced CPU-XLA score drift to `1.1843044423021696e-09`. This backend is
     selected only by `SIR-UKF`; no global default changes.
4. **Conditional target/registry admission**
   - If Phase 3 passes, issue the current direct target signature, add
     `SIR-UKF` to the executable registry, and mark completed cells separately
     from executable and blocked inventory.
   - If Phase 3 fails, localize the numerical gap under a bounded repair plan;
     do not train.
5. **Conditional target-specific NeuTra campaign**
   - Requires a refreshed phase budget before launch.
   - Screen a target-specific architecture/optimizer family, run one fresh
     selected 5,000-step batched GPU/XLA transport, run the generic broad-grid
     tuning route, validate the complete next-round candidate set with the
     shared sequential controller, and compare valid retained samples with a
     same-target plain-HMC reference using prospective uncertainty criteria.

## Compute And Attempt Budget

- Phases 0-2 engineering checks: CPU only, at most 30 minutes wall time.
- Phase 3 parity: one CPU reference and one trusted GPU/XLA canary, at most 30
  minutes total, plus one localized infrastructure retry if no numerical
  result was produced.
- Phase 3 localized numerical repair: at most three CPU diagnostic/repair
  attempts and one fresh trusted GPU/XLA canary after CPU XLA passes, at most
  60 minutes total. A candidate repair must preserve eager/non-XLA value,
  score, and status behavior while meeting the frozen CPU XLA score gate.
- No target-specific training, broad-grid GPU tuning, or retained HMC is
  authorized until Phase 3 has a terminal result and Phase 5 records a bounded
  serious-run budget based on measured `SIR-UKF` throughput.
- Every attempt uses a fresh output root and preserves prior evidence.

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| Broad primary grid | owner-approved PP-UKF protocol and `hmc_operational_broad_grid.py` | covers `L=3..25` non-directionally with six primary values | another target may require different resolution | preserve full primary outcomes and treat coverage/ranking as nonclaims | reviewed campaign route, not universal optimum |
| Per-primary epsilon tuning | corrected PP-UKF diagnosis | acceptance depends on both `L` and epsilon | shared epsilon creates false rejection or acceptance | callback-lineage tests and per-primary tune call count | required |
| One-hop same-epsilon coverage | owner interpretation | fills grid holes without pretending to retune neighbors | coverage could be mislabeled as a parent veto | next-round union tests | required coverage role |
| `SIR-UKF` parity tolerance | frozen P6 prospective gate | avoids post-result threshold changes | device roundoff may exceed the strict gate despite a mathematically consistent score | report exact gap and localize; do not relax after observing | binding admission gate |
| Existing `SIR-UKF` adapter | current graph-native target code | smallest unresolved cell with an implemented observed-data adapter | historical blocker may persist or target code may have drifted | CPU/GPU value, score, status, signature checks | candidate, not admitted |
| Training recipe | not yet selected | target-specific protocol is required | inherited SIR-SGQF or PP settings fail for the wrong reason | Phase 5 architecture/optimizer plan before training | deliberately unresolved |

## Skeptical Pre-Execution Audit

- **Wrong baseline:** completed cells are excluded; `SIR-UKF` is compared with
  the same current callable on CPU, not with `SIR-SGQF` or a Zhao-Cui route.
- **Proxy promotion:** parity can admit target execution only. It cannot admit
  training, HMC, convergence, or posterior validity.
- **Missing stops:** phase time limits, fresh roots, parity thresholds, device
  requirements, and no-training boundary are explicit.
- **Unfair comparison:** CPU and GPU use identical float64 audit points,
  observations, value/score callable, and status definitions.
- **Hidden assumptions:** the strict `1e-7` threshold and broad grid are
  recorded as reviewed campaign choices, not mathematical truths.
- **Stale context:** LGSSM, PP-UKF, PP-SGQF, SIR-SGQF, and STR-UKF later
  terminal evidence supersedes the July 18 all-model inventory.
- **Environment mismatch:** CPU explicitly hides CUDA; GPU parity is run only
  with trusted/escalated GPU access and verified memory growth.
- **Non-answering command:** the Phase 3 commands emit exact value/score/status
  parity fields and manifests, not only device visibility.
- **Misleading pass:** a parity pass can still precede poor training or a wrong
  approximate filter; later phases retain independent target, training,
  sampler, and same-target comparator gates.

Audit verdict: `PASS_FOR_PHASES_0_TO_3_ONLY`. Phase 5 is intentionally not yet
executable because a target-specific training/default audit and measured
serious-run budget do not exist. This is a scientific boundary, not a
procedural approval token.

## Planned Artifacts

```text
docs/plans/artifacts/bayesfilter-neutra-remaining-models-20260730/
  sir-ukf-parity-cpu-attempt-01/
  sir-ukf-parity-gpu-attempt-01/
  sir-ukf-cpu-xla-localization-attempt-01/
  sir-ukf-parity-cpu-attempt-02/
  sir-ukf-parity-gpu-attempt-02/
```

The terminal Phase 3 result will update this plan and a reset memo with either
`SIR_UKF_TARGET_EXECUTION_ADMITTED` or the exact current blocker and repair
rung.
