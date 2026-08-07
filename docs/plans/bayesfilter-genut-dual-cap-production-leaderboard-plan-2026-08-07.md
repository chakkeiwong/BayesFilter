# GenUT Dual-Cap Production Leaderboard Plan

Date: 2026-08-07  
Status: `AUDITED_AND_AUTHORIZED_FOR_EXECUTION`

## Research Intent

Make the owner-selected dual-cap GenUT family a repository-owned selectable
default, then regenerate the six-model same-target feasibility leaderboard on
GPU/XLA. The Austria SIR Zhao-Cui cell is outside this campaign because another
agent owns that implementation program.

The claimed target is the finite GenUT value program and its complete manual
forward sensitivity under the selected controls. Approximate SGQF and Zhao-Cui
cells are comparators, not nonlinear truth oracles.

## Evidence Contract

| Field | Frozen decision |
|---|---|
| Models | LGSSM `T=50`; KSC SV `T=10`; exact transformed SV `T=10`; generalized SV `T=10`; predator-prey `T=20`; Austria SIR `T=20` |
| Particle count | `N=1008` |
| Baseline | Scope-tuned diagonal GenUT from the July same-target campaign |
| Candidate | `dual_cap`: diagonal correction plus four pairwise steps, radial RMS cap `2`, standardized coordinate cap `b=.98,p=8`, affine restoration |
| Pairwise strength grid | `0.02, 0.05`; all other dual-cap controls are frozen. Select the hard-valid arm with minimum validation mean normalized pairwise residual; break ties by validation scaled value/score variance, then calibration residual, then lower strength. |
| Calibration/validation | Existing disjoint generated datasets in the six-row harness; claim observations are not read during selection. Tuning seeds are `98501,98502` for the first five scopes and `98301,98302` for Austria, matching its frozen T20 scope artifact. |
| Claim seeds | `98201..98216`, common between diagonal and dual-cap |
| Runtime | TensorFlow FP32, TF32 enabled, GPU, XLA, verified memory growth |
| Primary criterion | All six candidate rows finite/program-valid and pass reset, score-additivity, displacement, and cap-output gates |
| Promotion veto | Any candidate row fails a hard validity gate, selector semantics are ambiguous, or explicit historical options change behavior |
| Continuation veto | Target hash/event-order mismatch, invalid diagonal baseline, missing GPU/memory-growth evidence, corrupt artifact, or exhausted attempt budget |
| Explanatory diagnostics | Paired value/score changes, seed SD and MCSE, cap activity, exact/approximate comparator gaps, centered finite differences |
| Artifact root | `docs/benchmarks/artifacts/genut_dual_cap_production_leaderboard_20260807/attempt01/` |

The campaign may establish a reproducible production leaderboard implementation
and six-row feasibility. It will not establish exact nonlinear score
correctness, statistical superiority, posterior validity, HMC/NeuTra readiness,
or Zhao-Cui Austria completion.

## Research-Question Guardian

| Role | Diagnostic |
|---|---|
| Promotion criterion | Six valid dual-cap GenUT rows plus selector and replay tests |
| Promotion veto | Candidate hard-validity failure or changed explicit-option semantics |
| Continuation veto | Broken target/baseline/environment/artifact assumptions |
| Repair trigger | Local harness, serialization, import, or XLA infrastructure failure under unchanged scope |
| Explanatory only | Comparator proximity, FD residuals, cap activity, runtime, and descriptive cross-method gaps |
| Must not be concluded | Dual-cap is universally superior or its nonlinear score is exact |

## Default And Assumption Audit

| Choice | Provenance | Status | Failure mode | Early diagnostic |
|---|---|---|---|---|
| `b=.98,p=8`, radial `2` | Four-model campaign and owner decision | reviewed family default | cap changes most coordinates or shifts value | activity, displacement, paired value/MCSE |
| Four pairwise steps | Four-model campaign | reviewed family default | over-correction in a new scope | disjoint calibration validity and paired claim rows |
| Strength grid `.02,.05` | prior model campaigns | bounded tuning hypothesis | selected by noisy proxy | validation before claim; preserve all candidates |
| July diagonal controls | same target/horizon/`N` baseline artifact | baseline | stale implementation or target mismatch | hash/event-order check and fresh baseline execution |
| TF32 | repository execution target | execution default | derivative cancellation at small FD step | multi-step FD ladder, explanatory unless exact oracle exists |
| Sixteen seeds | prior leaderboard contract | descriptive replication | insufficient power for ranking | report SD/MCSE and no superiority claim |

## Skeptical Plan Audit

- **Wrong baseline:** controlled by exact target hash, event order, horizon,
  particle count, and fresh diagonal execution.
- **Proxy promotion:** moment residual and comparator proximity are explanatory;
  only hard program validity and reproducible selector wiring promote the route
  into this feasibility leaderboard.
- **Unfair comparison:** diagonal and dual-cap use common target data, particle
  count, arithmetic, and claim seeds. The method ladder is preserved.
- **Hidden implementation mix:** production changes are built in a clean
  worktree from `origin/main`; bounded-teacher and Austria-adapter research
  changes are explicitly excluded.
- **Stale controls:** inherited values seed only a two-value pairwise-strength
  grid, selected on disjoint validation data for each scope.
- **Missing stop conditions:** candidate failure blocks candidate promotion but
  preserves all other rows; target/environment invalidity stops the campaign.
- **Artifacts answering the question:** raw per-seed rows, tuning candidates,
  selector identity, source hashes, GPU/memory policy, FD ladder, and paired
  summaries are retained.

Audit decision: `PASS_AFTER_ISOLATING_THE_PRODUCTION_DIFF`.

## Implementation And Execution

1. Add a repository-owned algorithm selector with stable names. `default`
   resolves to `dual_cap`; explicit historical names retain exact controls.
2. Port only the radial and standardized-coordinate cap value/JVP operations,
   affine restoration, and diagnostics into the clean core.
3. Add unit tests for selector resolution, opt-in compatibility, affine moment
   restoration, cap bounds, and manual-JVP parity.
4. Run focused CPU tests with GPU hidden, then an escalated GPU/XLA smoke.
5. Tune the two pairwise strengths independently on each of six scopes and run
   16 common claim seeds for diagonal and dual-cap.
6. Audit hard vetoes first, then paired uncertainty and derivative evidence;
   write a result note and commit/push the bounded change.

## Budget And Stop Conditions

Budget one focused test pass, one six-model GPU campaign, and one localized
infrastructure retry. Maximum campaign GPU wall time is 45 minutes. Every retry
uses a fresh attempt directory. Do not expand the tuning grid after seeing
claim results. Stop on a continuation veto or budget exhaustion.
