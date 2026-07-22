# SGQF Whole High-Dimensional Leaderboard Repair Result

Date: 2026-07-22

Status: `PASS_SGQF_COLUMN_COMPLETE; FULL_THREE_WAY_COMPARISON_NOT_READY`

Governing program:
`docs/plans/bayesfilter-sgqf-whole-highdim-leaderboard-repair-master-program-2026-07-22.md`

Authoritative column artifact:
`docs/benchmarks/artifacts/sgqf_whole_highdim_leaderboard_repair_20260722/attempt07/sgqf-column/result.json`

## Decision

The applicability-aware SGQF high-dimensional column is operational. All six
SGQF-applicable main rows emit their required result, and the parameterized-SIR
local complete-data component is explicitly `not_applicable` rather than
reported as a blocker. The artifact reports `sgqf_column_complete=true`.

This does not make the full UKF/Zhao--Cui/SGQF comparison ready. The
row-selective program intentionally does not execute those other algorithms,
and the slow generalized-SV Zhao--Cui TT compatibility test was stopped after
two bounded attempts with no result. No cross-method ranking is claimed.

## Decision Table

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| SGQF column completion | pass: all six applicable rows meet value/score kind | no target, identity, finite-value, covariance, or manual-score veto | actual/KSC/LGSSM lack fresh graph-native GPU/XLA evidence | retain column as complete; refactor those routes only if universal GPU/XLA is required | full comparison readiness |
| Source-timing reset | pass for SV, fixed SIR, and predator-prey | old initial-observation hashes rejected/quarantined | TensorFlow streams are not MATLAB `rng(1)` replay | keep source-model synthetic classification | author-stream reproduction |
| Generalized-SV route | pass for the declared scalar raw-y Gaussian projection | T1/T2, FD, dense refinement, CPU/GPU XLA all pass | Gaussian projection is an approximation | use only under its explicit route id | exact likelihood/posterior |
| Fixed SIR J=9, d=18 | pass as value-only | finite covariance, CPU/GPU XLA and identity pass | 37-point cloud has negative center weight and misses mixed fourth moments | keep score absent; use refinement/PF only as diagnostics | parameter score or superiority |

## Per-Row Results

| Row | Required kind | Status | Log likelihood | Score | Route identity / evidence |
| --- | --- | --- | ---: | --- | --- |
| LGSSM T50 | value+score | pass | `-136.0759748579247` | 5-vector, exact affine route | `539e0b437f07166427d6ff97d80862d9777a2e406ec8c997e2db39bc9ecdda5d`; no fresh GPU claim |
| Actual SV T1000 | value+score | pass | `-2303.576523800024` | `[2.8423172528, 1.8104302761]` | `c517ef670c9e20bb99a3a26ffa55395ef6b8c230a48996e823559015db2a21f7`; no fresh GPU claim |
| KSC SV T1000 | value+score | pass | `-2298.854329179004` | `[9.4787920316, -6.9518266802]` | `db1c94c807ac7cca71dd7cc4ba5269261db0b83aceb1a979dfbbd47d4f9fc9f9`; no fresh GPU claim |
| Fixed Austria SIR J=9, d=18, T20 | value only | pass | `-691.3692068263657` | not applicable | `a2371527017b1882e066059e1d146aaa79bed632ac8a1a028a25de67c55062df`; trusted GPU pass |
| Parameterized SIR local component | not applicable | pass/excluded | N/A | N/A | scoped Zhao--Cui-only row |
| Predator-prey T20 | value+score | pass | `-102.62270352134469` | physical 6-vector | `30b284e772bd49c12c65d1b1fffe3d0696e565fe8a82e02d6ccbfecbd13bd015`; trusted GPU pass |
| Generalized SV T1008 | value+score | pass | `-1437.9923718376479` | `[2.8304730706, 1.1148893545, -0.0686626853]` | `648b817c259506101c0ec6e9eefd9afe14c676db1ee08e7163b8f74edfc527f7`; trusted GPU pass |

Actual SV and KSC share one reset transition-first dataset identity:
`79677dd860af98af88b22a24fe9776f07729b007e739dc1fb4a26a1e9eadfd65`.
They do not share a likelihood target.

## Score And Refinement Evidence

| Row/check | Result | Gate |
| --- | ---: | --- |
| Actual SV full-horizon central FD max error | `2.7959132e-8` | pass |
| KSC SV full-horizon central FD max error | `5.4618567e-9` | pass |
| Predator-prey physical-coordinate FD | passed focused route tests | pass |
| Generalized SV full-horizon central FD max error | `1.4459088e-7` | pass `<5e-6` |
| Generalized SV level 3 vs level 5 value gap | `6.0507077e-5` | pass `<1e-4` |
| Generalized SV level 3 vs 41-point dense value gap | `6.0512109e-5` | pass `<1e-4` |

Actual/KSC FD artifact:
`docs/benchmarks/artifacts/sgqf_whole_highdim_leaderboard_repair_20260722/actual-ksc-source-reset-fd.json`.

## Engineering Ledger

| Item | Status | Evidence |
| --- | --- | --- |
| Row-selective master execution | pass | attempt07 JSON/Markdown, seven rows |
| Applicability schema | pass | value+score, value-only, and not-applicable terminal kinds |
| Repository-issued identities | pass | every executed SGQF row has a 64-character route identity |
| Fixed-SIR CPU/GPU XLA | pass | attempt02 CPU/GPU artifacts; GPU result `/GPU:0` |
| Predator-prey CPU/GPU XLA | pass | attempt01 CPU/GPU artifacts; GPU result `/GPU:0` |
| Generalized-SV CPU/GPU XLA | pass | attempt03; exact CPU/GPU value parity, score max gap `2.66e-15`, peak allocator `21504` bytes |
| Actual/KSC/LGSSM graph-native GPU XLA | not run | existing public routes are eager/Python-loop APIs; no false GPU claim issued |
| Focused test suite | pass | `32 passed`; later identity-column suite `3 passed`; `git diff --check` pass |

## Numerical And Scientific Ledgers

Engineering correctness is supported by deterministic replay, sealed hashes,
same-route value/score validation, focused tests, and the column artifact.

Numerical validity is supported by finite covariance/variance diagnostics,
manual FD, refinement, and CPU/GPU parity where graph-native routes exist.

Scientific interpretation is narrower. SGQF computes declared deterministic
approximations. Except for the affine LGSSM row, these values are not exact
nonlinear likelihoods. The evidence does not support a ranking against UKF,
Zhao--Cui, PF, GenUT, or any other method.

## Inference Status

| Inference question | Status |
| --- | --- |
| Hard veto screen | pass for all SGQF-applicable rows |
| Viable candidates | all six SGQF main-row routes |
| Statistically supported ranking | none; deterministic column execution is not a ranking experiment |
| Descriptive-only differences | all cross-method values, runtimes, tails, and legacy PF/UKF differences |
| Default readiness | not established |
| Next evidence needed for ranking | same-target complete comparator rows, predeclared uncertainty analysis where stochastic, and veto-first numerical checks |

## Attempts And Repairs

- Attempts 1--2 completed fixed SIR and predator-prey CPU/GPU evidence; failed
  serialization/artifact-write attempts were preserved where files existed.
- Generalized-SV CPU computation first failed only at sandboxed artifact
  creation, then passed with trusted repository write permission.
- Generalized-SV GPU attempt 1 exposed GPU-visible data/cloud identity drift.
  CPU-pinning fixture and cloud construction repaired it; GPU attempt 2 passed.
- SGQF column attempt04 failed analytical-provenance metadata validation.
  The already manual analytical routes were labeled precisely.
- Attempts05--06 passed but were superseded to add explicit not-applicable
  status, route identities, trusted GPU links, and the complete run manifest.
- Attempt07 is terminal.

## Post-Run Red Team

The strongest alternative explanation is that the routes are internally
consistent approximations but materially inaccurate for their nonlinear
targets. This result does not refute that explanation: FD and XLA verify the
implemented scalar, not the exact likelihood. Independent same-target dense or
particle references with uncertainty would be needed to assess accuracy.

The conclusion would be overturned by a source-timing error, data-hash
mismatch, manual-score failure, covariance failure, or evidence that a route
computes a different likelihood from its label. None was observed in the
checked SGQF column. The weakest remaining engineering evidence is the lack of
fresh graph-native GPU/XLA artifacts for LGSSM, actual SV, and KSC SV.

## Terminal Nonclaims

- not full three-way leaderboard readiness;
- not exact nonlinear filtering likelihood or posterior correctness;
- not SGQF superiority or statistically supported ranking;
- not HMC, production, or default readiness;
- not MATLAB random-stream reproduction; and
- not evidence about concurrent GenUT, transport, structural-UKF, or
  Zhao--Cui APF work, which was not modified by this lane.
