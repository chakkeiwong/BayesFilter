# LGSSM N=2000/N=5000 Kalman Bias Ladder Plan

Date: 2026-07-20
Status: `CLOSED_N5000_BIAS_SMALLER_DESCRIPTIVELY_SCREEN_FAIL`
Parent reset memo:
`docs/plans/bayesfilter-lgssm-selected-controls-kalman-certification-reset-memo-2026-07-19.md`

> Operational correction, 2026-07-20: the size-one `N=5000` microbatch below
> records the conservative geometry used by this closed campaign; it is not an
> active recommendation. The follow-up capacity plan tests eight concurrent
> seeds on the same canonical scope:
> `docs/plans/bayesfilter-lgssm-n5000-seed-batch-capacity-plan-2026-07-20.md`.
> The result found batch eight memory-feasible but wrong relative to same-seed
> singleton value/score parity. Future runs must use the follow-up result:
> retain size one as a correctness fallback until the batched drift is repaired,
> without misrepresenting size one as a memory requirement.

## Research Intent Ledger

| Field | Declaration |
| --- | --- |
| Main question | At `T=50`, does increasing the canonical Contract E--Chol particle count from `N=1024` to independently tuned `N=2000` and `N=5000` materially reduce the finite-particle value and HMC-score bias against the exact differentiated Kalman likelihood? |
| Candidate/mechanism | More particles reduce finite-particle approximation error in the shared proposal, importance-weight, and Contract E reset recursion. |
| Exact baseline | The current-source `T=50,N=1024` selected-control certification result, especially `q_scale=-31.65%` mean relative error with simultaneous 95% interval `[-43.59%,-19.71%]`. |
| Promotion criterion | For each untouched 16-seed claim, all hard engineering gates pass; value relative-bias simultaneous 95% interval is inside `[-0.001,+0.001]`; every HMC-score relative-bias simultaneous interval is inside `[-0.05,+0.05]`. |
| Promotion veto | Any interval wholly outside its frozen region, or any hard engineering, scope, chunk, replay, finite, marginal, GPU/XLA, work-accounting, or target-identity failure. |
| Continuation veto | Invalid/corrupted target or artifact, wrong canonical route, inability to run the required policy chunk under the 8192 MiB limit even with seed microbatch size one, exhausted campaign budget, or a failed exact-scope tuning grid. Scientific failure at `N=2000` is not a continuation veto for `N=5000`. |
| Repair trigger | Resource failure above microbatch size one triggers a fresh versioned retry with a smaller fixed microbatch. Candidate marginal failure advances the declared cheaper-first grid. A localized harness/serialization failure triggers focused repair and fresh attempt without changing the scientific contract. |
| Explanatory diagnostics | Raw values/scores, absolute errors, per-seed SD/SE, Kalman predictive-score increment energy, OPG diagnostics, runtime, peak allocator bytes, and descriptive change from `N=1024`. |
| Must not be concluded | No `1/N` rate, method superiority, HMC/posterior readiness, parameter-region validity, nonlinear-model validity, universal controls, or new default follows from this ladder. |

## Evidence Contract

Each particle count is a separate tuning scope and must use its own repository-
issued scope identity and selected-control artifact. Selection is Kalman-blind:
only direct `TV_col <= 1e-4`, `E_row <= 1e-2`, finite value/total score, valid
chart/reset, exact work accounting, `StatelessWhile`, no Python horizon loop,
and exact GPU/TF32/XLA/chunk/scope identity may select controls.

After controls are frozen, the untouched claim is compared with the exact
Kalman value and HMC score using two-sided Bonferroni-Student intervals over six
outputs (value plus five score coordinates), 16 estimator seeds, critical value
`3.036283222821165`, value margin `0.001`, and score margin `0.05`.

Artifacts will be written without overwrite under:

`docs/benchmarks/artifacts/lgssm_particle_bias_ladder_20260720/`

Every launch receives a fresh attempt directory. The result artifact will
preserve exact commands, Git commit, source hashes, conda environment, GPU,
TF32/XLA, logical memory limit, particle/chunk geometry, seed partitions,
microbatch geometry, wall time, selected controls, and source artifact hashes.

## Frozen Scope And Seed Policy

| Scope | Required chunk | Calibration | Validation | Untouched claim |
| --- | --- | --- | --- | --- |
| `T=50,N=2000` | `K=2000`, `1 x 1` | `81900..81907` | `81908..81915` | `81920..81935` |
| `T=50,N=5000` | `K=2500`, `2 x 2` | `82000..82007` | `82008..82015` | `82020..82035` |

The `N=1024` controls `(20,8)` are a warm-start hypothesis only. For each new
scope, test `sinkhorn_steps=20` with the cheaper balance ladder
`3,5,8,12,16,25,32`; only after exhausting it may Sinkhorn advance through
`25,30,40`, with the full balance ladder repeated at every rung.

The initial resource probe uses one seed and `(20,8)`. Planned fixed seed
microbatch sizes are `4` for `N=2000` and `1` for `N=5000`, subject only to a
downward resource repair after a recorded failure. Microbatching changes the
supervisor/resource schedule, not the per-seed finite scalar. Aggregate value
and score are recomputed from the ordered concatenated per-seed outputs.

## Default And Assumption Audit

| Choice | Provenance/status | Justification | Failure mode | Early diagnostic |
| --- | --- | --- | --- | --- |
| Contract E--Chol total derivative route | Owner-mandated canonical route | Only reset/gradient identity eligible for claim evidence | Route drift would test a different scalar | Repository route IDs, preparation identity, source hashes |
| `T=50` observation prefix and theta | Exact prior certification target | Holds the scientific target fixed while changing `N` | Float64/original-data oracle mismatch | Recompute Kalman from float32-rounded production observations |
| `(20,8)` first useful resource probe | Warm start, not selected control | Previously passed direct gates at `N=1024,T=50` | Larger `N` can change terminal balance needs | One-seed probe, then blind full ladder |
| Balance-first grid | Prior T=50 tuning protocol | Balance is cheaper and directly addresses row residuals | Grid may omit needed controls | Exhaust full declared ladder before increasing Sinkhorn |
| 16 tuning plus 16 claim seeds | Reviewed prior protocol | Preserves calibration/validation/claim separation | Limited precision for small effects | Report simultaneous intervals and descriptive-only differences honestly |
| Microbatch `4` / `1` | Resource hypothesis | Keeps exact per-seed program within 8 GiB while preserving 16-seed evidence | Cross-microbatch aggregation or replay bug | Focused merge tests and same-seed parity at a small scope |
| Float32/TF32/GPU/XLA | Repository production target | Matches the route under certification | CPU/non-XLA evidence would answer a different question | Trusted GPU probe and artifact device/graph gates |
| 8192 MiB logical GPU limit | Existing campaign resource contract | Guarantees bounded sharing without whole-device preallocation | OOM at policy chunk | One-seed resource probe; size-one failure is continuation veto |

## Skeptical Plan Audit

Verdict: `PASS_AFTER_HARNESS_REPAIR`, so implementation and GPU execution may
begin only after the listed repair and focused checks pass.

- Wrong baseline: prevented by using the current-source `N=1024,T=50`
  certification and holding observations, theta, route, dtype, TF32/XLA, and
  thresholds fixed.
- Proxy promotion: direct marginal gates select numerical controls only;
  Kalman value/score are hidden until the untouched claim.
- Missing stop conditions: explicit resource, artifact, route, tuning-grid,
  and total-budget vetoes are declared; `N=2000` scientific failure does not
  incorrectly stop the discriminating `N=5000` rung.
- Unfair comparison: every `N` receives independent tuning and exact policy
  chunks. No cross-`N` selected setting is promoted.
- Hidden assumptions/defaults: particle count, chunks/block grid, microbatch
  geometry, seed partitions, controls, precision, and memory mode are explicit.
- Stale context: the current harness still hardcodes `N=1024` and assumes
  `K=N`; it must be repaired before launch.
- Artifact fitness: the repaired campaign must preserve per-microbatch nodes,
  ordered per-seed outputs, exact scope hashes, and a claim aggregate that is
  sufficient for the frozen simultaneous Kalman screen.

## Execution Budget And Stop Policy

- At most two versioned attempts per particle scope: one planned launch plus
  one localized infrastructure/resource retry.
- At most 12 GPU-hours total across probes, tuning, claims, and localized
  retries; at most 2 GPU-hours for one candidate or claim node.
- Preserve every completed or failed attempt. Never overwrite artifacts.
- Stop before `N=5000` only for a true continuation veto, not because the
  `N=2000` candidate remains biased.
- If `N=5000` is not `screen_pass`, do not launch nonlinear work. Write the
  required time-local active/no-reset/Kalman `q_scale` decomposition plan or
  result next.

## Preflight And Planned Commands

Before GPU execution: focused unit tests, Python compilation, `git diff
--check`, trusted `nvidia-smi`, and a trusted TensorFlow device/memory-policy
probe. CPU-only test commands must set `CUDA_VISIBLE_DEVICES=-1` before
TensorFlow import and are mechanics checks only.

The exact campaign commands will be recorded in each run manifest after the
harness repair. They must pass explicit `--num-particles`, fixed
`--seed-microbatch-size`, the seed starts above, the declared grids, this plan
path, and fresh output roots.

## N=5000 Claim-Repair Amendment

Date: 2026-07-20
Status: `READY_AFTER_SKEPTICAL_REPAIR_AUDIT`

The first executed `N=5000` tuning phase selected `(20,3)` using calibration
seeds `82000..82007` and validation seeds `82008..82015`. Its untouched claim
on `82020..82035` failed the direct row-marginal gate: seeds `82024`, `82027`,
and `82030` exceeded `E_row <= 0.01`, with worst `E_row=0.02010399`. Values and
total scores were finite, replay was bitwise exact, the policy chunk identity
was correct, and work accounting passed. This is a tuned-candidate rejection
and repair trigger, not a failure of the harness, target, canonical route, or
particle-scaling research question.

The failed claim is preserved as holdout evidence and is forbidden as a tuning
or selection input. One fresh repair phase is authorized within the unchanged
12 GPU-hour campaign budget:

- calibration seeds `82200..82207`;
- validation seeds `82208..82215`;
- untouched claim seeds `82220..82235`;
- fixed `T=50,N=5000,K=2500,2 x 2`, float32/TF32/GPU/XLA scope;
- fixed seed microbatch size one;
- start at `sinkhorn_steps=20` and balance ladder `5,8,12,16,25,32`;
- only after exhausting that ladder may Sinkhorn advance through `25,30,40`,
  repeating the full repair balance ladder at each rung; and
- selection remains Kalman-blind and uses the original direct gates.

The original two-attempt operational allowance covered the pre-candidate
harness retry and the first scientific tuning/claim. This amendment adds one
scientific claim-repair phase because the owner policy explicitly requires
fresh scope-specific tuning after a failed untouched claim and the total
campaign compute budget remains unchanged. No further `N=5000` claim-repair
phase is authorized by this plan.

Skeptical repair audit verdict: `PASS`.

- The failed claim is not reused, so there is no tuning on holdout data.
- The primary Kalman screen, direct gates, target, route, chunks, and hardware
  remain unchanged.
- Increasing balance follows the predeclared cheaper-control ordering and
  directly addresses the observed row-marginal failure; it is not tuning
  against Kalman bias.
- A failed repair grid or failed fresh claim closes `N=5000` as not validly
  certified and triggers the required time-local score decomposition, not a
  threshold relaxation or another seed transfer.
