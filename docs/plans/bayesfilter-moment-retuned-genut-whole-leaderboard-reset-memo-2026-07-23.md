# Moment-Retuned GenUT Whole Leaderboard Reset Memo

Date: 2026-07-23

## Current State

The current six-row feasibility leaderboard is complete as an artifact at:

`docs/benchmarks/artifacts/moment_retuned_genut_whole_leaderboard_20260723/attempt05_final/result.json`

It contains 18 cells:

- 6/6 GenUT value/recursive-score cells executed;
- 6/6 fixed SGQF value/analytical-score cells executed;
- 5/6 fixed Zhao-Cui value/analytical-score cells executed; and
- Austria SIR Zhao-Cui is blocked because the observed-data marginal analytical
  score route does not exist. The local complete-data score is not admissible.

Result note:

`docs/plans/bayesfilter-moment-retuned-genut-whole-leaderboard-result-2026-07-23.md`

## Active Scope

Rows are LGSSM `T=50`, KSC-SV `T=10`, exact transformed SV `T=10`,
generalized SV `T=10`, source-order predator-prey `T=20`, and score-capable
Austria SIR `d=18`, `J=9`, `T=20`. Do not add the demoted reduced SIR mechanics
fixture or fixed value-only Austria SIR row.

Every GenUT row uses `N=1008`, FP32, TF32, XLA, GPU memory growth, 16 claim
seeds, and scope-specific offline tuning. The scalar/low-dimensional rows
selected:

```text
epsilon=2
sinkhorn_steps=8
balance_steps=8
ridge=1e-5
higher_moment_correction_steps=4
higher_moment_strength=0.2
higher_moment_floor=1e-5
```

Austria SIR selected:

```text
epsilon=8
sinkhorn_steps=16
balance_steps=16
ridge=1e-5
higher_moment_correction_steps=4
higher_moment_strength=0.2
higher_moment_floor=1e-5
```

Do not copy either setting to a new model/horizon/particle count. They are
scope-bound tuning results.

## Important Corrections

1. The LGSSM seed-81100 dataset observes stationary `x0` first. A route that
   transitions before the first observation is a different target.
2. The source-order predator-prey row is `x0 -> transition_1 -> y1` through
   `transition_20 -> y20`. The generic P8 `simulate(final_time=19)` observations
   are `y0..y19` and must not be substituted.
3. The positive Gaussian axis GenUT has signed central weight at `d=18`.
   Austria SIR uses a positive `2d` cubature residual design plus the tuned
   higher-moment correction. Do not call a signed design an OT mass.
4. KSC/exact-SV artifacts must be hash checked. Attempt 05 freshly evaluates
   the current SGQF/Zhao-Cui comparators after stale hashes were rejected.
5. SIR `epsilon=2` failed the unchanged quotient-column TV gate. The valid
   repair was scope-specific Sinkhorn tuning, not gate relaxation.

## Evidence Boundary

The matrix is a feasibility leaderboard, not evidence of superiority or a new
default. LGSSM value remains biased relative to the exact affine comparator.
Austria SIR value is stable but its recursive score has very high variance.
The predator-prey Zhao-Cui result is explicitly `extension_or_invention`.

The remaining Zhao-Cui SIR blocker is architectural: a valid fixed-variant
route needs an offline 18/36-dimensional TTSIRT proposal and the full
observed-data APF score closure. The existing local complete-data score and
demoted retained-grid route cannot fill the cell.

## Verification

- `21 passed` for:

```text
CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/highdim/test_cubature_genut_candidate.py \
  tests/highdim/test_cubature_genut_adapters.py \
  tests/highdim/test_higher_moment_contract_e.py
```

- `git diff --check` passed for the touched implementation, tests, runner,
  and plan.
- Attempt 05 source hashes and all six row checkpoints match the final result.
- TensorFlow peak live allocator use was about `128.3 MiB`.

## Next Work

The smallest justified next tasks are:

1. treat the current artifact as the baseline for the remaining leaderboard;
2. reduce Austria SIR recursive-score variance without sacrificing the finite
   value or changing the score target; and
3. keep Zhao-Cui SIR as a separate source-anchored architecture program rather
   than inventing a comparator inside the GenUT lane.

