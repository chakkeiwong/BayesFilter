# SVX-ZC Capacity Self-Convergence Tuning Execution Plan

Date: 2026-08-01

Parent protocol:
`docs/plans/bayesfilter-capacity-self-convergence-likelihood-tuning-plan-2026-08-01.md`

## Objective

Implement the generic likelihood-value capacity tuner and run one bounded
SVX-ZC experiment to answer two questions:

1. Does the tuner correctly identify three-significant-digit likelihood
   stability without an exact or dense likelihood reference?
2. Does the current SVX-ZC fixed-adjacent squared-TT route contain a finite
   degree/rank/quadrature configuration that is self-converged under that rule
   at the center and declared validation parameter points?

The experiment tunes likelihood values only. It must not compute empirical
score deltas, execute the score route, run finite differences, or use HMC.

## Research intent ledger

| Field | Declaration |
|---|---|
| Main question | Does a bounded capacity ladder stabilize the SVX-ZC likelihood at the requested three-significant-digit resolution? |
| Mechanism under test | Nested degree/rank capacity with capacity-appropriate quadrature, followed by a quadrature refinement check. |
| Expected failure mode | The likelihood remains sensitive to degree or rank, a numerically invalid cell prevents comparison, or center-only stability fails at validation points. |
| Promotion criterion | Smallest cell with stable immediate degree and rank refinements at calibration, followed by stable selected/higher-neighbor comparisons at all validation points and stable quadrature refinement. |
| Promotion veto | Non-finite result, failed mass/positivity/conditioning invariant, unauthorized configuration drift, or missing required total-likelihood comparison. |
| Continuation veto | Invalid harness, corrupt/incomplete artifact, inability to execute the declared cells, or exhausted campaign budget. A finite under-resolved candidate is not a continuation veto. |
| Repair trigger | Serialization, runner, manifest-diff, or local resource failure under the unchanged ladder and budget. |
| Explanatory diagnostics | Fit residual, holdout residual, per-time increment deltas, condition number, UKF initializer record, wall time, and any optionally available reference gap. |
| Not concluded | No exact-likelihood accuracy, score accuracy, HMC readiness, posterior correctness, GPU/XLA readiness, production readiness, or cross-model transfer. |

## Frozen target scope

- Row: `actual_sv` / `SVX-ZC`.
- Route: `zhao_cui_fixed_adjacent_state_squared_tt_v1`.
- Route classification: `extension_or_invention`, with `docs/main.tex` and its
  included high-dimensional filtering chapters as the mathematical authority.
- Dataset seed: `81101`.
- Horizon: `T=10`.
- Center parameter:
  `[0.2533471031357997, -0.916290731874155]` in
  `[gamma_unconstrained, log_beta]` coordinates.
- Coordinate half-width: `8.0`.
- Dtype: TensorFlow `float64`.
- Backend for this first mechanism test: deliberate CPU-only TensorFlow,
  `CUDA_VISIBLE_DEVICES=-1`, non-XLA diagnostic execution.
- Initialization: repository UKF initializer, including its identity and core
  hashes in every cell.
- Density tau: `0.0` in the value configuration; no density-floor change
  across cells.
- Ridge, sweeps, sweep order, condition thresholds, storage budgets, measure
  convention, transition timing, observations, and deterministic branch seed
  policy: unchanged from the current comparator configuration except for
  capacity-derived row/column budgets.
- Campaign invariant thresholds: marginal mass error `<=1e-10`, queried density
  values `>=-1e-14`, and maximum recorded solved-system condition number
  `<=1e10`. The fitter's internal `1e16` fail-closed limit remains unchanged;
  the stricter `1e10` campaign condition screen is recorded separately.
- Density finiteness/nonnegativity is queried on a fixed order-9 Legendre grid
  on every retained density axis. This is an invariant smoke, not an accuracy
  comparator.

This is intentionally a CPU/non-XLA diagnostic of whether the tuning protocol
works. It cannot select a GPU/XLA production default. A later claim-bearing
GPU/XLA route must repeat the tuning within that execution scope.

## Default and assumption audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|---|
| Three-significant-place value rule | Owner instruction in this task | Directly expresses the requested stopping precision | Stable capacities may still share a common approximation bias | Preserve nonclaim and any optional reference as explanatory only | Reviewed experiment rule |
| Degree set `[4,6,8,10,12]` | Extends the historical degree-8 cell in both directions | Bounded even-step ladder reaches one materially higher basis | Convergence begins only above degree 12 | Boundary classification `UNDER_RESOLVED_DEGREE` | Hypothesis, not default |
| Rank set `[2,4,6,8]` | Extends historical ranks `(1,2,4,6)` above rank 6 | Covers low through higher feasible TT coupling without the known weak rank-1 arm | Convergence begins only above rank 8 | Boundary classification `UNDER_RESOLVED_RANK` | Hypothesis, not default |
| Common order 25 | Exact-product order rule at maximum degree 12 | Separates degree/rank effects and avoids under-integrating the declared squared polynomial space | Non-polynomial target projection remains order-sensitive | Separate order 25/29/33 confirmation | Reviewed calibration control |
| Center-only full grid | Existing truth parameter from seed-81101 dataset | Bounds cost while allowing a clean calibration screen | Center nomination is locally unrepresentative | Four untouched validation points | Convenience choice with veto |
| Validation perturbation `0.05` | New local-neighborhood design choice | Material but bounded movement in each unconstrained coordinate | Does not cover posterior tails or correlated directions | Report scope and fail on any axis-point instability | Hypothesis, not posterior coverage |
| UKF initialization | Current repository policy and repaired route | Keeps initialization fixed while capacity varies | Scalar augmented-noise UKF remains geometrically degenerate | Record UKF moments and core hashes; residual is explanatory | Frozen route default, not truth |
| CPU, float64, non-XLA | Small bounded mechanism diagnostic | Avoids GPU campaign cost while testing deterministic numerical behavior | Does not establish default GPU/XLA scope | Manifest labels CPU diagnostic and blocks production promotion | Explicit exception |
| `T=10` | Existing SVX-ZC admission diagnostic scope | Preserves comparable bounded runtime | Capacity may differ at longer horizon | Explicit cross-horizon nonclaim | Historical scope, not transferable default |

No setting in this table is silently promoted to another model, horizon,
dataset, parameter region, dtype, backend, or HMC run.

## Frozen parameter points

### Calibration

Use only the center point above for the full capacity grid.

### Untouched validation

After the grid nominates a cell, evaluate that cell and its required higher
neighbors at these four axis perturbations:

```text
[center_gamma - 0.05, center_log_beta]
[center_gamma + 0.05, center_log_beta]
[center_gamma, center_log_beta - 0.05]
[center_gamma, center_log_beta + 0.05]
```

These points are frozen before execution and cannot nominate a different
capacity. They may only confirm the nomination or return `under_resolved`.
No post-hoc replacement validation point is allowed in this campaign.

## Three-significant-digit rule

For adjacent values `v_low` and `v_high`, define

```text
scale = max(abs(v_low), abs(v_high))
place = 10 ** (floor(log10(scale)) - 2)       # scale > 1e-12
place = 1e-3                                  # otherwise
prefix(value) = (sign, decimal_exponent, first_two_significand_digits)
stable = prefix(v_low) == prefix(v_high)
```

The first two significant digits are the stable prefix because the third digit
is explicitly allowed to change. The result must also record the raw delta,
normalized delta, both values rounded to three significant digits, and both
prefix tuples. A change from `-20.447` to `-20.181` therefore passes; a change
from `-20.447` to `-21.181` fails.

For each time index, apply the same rule to the two log-likelihood increments.
If the total is stable but any increment is unstable, attach
`cancellation_warning=true` and preserve the increment table as an explanatory
diagnostic. It does not change the total-likelihood stability decision in this
phase.

## Frozen capacity ladder

### Calibration grid

Use:

```text
degrees = [4, 6, 8, 10, 12]
ranks   = [2, 4, 6, 8]
order   = 25
```

Execute only algebraically feasible cells satisfying `rank <= degree + 1`:

```text
(4,2), (4,4),
(6,2), (6,4), (6,6),
(8,2), (8,4), (8,6), (8,8),
(10,2), (10,4), (10,6), (10,8),
(12,2), (12,4), (12,6), (12,8)
```

There are 17 calibration cells. Every cell uses the common order 25, which is
the existing `2*d+1` rule evaluated at the largest degree 12. This prevents a
degree comparison from silently changing quadrature at the same time and
avoids under-integrating the highest-degree squared polynomial representation.

### Nomination rule

For a grid cell `(d,r)` to be nominated, all of the following must exist and
pass:

- `(d,r)` itself passes numerical invariants;
- degree neighbor `(d+2,r)` passes and is `value_stable_3sf` relative to
  `(d,r)`;
- rank neighbor `(d,r_next)` passes and is `value_stable_3sf` relative to
  `(d,r)`, where `r_next` is the next value in `[2,4,6,8]`;

Choose the feasible nominee minimizing, in order:

1. `2 * (degree + 1) * rank`, the exact scalar count in two TT cores of shapes
   `[1, degree+1, rank]` and `[rank, degree+1, 1]`;
2. degree;
3. rank.

The coefficient count determines cost ordering only after every hard criterion passes. It
does not rank scientific accuracy. Boundary cells without both higher
neighbors cannot be nominated; they only establish that the tested grid ended.

### Quadrature confirmation

For the nominated `(d,r)`, run two additional center cells with orders

```text
29
33
```

The base-to-first and first-to-second total-likelihood comparisons must both be
`value_stable_3sf`. If not, return `under_resolved_order` and do not expand the
order ladder during this campaign. Increment differences remain reported.

### Validation comparisons

At each of the four validation points, execute:

- nominated `(d,r,25)`;
- degree neighbor `(d+2,r,25)`;
- rank neighbor `(d,r_next,25)`;
- nominated `(d,r,33)` for quadrature confirmation.

This is at most 16 validation evaluations. Duplicate cells at a point are
executed once. The final result is `SELF_CONVERGED_VALUE` only if all required
degree, rank, and quadrature comparisons pass at all four points.

## Implementation work

### Generic utility

Add `bayesfilter/highdim/capacity_tuning.py` with TensorFlow/Python-standard-
library implementations of:

- a frozen comparison-policy dataclass;
- scale-aware significant-place computation;
- total and per-increment adjacent-value comparison;
- structured statuses:
  `value_stable_3sf`, `value_unstable`, and `invalid_input`, plus a separate
  cancellation-warning field;
- candidate nomination from a rectangular/feasible capacity ledger;
- stable JSON-ready manifest payloads without NumPy.

Do not import NumPy and do not add score evaluation to this module.

### Tests

Add `tests/highdim/test_capacity_tuning.py` covering:

- positive and negative likelihoods;
- magnitudes above and below one;
- zero and near-zero values;
- exact tolerance boundaries;
- total stability with increment cancellation;
- missing/invalid/non-finite cells;
- deterministic minimal-capacity nomination and tie-breaking;
- rejection of boundary cells without higher neighbors;
- manifest-diff rejection when a non-capacity field changes.

### Runner

Add
`docs/benchmarks/run_svx_zc_capacity_self_convergence_tuning_20260801.py`.
It must:

- call `scalar_adjacent_state_fixed_tt_value` exactly once per requested cell;
- never call `scalar_adjacent_state_fixed_tt_score`, finite differences, or
  `exact_transformed_sv_scalar_dense_reference`;
- expose `--smoke` and full modes;
- refuse to overwrite an existing output root;
- write each cell immediately so interruption preserves completed work;
- resume only by reading verified completed cells into a fresh attempt root;
- record total likelihood and `ScalarAdjacentTTResult.log_increments`;
- record capacity identity, UKF identity/core hashes, route/config manifests,
  condition diagnostics, fit residuals, invariant status, wall time, and
  environment;
- compare structured manifests and permit only degree, rank, quadrature, and
  their derived budgets/projection-order, capacity-bound seed, and UKF core
  fields to differ; compare a separately constructed frozen-scope manifest for
  exact equality so those authorized differences cannot hide model/data/route/
  solver/policy drift;
- emit the selection and all adjacent comparisons in `result.json`.

## Verification sequence

1. Compile the new module and runner.
2. Run focused unit tests.
3. Run a two-cell `T=3` smoke using `(degree,rank)=(4,2),(4,4)` into a fresh
   smoke root. The smoke checks mechanics only and cannot select capacity.
4. Validate JSON schemas, hashes, finite values, and manifest-diff logic.
5. Run the 17-cell `T=10` calibration grid.
6. If a cell is nominated, run its two quadrature confirmations.
7. If quadrature confirms, run the four-point validation comparisons.
8. Write the result note and run `git diff --check` plus focused tests again.

The result note must contain separate ledgers for engineering execution,
numerical validity, and scientific interpretation. A successful process exit
can pass the engineering ledger while failing numerical validity or returning
an under-resolved scientific result.

## Exact commands

Focused verification:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
python -m py_compile \
  bayesfilter/highdim/capacity_tuning.py \
  docs/benchmarks/run_svx_zc_capacity_self_convergence_tuning_20260801.py

CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
pytest -q tests/highdim/test_capacity_tuning.py \
  tests/highdim/test_svx_zc_ukf_initializer.py
```

Smoke:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp \
python docs/benchmarks/run_svx_zc_capacity_self_convergence_tuning_20260801.py \
  --smoke \
  --output-root \
  docs/plans/artifacts/bayesfilter-svx-zc-capacity-tuning-20260801/smoke-attempt01
```

Full bounded run:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp \
python docs/benchmarks/run_svx_zc_capacity_self_convergence_tuning_20260801.py \
  --output-root \
  docs/plans/artifacts/bayesfilter-svx-zc-capacity-tuning-20260801/attempt01
```

## Attempt and compute budget

- Calibration: at most 17 value cells.
- Quadrature confirmation: at most 2 additional cells.
- Validation: at most 16 value cells.
- Total full-campaign cap: 35 `T=10` value evaluations.
- Smoke cap: 2 `T=3` value evaluations, outside the scientific cell count.
- Wall-time cap: 30 minutes for the full campaign.
- Attempt cap: one smoke attempt, one full attempt, and at most two localized
  repair/retry attempts within the unchanged 35-cell scientific budget.

Historical admission cells do not count as tuning evidence because their
runner also computed scores, finite differences, dense references, and used a
different promotion rule. They may inform the time estimate only.

## Failure classification and next action

| Outcome | Interpretation | Next action |
|---|---|---|
| `SELF_CONVERGED_VALUE` | The tuning mechanism found a scope-bound value-stable capacity. | Preserve selected capacity artifact; plan score/HMC suitability separately. |
| `UNDER_RESOLVED_DEGREE` | Higher degree still changes the likelihood materially. | Report boundary; propose a larger degree ladder only under a new budget. |
| `UNDER_RESOLVED_RANK` | Higher rank still changes the likelihood materially. | Report boundary; propose a larger rank ladder only under a new budget. |
| `UNDER_RESOLVED_DEGREE_AND_OR_RANK` | The bounded grid has no nominee and does not isolate one capacity axis as sufficient. | Report all adjacent comparisons; a new plan must decide which boundary to extend. |
| `UNDER_RESOLVED_ORDER` | Quadrature refinement is not stable. | Repair/order ladder plan; do not change degree/rank conclusion silently. |
| `VALIDATION_FAILED` | Center nomination does not transfer across the parameter neighborhood. | Report no selected capacity; redesign calibration coverage in a new plan. |
| `NUMERICALLY_INVALID` | Mathematical/numerical invariant failed. | Diagnose the smallest failed cell; do not call this evidence against capacity convergence. |
| `HARNESS_INVALID` | Runner, manifest, serialization, or artifact failed. | Repair and retry under the unchanged campaign budget. |

## Skeptical pre-execution audit

Verdict: `PASS_WITH_REVISIONS_APPLIED`.

The initial idea of independent one-axis ladders was revised to a feasible
degree/rank grid because degree and rank interact. The first grid draft tied
quadrature to degree, which would have confounded the degree comparison; every
grid cell now uses common order 25, followed by separate order 29 and 33 checks
at the nominated cell. The existing admission runner is not
reused as-is because it performs expensive score, finite-difference, and dense-
reference work that cannot answer this value-only tuning question. Branch
hashes are expected to change across capacities; the correct control is an
allowlisted structured manifest diff plus exact equality of a separately
constructed frozen-scope manifest, not full hash equality. Capacity-bound seed
strings and UKF cores are authorized to differ because their shapes and
identity bind each cell.

The main residual risk is that one center point may nominate a capacity that is
not stable elsewhere. The four untouched axis points test that risk before
selection. The perturbation size `0.05` is a frozen local-neighborhood design
choice, not evidence that it covers the eventual posterior. Failure at those
points blocks selection; passing them supports only this declared neighborhood.

The three-significant-place rule is intentionally coarse and owner-directed:
the first two significant digits must agree, while the third may change. It
measures capacity stability, not error relative to the unavailable exact
likelihood. Per-time increments expose possible cancellation but remain
explanatory because the declared tuning target is the total likelihood. No
unexamined residual, UKF, runtime, increment, or reference diagnostic is
allowed to become the promotion criterion.

## Pre-mortem

The run could pass while misleading us if every tested capacity shares the
same truncation bias, if the four axis points miss an unstable diagonal or tail
region, or if three-significant-place stability is too coarse for downstream
HMC. The result note must state all three limitations. Optional reference
values may expose common bias but cannot become a required comparator after the
run begins.

The run could fail for engineering rather than scientific reasons if a higher
cell exceeds a stale row/column budget, if the UKF initializer cannot represent
a requested rank, if artifact serialization loses a tensor field, or if an
allowlisted capacity-derived manifest change is mistakenly treated as route
drift. The smoke, static budget calculation, initializer-shape tests, immediate
per-cell serialization, and manifest-diff unit tests distinguish these cases
before interpreting the calibration grid.

The earliest stop is a harness or invariant failure. A finite but unstable
likelihood is valid negative capacity evidence and should continue through the
predeclared grid; it is not a reason to terminate the research direction.

## Artifacts

- Run root:
  `docs/plans/artifacts/bayesfilter-svx-zc-capacity-tuning-20260801/`
- Terminal result note:
  `docs/plans/bayesfilter-svx-zc-capacity-self-convergence-tuning-result-2026-08-01.md`
- Reset memo after a terminal result:
  `docs/plans/bayesfilter-svx-zc-capacity-self-convergence-tuning-reset-memo-2026-08-01.md`

Every serious run manifest must include commit, exact command, conda/Python/
TensorFlow environment, CPU-only declaration, `jit_compile=false`, model/data
identity, parameter points, seed policy, all capacity cells, wall time, output
paths, this plan, and the terminal result path.

The terminal result note must include a decision table and an inference-status
table. Since the capacity evaluation is deterministic under frozen inputs,
there is no stochastic candidate ranking; the note must say that explicitly
and must not convert deterministic capacity stability into statistical
superiority.

## Nonclaims

Even a terminal pass does not establish exact likelihood accuracy, score
accuracy, HMC validity or convergence, posterior agreement, statistical
superiority, GPU/XLA readiness, production readiness, transfer to another
model/data/horizon/neighborhood, or a universal default degree/rank/order.
