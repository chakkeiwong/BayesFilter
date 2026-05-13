# BayesFilter V1 Nonlinear Filtering Remaining Master Program Result

## Date

2026-05-12

## Scope

This result closes the third nonlinear filtering subplan in the BayesFilter V1
lane.  It leaves MacroFinance, DSGE, Chapter 18b, structural plans, and the
shared monograph reset memo untouched.

## Plan Audit

The execution plan was tightened before implementation.  The addendum blocks
Models B--C analytic score and HMC claims until explicit structural derivative
providers exist, keeps GPU/XLA as an optional escalated gate, and requires
benchmarks to distinguish exact references from dense one-step projection
references.

The audit conclusion is that the plan is executable if the pass is limited to
value filters, diagnostic vocabulary, CPU benchmarks, CI tiering, and
provenance.  It should not try to promote nonlinear Models B--C to score or HMC
readiness in this pass.

## Phase Results

### R0 Baseline

Baseline command:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_structural_svd_sigma_point_tf.py \
  tests/test_svd_cut_filter_tf.py \
  tests/test_svd_cut_branch_diagnostics_tf.py \
  tests/test_sigma_points_tf.py \
  tests/test_cut_rule_tf.py \
  -p no:cacheprovider
```

Result:
- `16 passed, 2 skipped, 2 warnings`.

Interpretation:
- the pre-existing nonlinear SVD/CUT value baseline was clean.

### R1-R2 Value Consolidation And Branch Diagnostics

Implemented:
- `bayesfilter/testing/nonlinear_diagnostics_tf.py`;
- testing exports in `bayesfilter/testing/__init__.py`;
- `tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py`.

Evidence:
- SVD cubature, SVD-UKF, and SVD-CUT4 share diagnostic snapshots for point
  count, polynomial degree, integration rank, support residual,
  deterministic residual, spectral gaps, active floors, PSD projection
  residuals, implemented covariance trace, branch label, and derivative target.
- Value branch summaries cover Models A--C for all three backends.
- Score branch summaries are intentionally affine-only.

Focused command:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py \
  -p no:cacheprovider
```

Result:
- `11 passed, 2 warnings`.

### R3 Approximation Effectiveness Benchmark

Implemented:
- `docs/benchmarks/benchmark_bayesfilter_v1_nonlinear_filters.py`;
- `docs/benchmarks/bayesfilter-v1-nonlinear-filter-benchmark-2026-05-12.json`;
- `docs/benchmarks/bayesfilter-v1-nonlinear-filter-benchmark-2026-05-12.md`.

Benchmark command:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 python \
  docs/benchmarks/benchmark_bayesfilter_v1_nonlinear_filters.py --repeats 1
```

Result summary:
- Model A matches exact Kalman to roundoff for all three value backends.
- Model B reports first-step dense projection errors:
  - cubature and UKF: about \(1.68\times 10^{-2}\) log-likelihood error;
  - CUT4: about \(3.64\times 10^{-3}\).
- Model C reports first-step dense projection errors:
  - cubature: about \(4.93\times 10^{-2}\);
  - UKF: about \(1.49\times 10^{-1}\);
  - CUT4: about \(3.39\times 10^{-2}\).
- Branch summaries were \(3/3\) finite value rows in each tiny parameter box.

Interpretation:
- CUT4 buys visible one-step moment-projection accuracy on Models B and C in
  this small CPU artifact, at the cost of 14 points versus 6 or 7.
- The result does not certify full nonlinear likelihood accuracy for Models
  B--C.

### R4 CI Runtime Tiering

Status:
- existing `pytest.ini` already defines `extended`, `hmc`, `external`, and
  `gpu` markers;
- the new branch diagnostic tests are fast CPU tests;
- the previous SVD-CUT branch-frequency tests remain opt-in extended tests;
- benchmark scripts remain executable artifacts outside default pytest.

### R5 GPU/XLA Gate

Status:
- deferred.

Reason:
- CPU value and score evidence is stable enough for local artifacts, but GPU
  work requires escalated CUDA visibility under the repository policy;
- this pass did not need GPU execution to close the CPU benchmark and
  diagnostics gaps.

### R6 HMC Readiness Gate

Status:
- blocked for nonlinear Models B--C.

Reason:
- HMC requires a stable score path on the target parameter box;
- Models B--C do not yet have explicit first-derivative providers for the
  analytic sigma-point score;
- the affine smooth score rung is useful evidence, but it is not a nonlinear
  HMC readiness claim.

### R7 Documentation And Provenance

Updated:
- Chapter 28 now explains current V1 evidence and limitations;
- `docs/source_map.yml` now registers this result and benchmark artifacts;
- this result file and the lane-specific reset memo record the execution
  status.

Validation:
- focused nonlinear suite:
  `54 passed, 2 skipped, 2 warnings`;
- `python -m py_compile` passed for touched Python modules/tests/scripts;
- `git diff --check` passed;
- `docs/source_map.yml` parsed with `yaml.safe_load`;
- nonlinear benchmark JSON parsed with `python -m json.tool`;
- scan for `import numpy`/`from numpy` in production nonlinear code and the
  new testing diagnostic helper found no matches;
- `latexmk -cd -pdf -interaction=nonstopmode -halt-on-error docs/main.tex`
  completed successfully.  The monograph still has pre-existing undefined
  citation/reference warnings outside this change.

## Remaining Hypotheses

H-NL1:
Models B--C can be equipped with explicit structural derivative providers
without changing the production structural callback contract.

H-NL2:
Once those providers exist, SVD cubature, SVD-UKF, and SVD-CUT4 analytic scores
on Models B--C will match finite differences of the implemented value filter
on smooth no-floor branches.

H-NL3:
CUT4's larger point set will continue to reduce one-step projection error on
strongly nonlinear rows enough to justify GPU/XLA point-axis vectorization
tests.

H-NL4:
Branch diagnostics over target-specific parameter boxes will identify whether
nonlinear HMC should proceed, remain blocked by weak spectral gaps/floors, or
use a different filter backend.

## Suggested Next Phases

1. Add derivative providers for Model B, then Model C.
2. Extend score finite-difference tests from the affine fixture to Models B-C.
3. Rerun branch summaries on score paths over the same parameter boxes used by
   the value benchmark.
4. Only after those pass, run escalated GPU/XLA parity and point-axis scaling
   benchmarks.
5. Select a single nonlinear HMC target and write an HMC readiness plan with
   explicit R-hat, ESS, divergence, and posterior-recovery thresholds.
