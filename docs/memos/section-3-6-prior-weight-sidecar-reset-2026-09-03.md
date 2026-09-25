# Section 3.6 Prior-Weight Sidecar Reset Memo

Date: 2026-09-03  
Status: `EXPERIMENT COMPLETE; NOT INTEGRATED; NO PROMOTION`  
Owner repository: `/home/chakwong/BayesFilter`

## Why this memo exists

A session interruption and subsequent worktree confusion made it unclear which
checkout contained the Section 3.6 work. This memo is the authoritative context
reset for that work. It records the mathematical motivation, the exact
implementation boundary, the executed evidence, the current Git state, and the
conditions for any future port or merge.

No Section 3.6 implementation was merged into `main`, rebased onto current
`main`, or pushed. The memo itself is the only new file being added to the
current checkout by this reset operation.

## Research question

The project is investigating whether a smooth, kernelized observation mixture
can reduce finite-particle score error when used alongside the GenUT dual-cap
trust-region filter. The intended use is a deterministic proposal force for an
HMC leapfrog integrator, not an automatic replacement of the canonical GenUT
likelihood or analytical score.

The proposal was narrowed after reviewing the alternatives:

- PaRIS and quadratic Poyiadjis-style interaction estimators are outside the
  runtime budget because they are quadratic in the particle count.
- Nemeth, Fearnhead, and Mihaylova provide a linear-cost score construction,
  but it is a different recursively regularized estimating equation. It is not
  automatically the gradient of the finite GenUT value program, so substituting
  it would change the HMC target/force relationship.
- Younis and Sudderth's regularized particle and mixture-density filters
  motivate replacing Dirac components by kernels. Their smoothing/training
  constructions have different objectives, and their reported mixture
  operations do not establish correctness for the GenUT transport and reset.
- A fixed random stream makes the finite program a deterministic function of
  parameters and observations, which is useful for HMC differentiation and
  paired comparisons. It does not remove finite-particle bias or create an
  exact likelihood.

The selected Section 3.6 construction is therefore a separate, linear-cost
Gaussian-kernel observation sidecar with an explicit derivative boundary. The
canonical GenUT dual-cap/Contract-E value, reset, and analytical LEDH score
routes remain the claim-bearing routes.

## Mathematical correction

At observation time `t`, let `x_{t,i}^-` and `w_{t,i}^-` denote the transitioned
particle and normalized weight before assimilating the current observation,
and let

```text
g_{t,i} = g_theta(y_t | x_{t,i}^-)
Z_t     = sum_i w_{t,i}^- g_{t,i}.
```

The finite observation increment is `ell_t = log Z_t`. The posterior weights
are then

```text
w_{t,i}^+ = w_{t,i}^- g_{t,i} / Z_t.
```

The GenUT transport/reset is applied only after this update. The distinction
between `w^-` and `w^+` is part of the estimand, not a naming detail.

### Historical mistake

The original hook was named `capture_pre_reset_observation`, but it captured
the normalized post-observation weights `w^+` before the reset. Reusing those
weights with the current observation factor in a sidecar computes

```text
tilde Z_t = sum_i w_{t,i}^+ bar g_{t,i}
          = (sum_i w_{t,i}^- g_{t,i} bar g_{t,i}) / Z_t,
```

which is generally not the intended sidecar increment
`bar Z_t = sum_i w_{t,i}^- bar g_{t,i}`. At zero kernel bandwidth,
`bar g = g`, so

```text
tilde Z_t - Z_t = Var_{w_t^-}(g_t) / Z_t >= 0.
```

For two equally weighted particles with observation factors `(1, 3)`, the
intended increment is `Z_t = 2`, while post-update reuse gives `tilde Z_t =
2.5`. The old route is therefore a different, double-weighted estimand when
advertised as the current likelihood increment.

### Corrected sidecar

The corrected route captures, immediately after transition and before the
current observation factor,

```text
(x_{t,i}^-, w_{t,i}^-, d x_{t,i}^-, d w_{t,i}^-).
```

For a fixed Gaussian state kernel with covariance `B`, observation matrix `C`,
and observation covariance `R`, each component is

```text
bar g_{t,i} = Normal(y_t; C x_{t,i}^-, R + C B C^T),
bar ell_t  = log sum_i w_{t,i}^- bar g_{t,i}.
```

The implemented score arm is explicitly conditional: it holds the captured
cloud, prior weights, and kernel covariance fixed and differentiates the
declared observation-scale covariance dependence. A total derivative would
also need every declared dependence through particles, weights, kernel, and
reset. Consequently the sidecar score is not the canonical GenUT total score
and is not a claim-bearing target gradient.

## LaTeX specification

The standalone derivation and implementation note is:

`docs/papers/section_3_6_prior_weight_sidecar/section_3_6_prior_weight_sidecar.tex`

Rendered PDF:

`docs/papers/section_3_6_prior_weight_sidecar/section_3_6_prior_weight_sidecar.pdf`

The note covers the state-space definition, observation timing, Gaussian
convolution, mixture differential, prior/post weight mismatch, smoothed
moments, fixed-stream interpretation, and reversible volume-preserving KDK
proposal mechanics. It states the source boundary and nonclaims directly.

The final PDF was 17 pages. The final source and PDF checksums were:

```text
TeX: c16ca10e41a4acf917f51e39c63301978dffd2a605b68842ebcd7ef27022394e
PDF: 2761c0ac28835e4e893cfeab2b0ad8ed43945db40f37d4917ce8dad1043bcbb6
BIB: fb4d28a9f202902f944ccbbd3beeca121be6196ce8aea9485479672027701e4e
```

The PDF build completed without LaTeX errors or undefined citations/references.
Representative rendered pages were inspected for clipping and overlap; none
was found. The log retained only ordinary layout warnings in dense material.

## Experimental implementation

The implementation was developed in the isolated Git worktree:

```text
/home/chakwong/BayesFilter/.claude/worktrees/section-3-6-prior-weight-sidecar
branch: experiment/section-3-6-prior-weight-sidecar-20260903
HEAD:   97b9c774c22c9f51b98d30179dcab189f46038b1
```

That branch was based on the then-current `main` commit `2f445aaf`, before the
later RQMC commits now present on `main`. Its relevant changes are currently
uncommitted in that worktree.

### Producer and contract

`bayesfilter/highdim/cubature_genut_batch_tf.py` adds a diagnostic-only
`capture_prior_observation` hook and reserved tensors for:

- transitioned prior particles;
- normalized prior weights;
- prior particle tangents;
- prior weight tangents; and
- a prior-stage validity mask.

The ordinary producer return tuple and finite value/score path are preserved
when capture is disabled. The capture path requires a static horizon and checks
finite particles/tangents, strictly positive unit-mass weights, and zero-mass
weight tangents.

`bayesfilter/highdim/genut_observation_cloud_contract.py` issues the separate
repository-owned contract:

```text
schema:           bayesfilter.highdim.genut_prior_observation_contract.v1
route_id:         batch_genut_prior_observation_cloud_diagnostic_v1
stage:            prior_observation
weight semantics: normalized_prior_weights_before_observation_v1
authority:        genut_prior_observation_diagnostic_only
```

The contract metadata is issued by the builder, not caller-stamped. The old
post-observation route remains readable under its historical schema and cannot
be relabeled as the prior route.

### Master program

The only new entry point for the experiment is:

`docs/benchmarks/run_section_3_6_prior_weight_sidecar_master.py`

Its call chain is:

```text
master driver
  -> build_batch_genut_prior_observation_snapshot
  -> batch_finite_value_score_manual_jvp_diagnostic(
       capture_prior_observation=True)
  -> prior capture tensors
```

The source guard rejects the historical builder, the historical capture flag,
NumPy, `tf.map_fn`, `tf.vectorized_map`, and pfor in this driver. The sidecar
uses per-particle Gaussian solves and reductions, with no `N x N` particle
matrix. The independent diagonal Kalman recursion is used only as an R1 oracle
and does not call the GenUT producer.

The driver has three explicit phases:

1. `calibration`: select a bandwidth on disjoint calibration paths and issue a
   repository-owned scope/tuning artifact.
2. `validation`: use only the matching calibration artifact on untouched paths,
   report paired path-level squared-bias intervals and stream MCSE.
3. `mechanics`: test the frozen position-only force with an exact endpoint
   potential using reversibility, replay, Jacobian, and energy-accounting
   diagnostics.

No phase registers the sidecar as a canonical score, likelihood, HMC target,
or default.

## Review and verification

The plan and skeptical review are preserved in:

- `docs/plans/bayesfilter-section-3-6-prior-weight-sidecar-master-program-2026-09-03.md`
- `docs/plans/bayesfilter-section-3-6-prior-weight-sidecar-master-program-review-2026-09-03.md`
- `docs/plans/bayesfilter-section-3-6-prior-weight-sidecar-master-program-result-2026-09-03.md`

The skeptical review identified and repaired the historical post-update
baseline error, an initially overstated row budget, and language that could
have promoted a conditional sidecar score by naming alone. Calibration and
validation partitions, scope hashes, output-directory protection, MCSE gates,
and nonclaims were made explicit before execution.

### Automated checks

Executed on the isolated branch with explicit CPU hiding:

```text
CUDA_VISIBLE_DEVICES=-1
conda environment: tftwogpu
Python 3.11.15
TensorFlow 2.20.0-dev0+selfbuilt
TensorFlow Probability 0.25.0
XLA/JIT: false
master seed: 20260903
```

Results:

- corrected-route focused suite: `54 passed, 2 warnings`;
- neighboring batch GenUT parity suite: `12 passed, 2 warnings`;
- touched-route coverage run: 28 tests passed, with line coverage of 40% for
  the large shared producer, 73% for the contract module, and 87% for the
  master driver;
- `py_compile`: passed;
- `git diff --check`: passed; and
- all pilot checksum files: passed `sha256sum -c`.

The producer coverage percentage includes many unrelated legacy branches and
is not a correctness certificate.

### MathDevMCP

The final-source rigor report is:

`docs/benchmarks/artifacts/section_3_6_prior_weight_sidecar_20260903/mathdevmcp/rigor.md`

Source digest:

`ac0f02165fa283295dae0ca734ae029fcbfc3a92bf5df6c4e5fb71eef6db7149`

The bounded audit covered all 26 selected equation labels. It retained four
diagnostic gaps and three concrete exposition repairs, chiefly inverse-domain
and matrix-shape assumptions. Direct Lean readiness was inconclusive in the
available environment; no typed Lean certificate was produced. MathDevMCP's
label-specific implementation probes returned `unverified` because their
operation classifier did not recognize the repository's named capture
operation. The evidence used for the implementation claim is therefore direct
source inspection plus executable tests, not a formal Lean proof.

## Executed pilot

The pilot was deliberately diagnostic and CPU-only. It used two clean-room
diagonal LGSSM scopes, each at `T=1`:

- `D=1`, `N=4`; and
- `D=3`, `N=6`.

Calibration used four paths, eight independent streams per scope, and six
bandwidths `{0, .05, .10, .20, .40, .80}`. The driver emitted exactly
`2 * 4 * 8 * 7 = 448` rows, including the canonical arm. Validation used eight
paths and eight streams, with canonical, zero-bandwidth, and selected-bandwidth
arms, for exactly `2 * 8 * 8 * 3 = 384` rows. The row budget was bounded at
1,500.

The versioned output roots are:

```text
docs/benchmarks/artifacts/section_3_6_prior_weight_sidecar_20260903/
  pilot01-mechanics/
  pilot01-calibration/
  pilot01-validation/
```

Each contains a manifest, result rows, summary, and checksums. Manifests record
the branch, base commit, command, environment, seed, CPU-only policy, scope
identity, route identity, and data hashes.

The compatible local custom-op binary was
`bayesfilter/ops/_symmetric_sylvester_ops.so`, SHA-256
`661f11b9db1f6e9ab9ce4aae8ae86591779cdbdfe6fda777a7c87956ec2686fa`. A source
rebuild was unavailable because the local CUDA headers were missing; the
hash-identical existing build was copied into the isolated worktree. This was
an infrastructure repair, not scientific evidence.

### Mechanics result

Both exact-Gaussian and finite-canonical endpoint arms passed. The largest
recorded reverse position/momentum residual was `1.11e-16`; the absolute
volume-Jacobian residual was `2.22e-16`; all transitions were finite and there
were no divergences. This establishes only the tested proposal mechanics, not
score correctness, posterior correctness, or HMC mixing.

### Calibration result

Both scopes selected the grid boundary `rho=0.80`. The calibration MCSE gate
failed in both scopes. The boundary selection means this small grid did not
identify an interior bandwidth optimum.

### Holdout result

The primary interval is the paired bootstrap 95% interval for relative
path-level squared-bias change of the selected sidecar versus canonical GenUT,
using the independent Kalman score as the R1 reference.

| Scope | Selected rho | Relative mean | Bootstrap 95% interval | Canonical MCSE ratio | Sidecar MCSE ratio | Status |
|---|---:|---:|---:|---:|---:|---|
| `D=1, N=4, T=1` | 0.80 | 6.6082 | `[-0.2825, 19.4854]` | 0.4709 | 0.4615 | `inconclusive_or_vetoed` |
| `D=3, N=6, T=1` | 0.80 | 0.2209 | `[-0.9342, 1.5096]` | 0.1756 | 0.1868 | `inconclusive_or_vetoed` |

The interval upper endpoints are neither below zero nor below the practical
threshold `-0.10`, and neither scope meets the MCSE threshold `0.10`. Thus the
pilot does not support a ranking or bias-reduction claim. All 448 calibration
and 384 validation rows were finite and valid, and selected-arm Cholesky
margins were positive. The failure is lack of statistical evidence, not a
nonfinite or non-SPD implementation failure.

## Current Git state

### Current `main`

The current repository checkout is:

```text
branch: main
HEAD:   9d646c36 Fix RQMC master program inconsistencies for executable state
status: ahead of origin/main; ongoing RQMC/C2 files are untracked or modified
```

The current `main` checkout does not contain the Section 3.6 master program or
the Section 3.6 PDF. Its untracked/modified RQMC and C2 files are unrelated to
this memo and must be preserved.

### Isolated Section 3.6 branch

The isolated branch still contains the Section 3.6 implementation and artifacts
listed above. Its implementation edits are uncommitted. It is behind current
`main` and must not be treated as a ready-to-merge patch without a fresh
compatibility review against `9d646c36`.

No merge, rebase, cherry-pick, or push has been performed. No long-running
process is active.

## Decision and nonclaims

The semantic correction is implemented and locally tested. The bounded pilot
is complete with no promotion. The following statements are explicitly not
supported:

- the sidecar is not proven unbiased at finite `N`;
- the sidecar is not proven equal to the exact nonlinear score;
- fixed random numbers are not claimed to remove bias;
- the conditional sidecar score is not the canonical total derivative;
- the Nemeth estimator is not silently substituted for the GenUT score;
- no posterior convergence, HMC mixing, or production-readiness claim is made;
- no result from this pilot transfers to nonlinear DSGE or general smoothing;
  and
- no pre-2026-08-21 LEDH result is reused as evidence.

## Next authorized step

Before any integration, create a fresh port/reconciliation plan against current
`main` `9d646c36`. That plan must:

1. inspect the current producer, contract, and package-export call chain for
   conflicts with the newer RQMC/C2 changes;
2. decide whether the diagnostic branch should be ported, archived, or
   discarded, without changing the canonical LEDH score route;
3. preserve the existing pilot artifacts as historical diagnostic evidence;
4. rerun the focused and neighboring parity suites after any port; and
5. require a new versioned experiment root for any larger calibration/holdout
   campaign. A larger run would need more replication and an expanded
   bandwidth grid before it could address the current MCSE and boundary-grid
   weaknesses.

Until that reconciliation is explicitly completed, the Section 3.6 master
program is an isolated diagnostic experiment, not part of the current `main`
execution path.
