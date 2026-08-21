# Fable Audit Request: Tuning-Test and Posterior-Coverage Adequacy

To: Fable
From: Codex
Date: 2026-08-17
Review type: read-only, skeptical plan audit

## Exact Plan

Review this updated plan first and use it as the primary authority:

`docs/plans/bayesfilter-tuning-streamline-refactor-plan-2026-08-16.md`

The plan now proposes two active interfaces, an explicit mass-matrix contract,
an analytic posterior-oracle gate, ordinary and fixed-transport holdouts, and
MacroFinance/dsge_hmc migration gates. Audit whether those additions are
sufficient and operationally complete. Do not assume that a named test file or
planned command is evidence until it exists and actually runs.

## Boundaries

Read-only review. Do not edit, commit, launch agents, install packages, mutate
environments, or run GPU/CUDA work. CPU-hidden checks are allowed only when
needed to verify a stated command. Use `conda run -n tfgpu` for TensorFlow/TFP
checks and set `CUDA_VISIBLE_DEVICES=-1` before imports. For dsge_hmc use
`BAYESFILTER_ROOT=/home/ubuntu/python/BayesFilter` and keep
`tests/archive` explicitly ignored while its known module-level segfault is
unresolved. Do not call omitted archive tests passing.

Do not infer posterior correctness, convergence, sampler superiority,
production readiness, or scientific validity from collection-only results,
short chains, acceptance rates, ESS/R-hat alone, fake runners, or Gaussian
smoke tests. Separate current evidence, planned evidence, and unimplemented
gates.

## Audit Questions

### 1. Test architecture

Audit the plan's three evidence layers separately:

1. Mechanics/contracts: shapes, finiteness, signatures, artifact identity,
   replay, stale-scope rejection, route classification, timeout, and repair
   state machines.
2. Numerical target validity: value/score correctness, mass construction,
   transformed-target identities, real-TFP execution, and posterior-oracle
   agreement on untouched holdouts.
3. Consumer integration: every active tuning caller in MacroFinance and
   dsge_hmc, public/private import boundaries, configuration handoffs, artifact
   consumption, and migration away from historical routes.

For each layer state what the plan requires, what current tests actually cover,
which checks are fake or deterministic plumbing, what is missing, whether the
gap blocks Phase 3/4 or only later promotion, and the exact test/artifact that
would close it. Treat `tests/test_hmc_tuning_posterior_oracle.py` as an open
deliverable until it exists and passes.

### 2. Posterior-fixture sufficiency

Build a fixture matrix. Audit whether the plan has explicit coverage for each
distinct problem below, and classify every fixture as `mechanics`,
`target-oracle`, `stress diagnostic`, `consumer integration`, or `historical`.

| Problem | Minimum question |
|---|---|
| Isotropic baseline | Does an exact `N(0,I)` value/score and identity metric work? |
| Shifted correlated Gaussian | Does nonzero mean/non-diagonal covariance give the exact score and moments? |
| Rotated/anisotropic Gaussian | Does dense covariance orientation matter for mass and transform? |
| Ill-conditioned Gaussian | Do eigenvalue floors/condition caps fail closed without silently changing the target? |
| Affine transformed Gaussian | Are Jacobian, score pullback, and base/transport moments correct? |
| Wrong-target controls | Do wrong mean, covariance, score, Jacobian, dimension, or coordinate signature fail? |
| Nonlinear curved target | Is there a banana/curved target where mass, epsilon, and `L` interact beyond Gaussian geometry? |
| Hierarchical/funnel target | Is position-dependent scale stress covered, or is the omission an explicit nonclaim? |
| Domain-specific synthetic target | Do MacroFinance and dsge_hmc test matched-DGP or analytic/synthetic posteriors for their actual adapters? |

For each row record target definition, reference type, controls exercised,
calibration/holdout split, seeds/chains/draws, MCSE or uncertainty method,
hard vetoes, explanatory diagnostics, supportable claim, nonclaim, current test,
planned test, and gap. Decide explicitly whether the Gaussian plus domain
fixtures are sufficient for this refactor or whether a nonlinear and/or
hierarchical stress fixture is required before default-readiness language.

### 3. Joint mass, step size, and leapfrog `L`

Audit the full intended sequence:

`geometry/initial mass -> bootstrap -> windowed/adapted mass -> epsilon tuning -> L/trajectory selection -> untouched verification`

Return a joint-control table with columns:

`control`, `candidate policy`, `stage`, `retuned after mass change?`,
`calibration data`, `holdout data`, `primary criterion`, `vetoes`, `repair
trigger`, `artifact identity`, `current test`, `planned test`, `gap`.

Verify all of the following:

1. The plan says whether the search is staged, conditional, or genuinely joint;
   it must not call sequential tuning joint without defining conditioning and
   retuning.
2. Every mass update invalidates epsilon/trajectory context and triggers a
   fresh seed and typed handoff. Epsilon is retuned for the selected mass, and
   `L` candidates use a declared fresh-epsilon policy.
3. Comparisons match target, adapter, coordinates, starts, seeds,
   calibration/holdout partitions, dtype/backend, XLA policy, and budget.
4. The plan distinguishes position covariance/preconditioner, affine factor,
   latent metric, and TFP momentum covariance/kinetic precision. Check factor
   orientation, including `F @ F.T` versus `F.T @ F`.
5. Tests cover identity, exact dense covariance, diagonal/scaled covariance,
   warmup-adapted covariance, and a deliberately bad mass under one target.
6. `L` is tested as trajectory control, not merely serialized. Include a
   negative control where acceptance alone would select the wrong `L`.
7. Selection gives target-preserving holdout validity priority. Acceptance,
   ESS, R-hat, runtime, energy tails, condition number, and mass closeness have
   explicit roles and cannot silently become superiority criteria.
8. Every mass/epsilon/`L` change has seed lineage, typed handoff, artifact
   identity, and replay coverage. Cross-scope setting transfer is not defaults.

Flag any place where mass construction is covered but mass selection is not,
or epsilon/acceptance is covered without the interaction with `L`.

### 4. MacroFinance callers and tests

Cross-check the plan against all tuning references, not only its six focused
files. At minimum inspect:

- MIDAS ordinary, robust broad-grid, bootstrap/geometry, and fixed-mass paths;
- CCMA fixed-metric, operational-grid, trajectory, and epsilon-repair paths;
- cross-country multi-asset HMC client and mass-preconditioner paths;
- mixed-frequency Phase-5T/budget-ladder callers;
- one-country/ZLB tuning and replay paths; and
- two-currency fixed-transport/NeuTra tuning paths.

At minimum cross-check the corresponding `test_daily_asset_midas_*tuning*`,
`test_cross_country_multi_asset_*hmc*tuning*`,
`test_mixed_frequency_tfp_c2_full_bayesfilter_hmc_tuning*`,
`test_run_ccma_*fixed_metric*`, one-country analytic/map-mass/replay, and
two-currency fixed-metric/tuning test families. Add any missed caller.

For every active caller verify public-interface classification, target scope,
coordinate/transport identity, mass policy, epsilon policy, `L` policy, seeds,
artifact handoff, behavior-level assertions, private-import migration tests,
and failure classification. A source-text assertion alone is not coverage.

Audit the separate MacroFinance full-suite baseline: 4,252 collected and 38
collection errors. Identify missing-dependency versus checkout-order causes;
do not call this a pass. Require root-cause resolution or explicit owner waiver
before full-suite promotion.

### 5. dsge_hmc callers and tests

Cross-check all tuning references, at minimum:

- BGS Stage-C grid and aggregate policy;
- Rotemberg fixed-transport/NeuTra and explicit-state fixed-mass paths;
- public path-resolution and private mass/bootstrap import contracts;
- BGS mass/geometry and synthetic recovery contracts; and
- Rotemberg fixed-grid, XLA, affine-posterior, and HMC component contracts.

For every active caller verify the same fields required for MacroFinance. Check
that the configured full-suite command uses `BAYESFILTER_ROOT`, `tfgpu`, CPU
hiding for this review, and `--ignore=tests/archive`; report all other errors
and order dependence. Confirm focused contracts cannot conceal an uncovered
production caller.

### 6. Command and evidence audit

Check the plan's command paths, environment, and question-answering power. If
running checks, use the plan's `tfgpu` CPU-hidden commands for BayesFilter,
MacroFinance, and dsge_hmc. Record exact collected/run counts, failures,
skipped paths, environment versions, and evidence class. Do not run GPU canaries
as part of this review.

## Required Deliverable

Return a Markdown audit with these sections, findings first:

### A. Findings

Order by severity and cite exact file/line anchors. Classify each as:
`BLOCKING`, `MATERIAL`, `NONBLOCKING`, or `CONFIRMED`.

### B. Posterior-fixture matrix

One row per distinct fixture/problem, with current/planned tests, oracle,
controls, evidence role, and gap.

### C. Joint mass/epsilon/`L` matrix

Use the requested columns and state whether the plan tests interactions or only
individual controls.

### D. MacroFinance coverage map

Every active caller family, interface, associated tests, command, and status;
include missing tests and full-suite blockers.

### E. dsge_hmc coverage map

Every active caller family, interface, associated tests, command, and status;
include `BAYESFILTER_ROOT` and archive omission.

### F. Required plan repairs

For each `BLOCKING`/`MATERIAL` finding give exact wording or a specific
test/artifact deliverable and state whether it blocks implementation,
cross-repo migration, target-validity testing, or only default promotion.

### G. Evidence and nonclaims

Separate hard veto, descriptive, statistical, collection, and unimplemented
evidence. State what this audit does not establish.

Use `REVISE` if the plan lacks a required fixture, misses an active consumer,
does not jointly retune mass/epsilon/`L`, or treats collection/mechanics as
target-validity evidence. Open implementation work is not itself a plan defect
when the plan makes it a fail-closed gate.

End with exactly two lines:

`AUDIT_VERDICT: AGREE` or `AUDIT_VERDICT: REVISE`

`PLAN_VERDICT: AGREE` or `PLAN_VERDICT: REVISE`
