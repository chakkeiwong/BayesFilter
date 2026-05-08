# Plan: differentiable particle filter Phase 1 audit

## Date

2026-05-08

## Purpose

This plan is the first execution step for the BayesFilter differentiable
particle-filter program.  It follows
`docs/differentiable-particle-filter-program.md` and deliberately stops before
implementation.

The goal of Phase 1 is to audit theory sources, existing code, client
interfaces, external exploratory implementations, and benchmark requirements
well enough to make the first TensorFlow/TFP implementation pass safe.

## Scope

Phase 1 covers:

- the EDH -> EDH/PF -> soft-resampling -> OT -> neural-OT algorithm ladder;
- the generic BayesFilter state-space-model API boundary;
- the relationship to existing Kalman, sigma-point, and NumPy particle-filter
  backends;
- quarantined experimental tracks for student-code comparison and a clean
  internal experimental baseline;
- the DSGE-HMC and MacroFinance adapter expectations;
- benchmark and HMC evidence standards;
- source and derivation blockers.

Phase 1 does not implement the differentiable particle filter.

## Non-goals

- Do not add TensorFlow/TFP DPF code in this phase.
- Do not port student code.
- Do not claim HMC readiness.
- Do not change DSGE or MacroFinance economic semantics.
- Do not rewrite the existing NumPy structural particle filter unless a
  separate implementation phase proves it necessary.
- Do not add neural OT before lower rungs have a recorded audit path.

## Evidence Policy

Every audit subsection should record:

- files or sources inspected;
- claim status;
- assumptions;
- required implementation obligations;
- tests that must exist before promotion.

Use the labels from `docs/differentiable-particle-filter-program.md`, notably
`exact_special_case`, `unbiased_likelihood_estimator`,
`biased_differentiable_surrogate`, `value_only_reference`,
`comparison_only`, `filter_consistency_evidence`, `sampler_not_justified`,
`source_missing`, and `human_review_required`.

## Priority Order

If Phase 1 has to be split, preserve this order:

1. theory claim table;
2. BayesFilter API and public-contract table;
3. benchmark and tolerance table;
4. TensorFlow/TFP readiness table;
5. controlled experimental baseline scaffold;
6. student baseline comparison scaffold;
7. client workflow audit;
8. external implementation inventory;
9. Phase 2 handoff.

## Preflight: workspace and recovery state

### Actions

1. Record:

   ```bash
   git status --short --branch
   git log -5 --oneline --decorate
   ```

2. Confirm the governing note exists:

   ```bash
   test -f docs/differentiable-particle-filter-program.md
   ```

3. Confirm the current BayesFilter filter modules:

   ```bash
   rg -n "StructuralStateSpaceModel|FilterRunMetadata|ParticleFilterConfig|sigma|Kalman" \
     bayesfilter tests docs
   ```

### Exit gate

Continue only if the worktree state is understood and Phase 1's write set is
limited to audit documentation.

## Phase 1A: theory source audit

### Motivation

The DPF ladder mixes exact special cases, particle-flow approximations,
importance correction, differentiable resampling, and HMC gradients.  These
claims must be separated before code is written.

### Sources

Inspect:

- `/home/ubuntu/python/latex-papers/CIP_monograph/main.tex`;
- `/home/ubuntu/python/latex-papers/CIP_monograph/chapters/ch26_differentiable_pf.tex`;
- `/home/ubuntu/python/latex-papers/CIP_monograph/chapters/ch27_ledh_pfpf_neural_ot.tex`;
- `/home/ubuntu/python/latex-papers/CIP_monograph/chapters/ch32_diff_resampling_neural_ot.tex`;
- relevant nonlinear filtering and HMC chapters:
  `ch17_nonlinear_filtering.tex`, `ch20_hmc.tex`, and
  `ch21_advanced_hmc.tex`.

### Required output

Create an audit table:

| Claim | Source location | Status label | Implementation obligation | Blocker |
| --- | --- | --- | --- | --- |
| EDH linear-Gaussian recovery | | | | |
| EDH flow ODE and discretization | | | | |
| importance weights after flow | | | | |
| log-Jacobian calculation | | | | |
| soft-resampling bias | | | | |
| OT resampling objective | | | | |
| neural OT amortization | | | | |
| HMC likelihood-gradient contract | | | | |

### Exit gate

Every algorithm rung has a theory source and a claim-status label, or is
marked `source_missing` / `human_review_required`.

## Phase 1B: external implementation audit

### Motivation

The student repositories are useful comparison inputs but are not correctness
references and are not production candidates.  They must be treated as
`comparison_only` even when permission to use and test them has been granted.

### Sources

Audit these repositories if available locally or after a separately approved
fetch:

- `https://github.com/ljw9510/2026MLCOE`;
- `https://github.com/younghwan-cho-dev/advanced_particle_filter`.

Also inventory any paper-author code that the project intends to use.

### Required output

For each implementation source, record:

| Source | Available locally? | Algorithm pieces | Framework | Reusable? | Risks |
| --- | --- | --- | --- | --- | --- |

Classify code as:

- comparison only;
- conceptual reference only;
- test fixture candidate;
- implementation pattern candidate;
- rejected;
- blocked by license or missing assumptions.

### Exit gate

No external code is copied or wrapped until permission, provenance, algorithm
assumptions, framework mismatch, and tests are recorded.  No production module
may import student code.

## Phase 1B2: controlled experimental baseline scaffold

### Motivation

In addition to comparing student implementations, BayesFilter needs a clean
experimental project that we control.  This project can move faster than the
production path and can produce a more consistent experimental baseline for
debugging, metrics, and side-by-side comparison.

### Scope

Create or specify a quarantined project under:

```text
experiments/controlled_dpf_baseline/
```

This project may implement small EDH, LEDH, importance-weight, and resampling
experiments for reproducibility and comparison.  It must not be imported by
`bayesfilter/`.

### Required output

Record:

| Component | Experimental purpose | Production boundary | Required comparison |
| --- | --- | --- | --- |
| EDH linear-Gaussian | | | |
| EDH/PF importance weights | | | |
| soft resampling | | | |
| OT resampling | | | |
| LEDH prototype | | | |

### Exit gate

The scaffold must state that promotion to production requires a separate
BayesFilter-owned implementation or audit.  Experimental success earns only
`filter_consistency_evidence` or `comparison_only`, not HMC readiness.

## Phase 1C: BayesFilter API and code audit

### Motivation

The DPF must share the same model-facing interface as existing filters.

### Files to inspect

- `bayesfilter/structural.py`;
- `bayesfilter/filters/kalman.py`;
- `bayesfilter/filters/sigma_points.py`;
- `bayesfilter/filters/particles.py`;
- `bayesfilter/adapters/dsge.py`;
- `bayesfilter/adapters/macrofinance.py`;
- `bayesfilter/backends.py`;
- relevant tests under `tests/`.

### Required output

Produce:

| Module | Current role | Reuse decision | DPF implication | Required tests |
| --- | --- | --- | --- | --- |

The audit must specify the first DPF public contract, including:

- constructor/config object;
- callable likelihood interface;
- result object fields;
- metadata fields;
- diagnostic fields;
- random seed / common-random-number policy;
- handling of masks and all-missing observations;
- treatment of deterministic structural coordinates;
- eager and compiled TensorFlow execution modes.

### Exit gate

Implementation may not begin until this table identifies the minimum public
API and all shared test obligations.

## Phase 1D: client workflow audit

### Motivation

The DPF is intended for DSGE-HMC and MacroFinance use, but those projects
should remain clients, not hidden dependencies of the BayesFilter core.

### Sources

Inspect read-only:

- filter abstractions and UKF use under `~/python/dsge_hmc`;
- model/adaptation patterns under `~/python/MacroFinance`;
- existing BayesFilter DSGE and MacroFinance adapters.

### Required output

Record:

| Client | Existing filter API | Adapter boundary | DPF requirement | Do-not-change semantics |
| --- | --- | --- | --- | --- |

### Exit gate

The audit must name the adapter boundary that lets DSGE and MacroFinance use
the DPF without model-specific logic in the filter core.

## Phase 1E: benchmark and tolerance design

### Motivation

Filtering quality and HMC usefulness are separate claims.  The first test
ladder must prove value and gradient behavior before sampler convergence is
discussed.

### Required benchmark ladder

1. Linear-Gaussian SSM:
   - Kalman log-likelihood comparison;
   - filtered mean and covariance comparison;
   - finite-gradient check if the TensorFlow path exists.
2. Small nonlinear SSM:
   - UKF/EKF comparison;
   - state RMSE;
   - predictive likelihood;
   - finite diagnostics.
3. Flow stress cases:
   - small observation noise;
   - near-singular covariances;
   - particle-flow stiffness;
   - ESS collapse;
   - resampling bias diagnostics.
4. Synthetic HMC recovery:
   - only after value-gradient gates pass;
   - parameter recovery against true parameters;
   - R-hat, bulk ESS, tail ESS, divergences, acceptance rate, and runtime.

### Required output

Create a benchmark table:

| Benchmark | Reference filter | Metrics | Tolerance | Promotion label |
| --- | --- | --- | --- | --- |

The audit must specify concrete first-rung tolerances for:

- total Kalman log-likelihood agreement;
- filtered mean trajectory agreement;
- filtered covariance trajectory agreement;
- finite-difference gradient agreement;
- eager versus `tf.function` parity;
- controlled experimental baseline versus production candidate comparison.

### Exit gate

At least one linear-Gaussian and one nonlinear benchmark fixture must be fully
specified before implementation begins.

## Phase 1F: TensorFlow/TFP implementation readiness audit

### Motivation

The eventual production backend must be differentiable and HMC-compatible in
TensorFlow/TFP, not merely numerically plausible.

### Required checks

Design, but do not implement, the required TensorFlow/TFP policies:

- dtype policy;
- shape policy for particles, time, state, observations, and parameters;
- `tf.function` requirements;
- use of `tfp.distributions` for log densities where appropriate;
- linear algebra backend policy;
- random stream / common-random-number policy;
- finite-gradient checks;
- HMC target wrapper contract.

Eager-only success is a debugging result.  It is not sufficient for production
HMC promotion.  Any sampler-facing claim requires a compiled value-gradient
path with static-shape assumptions recorded.

### Required output

Record:

| Concern | Phase 2 decision | Required test |
| --- | --- | --- |
| dtype | | |
| static shapes | | |
| random numbers | | |
| linear algebra | | |
| autodiff through flow | | |
| autodiff through resampling | | |
| HMC target wrapper | | |

### Exit gate

The audit must define what "compiled differentiable likelihood path" means for
the first implementation rung.

## Phase 1G: Phase 2 implementation handoff

### Required output

Write a concise handoff section with:

- first implementation rung;
- files expected to be created or touched;
- tests to write first;
- benchmark data fixture policy;
- explicit blockers deferred to later rungs;
- acceptance gate for ending Phase 2.

Recommended first implementation rung:

```text
EDH linear-Gaussian recovery and EDH/PF importance-weight skeleton,
implemented in TensorFlow/TFP against the generic structural SSM contract,
with Kalman value comparison and no differentiable-resampling claim yet.
```

### Exit gate

Phase 2 begins only after the audit can answer:

- What exact scalar does the first DPF return?
- Which model interface does it consume?
- Which gradients are expected to exist?
- Which benchmark proves the first rung is not broken?
- Which claims are explicitly not made?

## Mechanical checks

After writing the Phase 1 audit result, run:

```bash
git diff --check
rg -n "guarantee|production-ready|converged|certified|unbiased|exact likelihood" \
  docs/differentiable-particle-filter-program.md docs/plans
```

Interpret any hits.  Terms such as `unbiased` and `exact` are allowed only
when tied to a stated status label and source.

## Final deliverables

Phase 1 should produce:

1. a source/API/code audit note under `docs/plans`;
2. a benchmark and tolerance table;
3. a TensorFlow/TFP readiness table;
4. a Phase 2 implementation handoff;
5. a short audit or result note approving, revising, or blocking Phase 2.
