# C2 Phase 2 Generic-DMIS and Recursive-Map Repair Campaign

Date: 2026-09-02  
Status: superseded for execution by the contract-correction re-run; original
contract preserved  
Scope: C2 n=4 diagnostic holdout only; no production or default change

## 1. Why this plan exists

The 2026-08-31 coherent campaign tested the exact-factor proposal ladder and
then ran the original frozen-target Stage 2 integration. That Stage 2 run
stopped with the declared `christoffel_qmc_uncertainty` continuation veto. The
2026-09-01 generic-DMIS plan validated a model-independent finite-bank kernel
and a one-step C2 compatibility smoke, but did not wire that kernel into the
frozen Stage 2 snapshots and did not execute the recursive moment-map stage.

This plan closes those two explicit gaps. It is a diagnostic campaign, not a
claim that the C2 carried approximation is the true model likelihood.

The first driver did not apply the plain-DMIS precision criterion exactly as
written here; the corrective plan and contract-faithful result are
`bayesfilter-c2-phase2-generic-dmis-contract-correction-plan-20260902.md` and
`bayesfilter-c2-phase2-generic-dmis-contract-correction-execution-result-20260902.md`.

## 2. Research question and hypotheses

### Main question

Can a complete deterministic-mixture importance correction, optionally using
the retained squared-TT as a known-integral control variate, repair the Phase 2
finite carried-target normalizer while preserving the analytical derivative of
the same frozen finite program? After that correction is established, does a
lagged moment-derived coordinate map reduce recursive fitting error without
changing the exact target?

### Hypotheses

* H1 (measure/integration): the prior Stage 2 ambiguity is primarily integration
  variance or a mismatched row law. A complete mixture with a defensive
  Student component will give stable, cross-scramble estimates of the same
  frozen target.
* H2 (representation): after H1 is resolved, the retained TT Gram normalizer
  remains separated from the independently integrated target. The separation
  is a fitting/tail error, not an importance-weight identity error.
* H3 (recursion): deriving the next map from lagged retained-density moments
  changes the map but not the target. It will be accepted only if held-out
  and recursive errors improve without a worse direct normalizer or a failed
  covariance/map validity check.

Failure of H1, H2, or H3 rejects only that candidate mechanism at this scope;
it does not reject the generic importance identity or the broader research
direction.

## 3. Evidence contract

### Target and comparators

For each captured pre-update snapshot at `t=2,3,4`, let

\[
  \gamma_t(u)=\eta_{2n}(u) E_t(u),
\]

where `E_t` is the exact branch-summed transition-observation target returned
by the existing snapshot evaluator and `eta` is the standard-normal reference
density. The target is the finite carried-density integral
`Z_t^fin = integral gamma_t(u) du`. It is not the exact C2 marginal
likelihood.

The baseline ladder is:

1. the stored retained-TT Gram value `Z_H`;
2. the original Christoffel, standard-normal, and Student-mixture integration
   arms;
3. plain complete-DMIS using fixed standard-normal and product-Student banks;
4. complete-DMIS plus the squared-TT control variate with its exact Gram
   integral; and
5. the recursive lagged-map candidate, evaluated with the selected correction.

The PF estimate is a compatibility comparator only. It carries an empirical
particle approximation and therefore is not substituted for `Z_t^fin`.

### Promotion criterion

The frozen DMIS/CV candidate is viable only when all captured steps satisfy:

* complete-mixture denominator uses every component;
* all row masses and mixture weights close to one;
* target, proposal, residual, and estimate are finite;
* target support and (for the tangent) mixture support are valid;
* independent complete-DMIS replications have a 95% log-normalizer half-width
  at or below `0.00125` nats and agree with the independently assembled target;
  disagreement among individual standard-normal, Student, or Christoffel arms
  is explanatory unless it persists after the DMIS precision check; and
* the explicit tangent agrees with a central finite difference of the same
  frozen value program (absolute and relative error at most `2e-6` in the
  diagnostic range).

The recursive map is a viable candidate only if it additionally has finite
positive-definite covariance and invertible lower-triangular map at every
step, and its held-out residual and cumulative direct-normalizer error are no
worse than the frozen-hint baseline. A lower training RMS alone is not a
promotion criterion.

### Vetoes and explanatory diagnostics

Hard vetoes: wrong target convention, incomplete denominator, invalid masses,
unsupported nonzero target/residual, nonfinite value, nonpositive estimate,
failed tangent parity, failed independent-integration precision, corrupted
snapshot identity, or a model-specific branch in the generic kernel.

Explanatory only unless promoted by a later plan: ESS, maximum weight, shell
residuals, Gram condition, ALS RMS, row-law differences, and runtime.

Nothing here establishes universal finite variance, exact posterior inference,
HMC readiness, production readiness, or superiority over all heuristics.

## 4. Assumption and default audit

| Choice | Provenance and role | Failure mode | Earliest check | Status |
| --- | --- | --- | --- | --- |
| Existing t=2,3,4 snapshots | preserved Stage 2 run | stale or altered target state | metadata/fingerprint and tensor hashes | frozen baseline |
| `gamma=eta*E` in reference coordinates | change-of-measure derivation above | double-counted or omitted eta/Jacobian | standard-normal identity fixture | required invariant |
| Equal component banks with masses `alpha/N_j` | deterministic-mixture identity | treating stratified rows as iid or using `1/N` | mass closure and label permutation | required invariant |
| Standard-normal plus product Student proposal | prior defensive proposal, proposal-only | poor overlap or divergent variance | cross-scramble half-width and tail shells | hypothesis |
| Squared TT control `h=eta*E_H` | exact stored Gram contraction | residual cancellation or nonpositive estimate | CV positivity and residual moment | optional candidate |
| Four scrambles and row ladder | inherited precision contract | underpowered uncertainty | half-width and cross-arm checks | pilot; extend within budget |
| One lagged map update per step | documented recursive construction | feedback amplification or covariance collapse | eigenvalue/map condition trace | hypothesis |
| Frozen rows, maps, and discrete decisions | analytical-gradient contract | omitted adaptation derivative | tangent versus finite difference | required labeling |
| float64 TensorFlow/XLA | repository backend | graph/device mismatch | eager/compiled parity and manifest | implementation choice |

## 5. Execution stages

### Stage A: preflight and target-convention check

1. Verify the snapshot metadata and all serialized tensor hashes for t=2,3,4.
2. Run the existing generic-DMIS unit tests and C2 regression tests with
   `CUDA_VISIBLE_DEVICES=-1`.
3. On a small analytic fixture, verify that `gamma=eta*E` with a standard
   normal proposal reproduces direct reference integration, and that replacing
   the component order leaves the result unchanged.
4. Check the call chain: the new C2 diagnostic must import and call
   `frozen_dmis_control_variate_tf` directly, while the legacy Stage 2 driver
   remains unchanged. A source-only function-existence check is insufficient.
5. Compile the new diagnostic script before any long run.

Failure stops the campaign before GPU work.

### Stage B: precision-repaired frozen integration

Create a fresh output directory under
`docs/benchmarks/artifacts/c2_phase2_generic_dmis_repair_20260902/` for each
attempt. At each captured time, construct paired banks from a standard-normal
component and a product Student-t component (`nu=5`). A calibration partition
tests the predeclared weights `alpha in {0.25, 0.5, 0.75}`; select at most one
weight by calibration half-width, then freeze it for the untouched claim
partition. Use the same rows for plain DMIS and CV.

Evaluate row counts `8192, 16384, 32768`; use four independent scrambles at
each count, extending to eight only if the half-width is above the threshold
and the remaining budget permits it. Keep the target evaluator, snapshots,
dtype, seeds, and map fixed. Record:

* plain DMIS and CV estimates and log estimates;
* direct `Z_H`, residual second moment, target ESS, and maximum weight;
* shell-wise target and residual energy;
* cross-scramble confidence intervals and cross-arm disagreements; and
* analytical tangent versus central finite difference at the frozen rows.

The result is a precision repair only if all three times pass the precision
criterion. Otherwise record `PHASE2_INTEGRATION_UNRESOLVED` and do not infer
a representation or recursion cause.

### Stage C: frozen one-step correction comparison

Only after Stage B passes, compare at the earliest divergent time and then at
the remaining captured times:

* old Gram normalizer;
* plain complete-DMIS;
* complete-DMIS with the squared-TT control variate; and
* the independent reference estimate formed from the highest-precision bank.

Use identical rows and target evaluations for the three estimators. The CV
estimate must use the stored exact Gram integral, not a row-estimated control
normalizer. This stage distinguishes a bad fitted representation from a bad
proposal correction.

### Stage D: recursive lagged moment-map pilot

Only after the frozen one-step comparison is valid, run one lagged update per
time step through the first four observations. The candidate construction is:

1. start with the previous retained quadratic form;
2. propagate it through the exact transition (linear moment identities for a
   linear fixture, or weighted transition evaluations for the nonlinear C2
   fixture);
3. compute predicted mean, covariance, and current/previous cross-covariance;
4. check SPD and construct the lower-Cholesky joint map;
5. fit the exact likelihood-corrected target in that map using fixed rows; and
6. retain the resulting quadratic form and pass only its lagged moments to the
   next step.

The implementation must expose a TensorFlow callable for the moment
contractions and map construction. If the current C2 engine cannot provide
that callable without changing its target convention, run the same mechanics
on the analytic linear fixture and record the C2 recursive stage as a true
continuation veto; do not silently substitute external GH9 moments and call
that a recursive repair. Compare against the unchanged external GH9 hint using
the same basis, rank, sweeps, rows, proposal, observations, and seeds. Do not
use an inner fixed-point loop. For each time record map eigenvalue margins,
Cholesky diagonals, map condition, held-out central and tail residuals, direct
normalizer error, and recursive cumulative error. Freeze all choices before
the tangent check; report a partial/frozen score if map derivatives are not
included.

### Stage E: bounded replicated decision run

If the pilot survives, run at most twelve paired branches on disjoint
calibration/claim partitions. Apply the constructed heuristic adversaries
(Gaussian moment match, Student tail proposal, bootstrap, stationary Gaussian,
and retained TT) conditionally at t=3, t=4, the minimum-ESS time, and the full
horizon. Use paired bootstrap intervals and a sign test. A heuristic win is a
promotion veto, not a tuning target.

No basis rewrite, Student TT measure rewrite, HMC run, or package installation
is part of this campaign.

## 6. Commands and environment

Preflight and unit tests:

```text
CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/highdim/test_frozen_dmis_control_variate_tf.py \
  tests/highdim/test_c2_ukf_guided_tt_dmis_tf.py \
  tests/highdim/test_c2_sv_frozen_proposal_apf_tf.py \
  tests/highdim/test_c2_gaussian_hermite_proposal_tf.py
```

The frozen integration diagnostic will be added as
`docs/benchmarks/run_c2_phase2_generic_dmis_repair_20260902.py` and invoked
with a required fresh `--output-root`, a fixed row ladder, and an explicit
`--jit-compile` flag defaulting to true. Serious GPU invocations must set
`TF_FORCE_GPU_ALLOW_GROWTH=true`, use the `tftwogpu` environment, record the
GPU placement probe, and run with trusted/escalated GPU access. CPU-only
invocations are mechanics/reference diagnostics and must say so in their
manifest.

## 7. Budget, stop rules, and artifacts

Budget: one preflight attempt, at most three localized implementation/harness
repairs, three row counts, eight scrambles maximum per time, and at most twelve
paired recursive branches. The campaign has a six GPU-hour and four CPU-hour
ceiling. Every attempt gets a new directory and preserves previous output.

Stop interpretation for any hard veto, missing artifact, target mismatch,
nonfinite value, failed derivative, or precision failure. A low ESS, a large
fit gap, or a candidate losing to a simple proposal is a repair trigger and
does not invalidate the algebraic identity.

Each result directory must contain `result.json`, `result.md`,
`run_manifest.json`, and (for recursive runs) captured moment/map diagnostics.
The manifest records commit, dirty-worktree digest, exact command, Python and
TensorFlow/TFP versions, dtype, XLA/TF32, devices and memory policy, seeds,
row counts, observations, snapshot fingerprints, wall time, and output hashes.

## 8. Pre-mortem

| Misleading outcome | Discriminating check |
| --- | --- |
| DMIS agrees because target and reference share a Jacobian bug | independent eta-weighted analytic fixture and a second direct density calculation |
| CV appears to repair the scalar through cancellation while being unstable | residual second moment, positivity, and independent scrambles |
| high ESS hides a wrong finite target | direct target convention and independent reference integral |
| recursive map improves central RMS but worsens tails | shell residuals and first-time recursive growth |
| tangent passes while map dependence is omitted | perturb frozen map/proposal values and compare partial versus total programs |
| GPU/XLA result differs from the tested path | callable-boundary parity and device/manifest checks |

## 9. Decision table

| Decision | Primary criterion | Veto | Next action | Not concluded |
| --- | --- | --- | --- | --- |
| Precision repair | all t=2,3,4 half-widths pass | QMC/cross-arm uncertainty | improve integration or stop classification | representation cause |
| Frozen DMIS | complete identity and independent agreement | target/support/tangent failure | retain as correctness comparator | recursive repair |
| TT control variate | lower residual variance with positive estimate | cancellation/nonpositive value | use plain DMIS | universal variance reduction |
| Recursive map | no worse held-out/direct error and valid maps | SPD/Cholesky/map failure | retain external hints and localize | global convergence |
| Replicated candidate | paired uncertainty and heuristic gate | heuristic dominance veto | keep diagnostic only | default readiness |

## 10. Required result interpretation

The final result note must answer separately whether the run invalidated the
harness, implementation, target convention, mathematics, or only the current
candidate. It must include a hard-veto table, an inference-status table, a
post-run red-team note, and an explicit statement of what remains untested.
