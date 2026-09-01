# C2 Phase 2 DMIS Contract-Correction Re-run

Date: 2026-09-02  
Status: executed; plain-DMIS precision gate failed  
Parent plan: `bayesfilter-c2-phase2-generic-dmis-recursive-repair-plan-20260902.md`  
Scope: C2 n=4 frozen diagnostic snapshots only; no production or default change

## 1. Reason for this correction

The parent plan, as present at the run commit, declared the primary precision
screen on the plain complete-deterministic-mixture importance estimate. The
driver used the field named `log_normalizer`, which is the squared-TT
control-variate estimate, for both calibration and the precision boolean. The
driver therefore did not execute the declared promotion screen, even though it
also recorded the plain estimate. This is an implementation/contract mismatch,
not evidence against either estimator.

The correction is deliberately narrow: preserve the target, snapshots, rows,
seeds, proposal families, mixture identity, control-variate calculation, and
tangent check; select the Student mass and apply the precision gate using the
plain complete-DMIS log estimate exactly as the parent plan specified. The CV
estimate remains a secondary diagnostic.

## 2. Research question and evidence contract

For each frozen snapshot, the target remains

\[
  \gamma_t(u)=\eta_{2n}(u)E_t(u),
\]

with the complete proposal denominator containing every standard-normal and
Student component. The primary question is whether the plain complete-DMIS
estimate has a cross-scramble 95% log-normalizer half-width no larger than
`0.00125` nats at every captured time and row count. The independently
assembled row-wise target check and frozen tangent check remain validity
requirements. The squared-TT known-integral control variate is reported for
variance/cancellation diagnosis, but it is not silently substituted for the
declared primary estimator.

The result cannot establish the true C2 marginal likelihood, posterior
correctness, recursive-map quality, or universal finite variance.

## 3. Correction and fixed choices

* In calibration, choose `alpha` from `{0.25, 0.5, 0.75}` by the maximum
  plain-DMIS log half-width, then freeze it.
* In the claim ladder, set `precision_pass` from
  `plain_log_normalizer['half_width_95']` at all times and row counts.
* Add explicit metadata naming `plain_complete_dmis_log_normalizer` as the
  precision estimator and `plain_complete_dmis_log_half_width` as the gate.
* Keep CV, Gaussian-only, Student-only, ESS, shell, Gram, and residual
  diagnostics unchanged and explanatory.
* Do not alter target assembly, proposal densities, masses, random seeds,
  snapshots, row counts, or the recursive-stage gate.

## 4. Execution

Run a fresh output directory (`attempt09`) with the parent ladder. `attempt08`
was launched before the corrected source was present and is retained as a
duplicate diagnostic, not as evidence for this contract:

* 8,192, 16,384, and 32,768 rows per component;
* eight claim scrambles and two calibration scrambles;
* float64 TensorFlow/XLA with the repository GPU memory-growth policy; and
* the same `tftwogpu` environment and preserved snapshots.

The run is valid only if all target/support/mass/mixture/tangent checks pass.
It is a precision success only if the plain-DMIS screen passes at all three
times and all three row counts. A precision failure is a candidate veto, not a
causal diagnosis of the TT representation or recursion.

## 5. Skeptical pre-run audit

Before launch, verify that:

1. the driver’s calibration and claim code read `plain_log_normalizer` for the
   selection and gate;
2. the result Markdown labels the primary estimator unambiguously;
3. the target and complete-mixture call chain are unchanged;
4. the fresh output root cannot overwrite attempts 02--07; and
5. the run manifest records the corrected source hash, command, environment,
   GPU placement, memory policy, and output hashes.

The run must stop on any source/target/seed drift, nonfinite value, unsupported
row, failed tangent, or missing manifest field.

## 6. Interpretation and next step

If the corrected plain-DMIS gate passes, the parent plan may proceed to its
one-step comparison and recursive-map callable work. If it fails, classify the
DMIS/CV proposal bank as unresolved at this scope and design a separate,
predeclared tail-aware proposal experiment. Do not claim that the recursive
map failed, because it has not been run under its own target and diagnostics.

## 7. Budget and artifacts

This correction consumes one bounded GPU attempt within the parent six-hour
GPU budget. It writes the fresh `attempt09` directory under
`docs/benchmarks/artifacts/c2_phase2_generic_dmis_repair_20260902/`, preserving
all earlier attempts. The correction plan, its review, the source hash, and the
result note are retained with the parent execution record.

## 8. Execution outcome

The corrected run completed with finite values, valid support and mixture
closure, and a passing frozen tangent check. The selected alpha was `0.25`.
The plain-DMIS half-widths still exceeded `0.00125` at t=2 for 8,192 and
32,768 rows and at t=4 for every row count (the largest was `0.4101` nats).
The claim status is therefore `PHASE2_INTEGRATION_UNRESOLVED`; recursive C2
testing remains unperformed. Full details are in
`bayesfilter-c2-phase2-generic-dmis-contract-correction-execution-result-20260902.md`.
