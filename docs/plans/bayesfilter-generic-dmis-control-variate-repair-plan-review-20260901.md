# Skeptical Review: Generic DMIS/TT-Control-Variate Repair Plan

Date: 2026-09-01  
Reviewed plan: `bayesfilter-generic-dmis-control-variate-repair-plan-20260901.md`

## Review method

I audited the plan before implementation against the repository governance
requirements: target identity, comparator ladder, promotion versus explanatory
diagnostics, stop conditions, default/assumption provenance, deterministic
versus iid sampling semantics, total-derivative scope, artifact reproducibility,
and the requirement that C2 remain a holdout rather than a runtime branch.

## Findings and repairs

1. **Sampling semantics.** A deterministic stratified bank does not become an
   unbiased estimator merely because it uses importance ratios. The plan now
   labels its finite-bank expression as a weighted quadrature value and reserves
   unbiasedness language for iid mixture draws. The focused tests use a
   closed-form finite sum for the deterministic bank; the iid statement is
   retained as a proposition/Lean boundary rather than a noisy promotion
   test.
2. **Control-variate positivity.** A signed residual estimator can be negative
   at finite sample size even though its population target is positive. The plan
   treats nonpositive realized control-variate normalizers as a validity flag,
   while retaining the plain DMIS estimate as the correctness comparator. No
   clipping or logarithm of an absolute value is permitted.
3. **Proxy metrics.** ESS, maximum weight, shell error, and conditioning are
   explicitly explanatory or nomination diagnostics. They cannot promote a
   candidate without the exact-target and derivative checks.
4. **Baseline and fairness.** The baseline is the current complete-mixture
   route, with Gaussian, Student, bootstrap, stationary, and retained-TT arms.
   Component banks share declared masses and paired inputs. No historical C2
   value is silently reused as a current measurement.
5. **Generality.** The planned module accepts tensors for arbitrary target and
   component log densities and contains no C2 import or model-name conditional.
   The C2 smoke is explicitly compatibility-only.
6. **Gradient scope.** The tangent formula differentiates the exact finite
   frozen value program. If a caller supplies map/proposal pathwise tangents,
   they are included; otherwise freezing is declared. This avoids claiming an
   adaptive gradient from a partial derivative.
7. **Budget and failure handling.** The plan has bounded retries, fresh output
   paths, and continuation vetoes. A weak proposal or failed C2 candidate is a
   repair trigger; an identity, support, or artifact failure stops
   interpretation.

## Execution-scope amendment

The initial review passed the implementation plan with MathDevMCP's standard
label budget.  Because the user explicitly requested a full document review
and the CLI exposes an `--max-labels 0` exhaustive mode, the execution scope is
amended to include one bounded all-label pass.  This does not change the target,
promotion criteria, vetoes, or compute class; it only replaces a partial
coverage diagnostic with the strongest local coverage run available.  The
report must still distinguish coverage from proof and preserve any labels that
the backend cannot formalize.

## Verdict

`PASS FOR EXECUTION`.

The plan has a coherent scientific question, an exact baseline and target
boundary, explicit hard vetoes, and a bounded implementation/test scope. The
remaining uncertainty is empirical proposal efficiency, which the planned
diagnostics are designed to measure and which is not being treated as a
mathematical consequence.

## Post-execution consistency check

The implementation added one fail-closed safety condition that was not
material to the original plan text: the tangent kernel masks zero proposal
density and reports `tangent_support_valid=false`, because the displayed
derivative theorem assumes positive density at every frozen row.  An explicit
regression test covers this boundary, and the plan now records it.  This is a
localized safety clarification; it changes neither the target nor the
promotion contract.
