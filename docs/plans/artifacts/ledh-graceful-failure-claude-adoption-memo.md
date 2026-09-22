# Claude adoption memo: revised LEDH graceful-failure program

**Prepared:** 2026-09-17. **Status:** proposed handoff; no implementation or campaign has been launched by this memo.

Adopt the [revised repair and validation program](../ledh-graceful-failure-revised-program-2026-09-17.md) as the active program when the owner directs execution. It replaces the active execution instructions of the earlier comprehensive testing plan. Preserve the earlier plans and results as dated evidence.

The starting verdict is **major revisions required**, supported by the [Codex technical review](ledh-graceful-failure-codex-review.md) and its [diagnostic manifest](ledh-graceful-failure-review-20260917-01/review-manifest.json). The reviewed revision is `8a5c23ab1172884ffca63cac390623bf7735afe7` on `surrogate-hmc`; check relevant dirty sources before starting.

The immediate defect is reproducible: the canonical routine's failure sentinel does not survive the real batch wrapper and dual target. Failed factors also enter derivative solves, and the safety helper has finite-input and batch-shape defects. A larger HMC campaign cannot resolve those implementation problems. Existing “production ready” statements are unsupported.

The revised program makes three independent decisions: whether numerical failures are safely contained; whether the analytical score differentiates the actual finite likelihood; and whether each sampled posterior agrees with its stated reference. Report these separately throughout the work.

Follow this execution order after authorization:

1. Verify the checkout, applicable instructions, review evidence, and actual consumer endpoints. Preserve unrelated edits and create a fresh versioned output root. Correct active status language without rewriting historical results.
2. Reproduce and repair R1/R2/R6 through the canonical routine, both batch wrappers, dual target, and real runner. Preserve a valid primal when only its force fails, make inactive derivative arithmetic safe, and retain per-row status across observation time.
3. Carry LG1 through the complete derivative, tuning, consumer, and posterior-reference sequence. This is the first end-to-end validation. Do not postpone it until every model adapter is built.
4. Expand to LG2/LG4 and eligible nonlinear cases, then costly LG20 posterior work. Use the corrected equations, explicit priors, and scope-specific tuning in the program. Fixed SIR is a filtering case with no parameter-recovery claim.
5. Preserve every failed attempt, report independent reference uncertainty and numerical exclusion, and close out with separate scope-level decisions. Run capacity/timing checks only after correctness.

Several requirements change the earlier proposal materially:

- A failed proposal force must not silently turn a valid endpoint likelihood into `-inf`. The proposed deterministic zero extension needs checked reversal, mass-coordinate, and endpoint-Metropolis mechanics.
- Numerical likelihood failure can restrict the target. Proposal rejection frequency does not measure excluded posterior mass; a reference using the same failure mask cannot detect the missing mass.
- Stationary initialization, parameter-dependent noise, prior transformations, and Contract E reset dependencies belong in the analytical score. Autodiff remains diagnostic only.
- The stable partially observed LGSSM replaces the original invalid Case 3. Predator–prey and SIR use the local equations. Basic SV uses its true state-dependent observation density and needs a compatible canonical proposal interface.
- LEDH approximation tuning and HMC kernel tuning have different authorities and purposes. Use the live public tuner registry and typed force binding where required; do not invent tuning artifacts.
- Truth within two posterior SDs, ESS above 100, a toy rejection test, and a seven-out-of-eight aggregate are not acceptance criteria. Use the program's reference, uncertainty, per-case, and conditional heuristic checks.

The proposed campaign ceiling is **64 GPU device-hours and 16 aggregate CPU worker-hours**, with at most three substantive attempts per case/contract. These are proposed bounds, not permission granted by this memo or a promise that all scopes fit. Forecast costs before expensive work; mark unaffordable claims under-budgeted instead of weakening their evidence requirements. The full program specifies allocations, repair rules, and artifact contents.

A later plain-language owner request to execute within these bounds supplies campaign authorization. Local repairs and retries within that authorization need no new procedural approval. The normal GPU/sandbox, external-action, resource, and scientific-direction boundaries still apply. Do not launch another agent or send an external message on the authority of this memo.

Read the full [program](../ledh-graceful-failure-revised-program-2026-09-17.md) before implementation. Its phase prerequisites, proposed values, calibration rules, and evidence limits are part of the task; this short memo is an entry point, not a substitute specification.
