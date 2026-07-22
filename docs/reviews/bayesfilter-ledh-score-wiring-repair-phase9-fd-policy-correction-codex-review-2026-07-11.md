# Codex Review: Phase 9 FD-Only Policy Correction

Date: 2026-07-11

## Supersession

This review replaces the earlier review at this path. The earlier
`VERDICT: AGREE` reviewed a `2%` RSS/RMS HMC-oriented interpretation that was
wrong relative to the owner's intended target. It is not authoritative.

## Review Scope

- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-fd-policy-correction-subplan-2026-07-11.md`
- `bayesfilter/ledh_fd_policy.py`
- `docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py`
- `docs/benchmarks/reclassify_ledh_phase9_fd_policy.py`
- `tests/highdim/test_ledh_compact_score_gpu_xla_harness.py`
- `tests/highdim/test_ledh_phase9_fd_policy_reclassifier.py`
- `docs/plans/ledh-score-wiring-repair-phase9-fd-reclassification-inputs-2026-07-11.json`
- `docs/plans/artifacts/ledh-score-wiring-repair-phase9-fd-policy-correction/phase9-fd-policy-reclassification-2026-07-11.json`
- `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-fd-policy-correction-result-2026-07-11.md`

Key reviewed hashes before this review file was written:

- policy module: `e1999cbe08312048abd2eab30c15dc33adaf520b5d745f47e139aac3eac79da2`;
- shared harness: `76604aae5460a2410e8c03e5a5778ab46dc80b44aef2b3201a6141037539499d`;
- reclassifier: `ffe0277122e8ba15e99992e6ae27723609ae3fbe1087d816df956e8ce93fa897`;
- correction plan: `935f27eea038813442be2cde7cd9ceece9a2827ed7dc6852a1c999476ad9d12a`;
- correction result: `12108b9dc32283c2a42ddcb72937a87853ba381cd97416a53d718e52a327bbaf`;
- reclassification JSON: `1ffa3fd9fdf74050d667b4205c8545e56657f0102b81fb28933894bd3644a4dd`.

## Review Question

Does the correction implement the owner-directed FD-only rule exactly,
preserve the original Phase 9 source evidence, reclassify all completed
comparisons coherently, and avoid promoting the FD result into a general HMC or
scientific-validity claim?

## Findings

No material finding.

The claimed target is now explicit:

```text
r_j = abs(score_j - FD_j) / max(abs(score_j), abs(FD_j), 1e-12)
threshold = 0.05 * sqrt(p)
pass iff max_j(r_j) <= threshold
```

The quantity actually computed equals that target. The reusable policy uses
the preserved historical coordinate denominator, selects the largest
individual-direction relative error, and compares it directly with
`0.05*sqrt(p)`. It does not compute RSS, RMS, or an average, and it has no
absolute-error pass branch. The serialized schema states
`diagnostic_scope=finite_difference_only`.

An adversarial focused test covers the interpretation boundary: four
coordinates at or below the individual-direction threshold pass even though
RSS aggregation would produce a different decision. Threshold equality,
dimension scaling, the denominator floor, nonfinite input, forged declarations,
source hashes, cross-shard score binding, and parameter order are also tested.

The reclassifier reads all 11 explicitly manifested Phase 9 score/FD pairs,
verifies their SHA-256 values and embedded FD-to-score binding, reproduces the
legacy stored coordinate errors, and applies the corrected policy. It no longer
assumes that every nonlinear row must fail.

Independent standard-library arithmetic checked every entry's boolean against
`max_error <= 0.05*sqrt(p)`. It also confirmed that the JSON has no
`relative_error_rss` or `relative_error_rms` result fields. The resulting
classification is:

- 9 stored comparisons pass;
- predator-prey fails Gate B `T=1,N=2`;
- generalized-SV passes Gate B and fails Gate C `T=4,N=10000`;
- fixed-SIR, Actual-SV, and KSC-SV historical terminal comparisons pass.

Actual-SV is correctly represented with two parameters,
`gamma_unconstrained` and `log_beta`. Its maximum direction is `log_beta` at
`0.0602924688125`; the threshold is
`0.05*sqrt(2)=0.0707106781187`, so its FD status is pass.

The `5%` value is owner-directed and motivated by the conventional 95%
threshold. The result correctly states that no confidence interval is actually
computed: there is no sampling distribution, standard error, coverage
calculation, or repeated-run calibration. Passing is therefore only an FD
diagnostic outcome, not general score correctness, HMC readiness, posterior
correctness, default readiness, or scientific validity.

No GPU rerun was performed or needed for this deterministic reclassification.
The future harness schema is v3. The historical exact-command manifest remains
blocked because it predates the corrected schema and targets immutable v1
source paths.

## Verification

- syntax compilation: passed;
- policy/reclassifier suite: `9 passed in 0.08s`;
- shared CPU-hidden harness suite: `89 passed, 2 warnings in 90.13s`;
- independent JSON arithmetic/field audit: `AUDIT_OK`, 9 pass / 2 fail;
- original Phase 9 source-shard SHA-256 comparison: passed;
- `git diff --check`: passed.

These are engineering and deterministic-reclassification checks. They do not
provide new GPU, HMC, posterior, or statistical evidence.

VERDICT: AGREE
