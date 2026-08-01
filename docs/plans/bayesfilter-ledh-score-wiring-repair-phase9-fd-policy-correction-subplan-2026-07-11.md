# Phase 9 Correction Subplan: Individual-Direction FD Policy

Date: 2026-07-11

Status: `IMPLEMENTED_RECLASSIFIED_OWNER_CLARIFICATION`

## Correction Objective

Replace the unsupported Phase 9 `0.005` absolute-or-relative finite-difference
gate with the owner-directed FD-only rule, reclassify the existing trusted
GPU/XLA shards without changing them, and update the shared runner for future
FD evidence.

Two owner clarifications supersede the first version of this correction:

1. The rule applies only to the finite-difference diagnostic. It is not a
   general score-validity, HMC-readiness, posterior-validity, or scientific
   screen.
2. Parameter directions are checked individually. A rung passes when the
   maximum coordinate-relative FD error is at most `0.05 * sqrt(p)`. The first
   correction's `2%` RSS/RMS interpretation was wrong and is superseded.

The original Phase 9 score and FD measurements remain valid observations. The
old `0.005` decisions and the intervening `2%` RSS/RMS reclassification are not
authoritative policy evidence.

## Research Intent Ledger

| Field | Intent |
| --- | --- |
| Main question | Does each stored same-scalar FD comparison satisfy `max_j(r_j) <= 0.05 * sqrt(p)`? |
| Candidate or mechanism | The existing compact score for each nonlinear row, compared with its already-computed central FD directions. |
| Expected failure mode | At least one individual direction has relative error above `0.05 * sqrt(p)`; alternatively, source identity or stored evidence is invalid. |
| Promotion criterion | FD diagnostic only: the maximum coordinate-relative error is at most `0.05 * sqrt(p)`. |
| Promotion veto | Nonfinite score or FD, missing/duplicated/misordered direction, source or cross-shard hash mismatch, or failure of the FD-only maximum-coordinate rule. |
| Continuation veto | A stored shard cannot be parsed or identity-bound, a required comparison is missing, or recomputed legacy fields disagree with preserved values. |
| Repair trigger | Any use of `0.005`, `2%`, RSS/RMS aggregation, or general HMC/scientific-screen language as the current FD policy. |
| Explanatory diagnostics | Per-coordinate values and errors, maximum-error direction, threshold, margin, FD step, precision, and rung shape. |
| Must not be concluded | Passing this FD check does not establish general score correctness, HMC readiness, posterior correctness, a calibrated 95% confidence interval, default readiness, full-row admission, or superiority. Failure identifies an FD mismatch at the measured rung but does not by itself isolate score math from finite-difference resolution. |

## Mathematical Policy

For analytic score coordinate `g_j` and its same-scalar finite difference
`d_j`, preserve the Phase 9 coordinate-relative-error definition:

```text
s_j = max(abs(g_j), abs(d_j), 1e-12)
r_j = abs(g_j - d_j) / s_j
```

For `p` individually checked parameter directions:

```text
threshold = 0.05 * sqrt(p)
max_coordinate_relative_error = max_j(r_j)
pass = max_coordinate_relative_error <= threshold
```

There is no RSS or RMS aggregation. For Actual-SV, `p=2`, so the threshold is
`0.05*sqrt(2) = 0.0707106781` (`7.071%`). Its stored maximum coordinate error
is approximately `0.060292` (`6.0292%`), so that comparison passes this FD
check.

The owner chose `5%` to reflect the conventional 95% confidence/significance
threshold analogy. This fixes the intended policy constant, but the FD error
calculation is not itself a confidence interval: it has no estimated sampling
distribution, standard error, coverage calculation, or repeated-sample
calibration. Artifacts must record that distinction without changing the
owner-directed threshold.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Engineering question | Can the runner and offline reclassifier implement the exact maximum-coordinate FD rule and preserve auditable source identity? |
| Exact comparator | The score and same-scalar central-FD values already stored in each original FD shard, cross-bound to its score shard by SHA-256. |
| Primary criterion | Recomputed coordinate errors, maximum, threshold, margin, and boolean decision agree between implementation, tests, JSON, and Markdown. |
| Veto diagnostics | Source/hash failure, nonfinite values, wrong parameter count/order, RSS/RMS aggregation, wrong constant, an absolute-error escape branch, or mutation of an original Phase 9 shard. |
| Explanatory only | Legacy decisions, FD step, precision, runtime, memory, and the magnitude/order of passing coordinates. |
| What will not be concluded | The nonclaims in the intent ledger remain binding even if a row passes the FD check. |
| Preserved artifact | A regenerated JSON reclassification artifact and Markdown result with every source path and SHA-256; original Phase 9 JSON shards remain byte-for-byte unchanged. |

## Skeptical Plan Audit

The inherited `0.005` baseline fails audit because it was an arbitrary CLI
default. The first attempted correction also fails audit because it converted
individually checked directions into an RSS/RMS statistic, used `2%` after the
owner intended `5%`, and broadened an FD-only check into HMC-oriented language.
Those are material target mismatches, so that implementation and its review
must be repaired before further interpretation.

The exact comparator remains the original SHA-256-bound score/FD shard pair at
each rung. The primary metric is now only the maximum coordinate-relative
error. Memory, runtime, aggregate error, and HMC diagnostics cannot substitute
for it. No GPU rerun is needed because the corrected decision is a
deterministic function of stored values; a rerun would not answer the policy
interpretation question.

The stored coordinate-error convention, including its `1e-12` denominator
floor, is preserved rather than introducing a new near-zero policy. Source
hashes will be captured before and after regeneration. A pass removes only
this FD veto. A fail does not establish the causal source of the mismatch.

Audit decision: `PASS_AFTER_OWNER_INTERPRETATION_REPAIR`. Offline
implementation and reclassification may proceed. GPU execution may not.

## Implementation Scope

1. Replace HMC-named/RSS policy symbols with FD-only maximum-coordinate policy
   symbols and fields.
2. Preserve the existing coordinate-relative-error denominator and implement
   `max_j(r_j) <= 0.05 * sqrt(p)` exactly.
3. Update the shared harness to serialize and recompute the FD-only policy.
4. Update the offline reclassifier to validate source hashes and legacy fields,
   then classify all 11 completed comparisons without requiring every row to
   have a stored failure.
5. Add focused tests for dimension scaling, threshold equality, multiple
   individually acceptable coordinates that RSS would reject, Actual-SV's
   `p=2` classification, nonfinite input,
   forged declarations, source hashes, and parameter order.
6. Regenerate the JSON and Markdown correction artifacts without modifying the
   original Phase 9 shards.
7. Replace the prior correction review with a review of the intended FD-only
   rule, then update the reset memo, execution ledger, stop handoff, and
   historical-result supersession notice.

## Planned Artifacts

- Reclassifier:
  `docs/benchmarks/reclassify_ledh_phase9_fd_policy.py`
- Reclassification JSON:
  `docs/plans/artifacts/ledh-score-wiring-repair-phase9-fd-policy-correction/phase9-fd-policy-reclassification-2026-07-11.json`
- Correction result:
  `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-fd-policy-correction-result-2026-07-11.md`
- Focused review:
  `docs/reviews/bayesfilter-ledh-score-wiring-repair-phase9-fd-policy-correction-codex-review-2026-07-11.md`

## Verification Commands

All checks are deliberate CPU-only engineering checks:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  bayesfilter/ledh_fd_policy.py \
  docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py \
  docs/benchmarks/reclassify_ledh_phase9_fd_policy.py \
  tests/highdim/test_ledh_compact_score_gpu_xla_harness.py \
  tests/highdim/test_ledh_phase9_fd_policy_reclassifier.py
```

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest -q \
  tests/highdim/test_ledh_compact_score_gpu_xla_harness.py \
  tests/highdim/test_ledh_phase9_fd_policy_reclassifier.py
```

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python \
  docs/benchmarks/reclassify_ledh_phase9_fd_policy.py \
  --manifest docs/plans/ledh-score-wiring-repair-phase9-fd-reclassification-inputs-2026-07-11.json \
  --output docs/plans/artifacts/ledh-score-wiring-repair-phase9-fd-policy-correction/phase9-fd-policy-reclassification-2026-07-11.json \
  --markdown-output docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-fd-policy-correction-result-2026-07-11.md
```

## Pre-Mortem And Stop Conditions

- Wrong aggregate rejection: RSS/RMS aggregation could reject several
  directions even though every individual direction is within the specified
  threshold. The maximum coordinate controls the decision and a focused test
  exercises that case.
- Misleading statistical claim: `5%` could be reported as if an actual 95%
  confidence interval had been calculated. Artifacts must call it the policy
  motivation and state that coverage is not calibrated.
- Historical drift: source shards could change during repair. Stop if any
  before/after SHA-256 differs.
- Selective reporting: only failed rungs could be reported. Reclassify all 11
  completed live Gate B/Gate C comparisons.
- Overreach: an FD pass could be promoted into HMC or scientific evidence.
  Keep those ledgers and claims separate.
- Stop immediately if a source pair is not identity-bound, legacy arithmetic
  cannot be reproduced, or focused tests disagree with offline arithmetic.
