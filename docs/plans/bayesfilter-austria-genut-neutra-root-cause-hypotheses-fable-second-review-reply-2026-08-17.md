# Fable Second Review Reply: Revised Austria GenUT NeuTra Root-Cause Plan

Date: 2026-08-17

Status: `SECOND_READ_ONLY_REVIEW_COMPLETE`

Reviewer: Fable (claude-fable-5), independent mathematical and code reviewer

Reviewed artifact (revised plan, mtime 21:17):
`docs/plans/bayesfilter-austria-genut-neutra-root-cause-hypotheses-fable-handoff-2026-08-17.md`

Review request:
`docs/plans/bayesfilter-austria-genut-neutra-root-cause-hypotheses-fable-second-review-request-2026-08-17.md`

First audit:
`docs/plans/bayesfilter-austria-genut-neutra-root-cause-hypotheses-fable-audit-reply-2026-08-17.md`

---

## VERDICT: AGREE

All seven first-audit findings (F1–F7) are closed by the revised plan. No
material finding blocks Phase 0. Phase 0 source/evidence freeze may begin,
while Austria remains blocked from NeuTra and the scientific target, default
policy, and tuning-artifact status remain unchanged.

Three non-blocking clarifications are recorded below (C1–C3). They are
experimental-design wording tightenings for Phases 2 and the H6 arm; none
blocks Phase 0, and each can be absorbed as a one-line note in the Phase 1/2
execution artifacts rather than another plan revision cycle.

### Source-state note

`cubature_genut_batch_tf.py` (mtime 21:21) and
`cubature_genut_neutra_targets.py` (mtime 21:54) carry modification times
*after* the first audit (18:51). I re-verified before this verdict: line
counts (1623 / 698), the full function map, and the critical anchor regions
(`_shape_iteration_batch_jvp` restandardization at
`cubature_genut_batch_tf.py:608-618`, `_higher_moment_batch_value` single
standardization at `:994-997`, `GenUTControls` fields at
`cubature_genut_neutra_targets.py:48-58`, `_AUSTRIA_CONTROLS` at `:116-126`)
are identical to the first-audit state. The mtimes are touches, not semantic
edits. The plan's own rule (Confirmed Route Identity / "Phase 0 preservation
test": reclassify `unknown_route_identity` if the callable changed) plus
Phase 0 step 1 source hashing is the correct and sufficient guard; Phase 0
hashing remains mandatory and definitive.

---

## Answers To The Acceptance Questions

Numbering follows the review request; the plan's own eight
"Second-Review Acceptance Questions" are covered by the same answers.

1. **F1 route identity — closed.** The "Confirmed Route Identity" section now
   states the route as diagonal correction plus ordinary Contract-E reset and
   final affine restoration, with the missing stages named exactly (pairwise
   co-skew/co-kurtosis, pairwise row-RMS radial cap, coordinate cap), states
   that `higher_moment_trust_radius` is not the dual-cap radial cap, and
   classifies the blocker as `batch_diagonal_candidate` that "cannot be
   reported as a failure of the promoted dual-cap algorithm". R0's
   source-confirmed classification and the `unknown_route_identity` fallback
   for post-audit callable drift are both correct.

2. **F2/H1 intervention — closed.** The revision ledger and Phase 3 pin the
   forward arm to removing exactly the start-of-iteration restandardization,
   retaining the shared outer standardization (in
   `_higher_moment_batch_jvp`) and the post-correction restandardization.
   The arm table keeps the forward arm target-preserving ("tests H1 while
   preserving value order") and the reverse arm explicitly target-changing
   and non-eligible as a repair. Correct as labeled.

3. **F3/H4 fail-closed semantics — closed.** H4 is rewritten to predict the
   NaN value/score pair with permanently latched invalid status; Phase 5
   predeclares that outcome; the decision rules now split
   `PASS_H4_FAIL_CLOSED_REGRESSION` from `CONFIRM_H4_FAIL_CLOSED_DEFECT`
   (finite scalar escape only). Endpoint validity-domain asymmetry and
   invalid-row diagnostics gated by `valid_pre_regularized_score` are both
   captured in H4's statement and required semantics. This matches the
   inspected source behavior (`cubature_genut_batch_tf.py:1612-1615`,
   `cubature_genut_neutra_targets.py:412-419`).

4. **F4/H3A — closed.** H3A is scoped to `minimum_gap_eigenvalue` and the
   `gap_valid` branch only, added as Phase 2 step 6, excluded as an
   explanation of the finite particle mismatch, and given the correct
   fail-closed expectation for a knife-edge flip. The description of the
   asymmetry matches the source (`_sym` before the difference in the JVP
   route via `_weighted_moments_jvp`, `cubature_genut_batch_tf.py:146` and
   `:457-465`, versus `_sym` of the difference only in the value route,
   `:382-392`). See C1 for a stop-rule wording note.

5. **F5 instrumentation — closed.** Phase 1 declares interior capture
   eager-only and graph/XLA endpoint-only with no interior fetches or debug
   identities; H3's discriminating test is amended consistently. This removes
   the self-firing continuation veto. See C2 for the H6 block-arm
   interpretation note.

6. **F6 tangent audit — closed.** H5 records the local direct terms
   (`∂κ/∂θ₀`, `∂ν/∂θ₁`, `∂σ²/∂θ₂ = 2σ²`) and RK4 JVP structure as passing
   static derivation review, lowers the prior accordingly, and still requires
   the composed autodiff-block and `h²` whole-program checks. Correctly
   scoped: local verification does not validate the composed `T=20`
   derivative.

7. **F7 test economy — closed.** Phase 2 steps 1–5 are labeled confirmatory
   regression; step 7 is the discovery boundary; the injected tangent-only
   failure is a fail-closed regression guard with the predeclared NaN
   outcome.

8. **Shared-primal repair architecture — closed.** Phase 3 requires the
   candidate production repair to be one shared-primal correction core with
   tangent computation under a Python-level `tangents is not None` branch,
   and explicitly forbids restoring separate value/JVP primal
   implementations. Phase 7 requires a batch-native dual-cap route whose
   ledger binds pairwise moments, pairwise radial cap, coordinate cap, and
   affine restoration before any dual-cap NeuTra claim. This prevents
   recurrence of the defect class.

9. **Execution order — valid.** `R0 → eager Phase 1/2 → H1 → H2/H3A/H3 →
   fail-closed H4/H7 → H5 → arithmetic replication → fresh tuning` is
   logically consistent with the dependency graph; H2 is not a substitute
   for H1; H5 is gated on primal identity. No proxy is promoted to a
   criterion (condition numbers, LM behavior, UKF/SGQF proximity, and FD
   residual size remain explanatory); exact equality is required where a
   cached primal tensor is contractually shared; the historical `2e-4`
   threshold is rejected for future admission; no tolerance may be invented
   post hoc; H8 and Phase 8 forbid stale tuning reuse, and the plan nowhere
   consumes the existing Austria tuning artifact as claim evidence.

10. **Remaining blockers — none material.** No missing hypothesis or
    confound blocks Phase 0. The clarifications below are non-blocking.

---

## Non-Blocking Clarifications (record in the Phase 1/2 execution note)

### C1 (experimental-design wording, Phase 2) — stop rule versus step 6

Phase 2 ends with "Stop at the first unequal tensor". Step 6 (H3A
covariance-gap comparison) is *expected* to show a ULP-level difference in
`minimum_gap_eigenvalue` while all particle tensors still agree. Read
literally, the stop rule would halt localization at step 6 and never reach
the step 7 discovery boundary. The surrounding text already distinguishes
step 6 ("records the validity-only H3A difference") from step 7 ("the
discovery boundary"), so the intent is unambiguous; the execution artifact
should state explicitly that the stop rule applies to particle-path tensors,
and that an unequal step 6 gap scalar with equal particles is the
predeclared `CONFIRM_H3A_VALIDITY_ONLY_ASYMMETRY` outcome, not a stopping
trigger.

### C2 (instrumentation interpretation, H6) — extracted-block graph/XLA arms

H6 runs "the same frozen lower-level boundary" under eager/graph/XLA. Running
an *extracted* block under `tf.function`/XLA and fetching only its output is
consistent with F5 (the fetch is the endpoint of that small program). But
XLA fusion of an extracted block can differ from fusion of the same block in
situ inside the full endpoint graph, so block-level graph/XLA parity must be
interpreted as mode-sensitivity evidence for the block only, never as proof
of endpoint-level in-situ arithmetic identity. Endpoint-level claims under
graph/XLA come only from the Phase 1 endpoint-only arms.

### C3 (baseline hygiene, Phase 2) — confirm eager reproduction before localization

The preserved endpoint gap and probe evidence do not record eager/graph mode
explicitly. H1 is a source-level program asymmetry and is expected to
reproduce in eager mode, and Phase 1's required `correction_steps=0`
regression plus the Phase 2 `T=1` baseline would surface any failure to
reproduce; the execution note should simply record the eager-mode endpoint
gap before starting tensor-level localization so the localization campaign
is anchored to a reproduced mismatch in the same mode.

---

## Not Checked In This Review

- The numerical observations (`5.6596` endpoint gap, `~24` zero-tangent
  first-iteration cloud difference, `~1.2e5` condition estimate) remain
  preserved diagnostics, not independently reproduced, per the review
  contract.
- The two touched source files were re-verified by line count, function map,
  and anchor-region spot checks, not by full byte-for-byte comparison
  against the first-audit state; Phase 0 source hashing is the definitive
  check and remains mandatory.
- No GPU command, test, or harness was executed; this review is static plan
  and source-claim validation only.

---

## Execution Boundary

This `AGREE` verdict authorizes exactly one thing: the implementing agent may
begin Phase 0 (route/evidence freeze, no GPU) of the revised plan, followed by
the subsequent phases under the plan's own budget (three attempts, 30
GPU-minutes for Phases 1–6) and stop conditions. It does not authorize
editing production source beyond the plan's diagnostic-only harness and
clearly labeled diagnostic intervention arms, changing the GenUT default or
the Austria scientific target, reusing the historical Austria tuning artifact
for any claim, running NeuTra or HMC, admitting Austria to NeuTra, applying
the diagonal-lane diagnosis to the dual-cap route, or treating any finite or
damped score as correct. A causally supported H1 repair still requires the
shared-primal architecture, Phase 7 cross-model regression, and fresh
scope-specific tuning (H8/Phase 8) before any admission claim.

VERDICT: AGREE
