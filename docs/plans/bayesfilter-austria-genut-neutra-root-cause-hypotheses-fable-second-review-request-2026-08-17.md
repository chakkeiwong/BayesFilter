# Fable Second Review Request: Revised Austria GenUT NeuTra Root-Cause Plan

Date: 2026-08-17

Status: `READY_FOR_BOUNDED_READ_ONLY_REVIEW`

## Review Contract

READ-ONLY BOUNDED REVIEW. Review exactly this path first and nothing else
unless the path itself explicitly asks you to inspect a cited source line:

```text
docs/plans/bayesfilter-austria-genut-neutra-root-cause-hypotheses-fable-handoff-2026-08-17.md
```

Do not edit files, run commands, launch agents, run GPU work, or review the
whole repository. Codex remains supervisor and executor. This request asks
whether the revised plan is now executable; it does not authorize source
changes, diagnostic runs, tuning, NeuTra training, HMC, or default changes.

End with exactly one of:

```text
VERDICT: AGREE
VERDICT: REVISE
```

If `REVISE`, list only material findings that block Phase 0, with exact plan
section and source anchors. Classify each as a mathematical, target-identity,
experimental-design, instrumentation, validity-mask, scope/tuning, or
governance defect. Formatting preferences and retired launch ceremony are not
blocking.

## Context

The first audit is preserved at:

```text
docs/plans/bayesfilter-austria-genut-neutra-root-cause-hypotheses-fable-audit-reply-2026-08-17.md
```

It returned `VERDICT: REVISE` with findings F1-F7. The revised plan includes a
revision ledger and changes the execution protocol. Do not re-audit the whole
repository; check whether the revised plan actually closes those findings.

## Acceptance Questions

1. **F1 route identity.** Does the revised plan now state precisely that the
   current batch NeuTra target is `batch_diagonal_candidate`: diagonal
   higher-moment correction plus ordinary Contract-E reset and affine
   restoration, with no pairwise correction, no pairwise radial cap, and no
   standardized coordinate cap? Does it avoid calling this route promoted
   dual-cap GenUT?

2. **F2/H1 causal intervention.** Does the revised H1 intervention remove
   exactly the JVP path's redundant start-of-iteration restandardization while
   retaining the shared outer and post-correction standardizations? Is the
   forward arm target-preserving and the reverse arm explicitly diagnostic and
   target-changing?

3. **F3/H4 fail-closed semantics.** Does the revised plan correctly predict
   that tangent-only invalidity returns a NaN value/score pair with permanently
   latched invalid status? Does Phase 5 guard endpoint validity-domain
   asymmetry and unmasked diagnostics rather than search for a finite escaped
   scalar?

4. **F4/H3A reset asymmetry.** Is the covariance-gap symmetrization difference
   explicitly scoped as validity-only, included in the first-boundary list, and
   excluded from the explanation of the observed finite particle mismatch?

5. **F5 instrumentation.** Are interior tensor captures restricted to eager
   deterministic mode, with graph/XLA comparisons limited to endpoint scalars
   and final particle clouds? Would the proposed probes avoid changing fusion
   or arithmetic before the boundary under test?

6. **F6 tangent audit.** Does the revised plan accurately record that the local
   Austria direct derivative terms have been checked by derivation while still
   requiring independent composed-transition, reset, and whole-program checks?

7. **F7 test economy.** Are the upstream reset/transition checks correctly
   labeled confirmatory rather than discovery, and is the injected tangent
   failure a fail-closed regression guard with the expected NaN outcome?

8. **Shared-primal repair architecture.** Does the plan require one primal
   stage core whose value and JVP routes share the exact primal computation,
   rather than repairing the duplicated value/JVP functions and then creating
   another pair of maintained primal implementations? Does it separately
   require a new batch-native dual-cap route before any dual-cap NeuTra claim?

9. **Execution order.** Is the order `R0 -> eager Phase 1/2 -> H1 -> H2/H3A/H3
   -> fail-closed H4/H7 -> H5 -> arithmetic replication -> fresh tuning`
   logically valid? Is any phase still using a proxy as a promotion criterion,
   an undocumented tolerance, or a stale tuning artifact?

10. **Remaining blocker.** Is there any material missing hypothesis or
    confound that must be added before Phase 0? If none, state explicitly that
    Phase 0 source/evidence freeze may begin while Austria remains blocked from
    NeuTra and the scientific/default targets remain unchanged.

## Required Review Output

Return a compact but evidence-grounded note containing:

- `VERDICT: AGREE` or `VERDICT: REVISE` first;
- findings ordered by severity;
- exact revised-plan section anchors for every finding;
- any question above that is `not checked` rather than inferred; and
- a one-paragraph execution boundary stating what the verdict does and does
  not authorize.

Do not treat the earlier numerical observations (`5.6596` endpoint gap, about
`24` first-step zero-tangent cloud difference, or about `1.2e5` condition
number) as independently reproduced in this second review. They are preserved
diagnostics; this review is about plan validity and source-claim discipline.
