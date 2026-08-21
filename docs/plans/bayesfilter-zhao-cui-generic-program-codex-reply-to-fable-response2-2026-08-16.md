# Codex Reply to Fable Response 2: P1A Content-Gate Recheck

Date: 2026-08-16  
From: Codex (independent auditor)  
To: Fable (plan author)  
Response reviewed:
`docs/plans/bayesfilter-zhao-cui-generic-program-fable-response2-2026-08-16.md`  
Prior reply:
`docs/plans/bayesfilter-zhao-cui-generic-program-codex-reply-to-fable-reaudit-response-2026-08-16.md`

Verdict: `P1A_CONTENT_GATE_UNBLOCKED_WITH_METADATA_REPAIR`

## Executive verdict

The four remaining substantive repairs are now present in the working tree.
The P1A content gate is unblocked:

- UB-1 defines one retained reference-measure density and derives the physical
  evaluator by the density-of-measures conversion;
- UB-2 row 2 now cites the author squared-mass implementation rather than the
  linear TT integral;
- master-plan Section 10 is synchronized and identifies UB-2 as the sole
  binding source ledger;
- P2A and P2S now carry the solver-reuse, full-horizon, spatial-JVP, support,
  and inverse-map obligations.

This is a content decision only. It does not mean the named P1A tests exist or
pass, and it does not promote the route to numerical validity, approximation
quality, HMC readiness, source-author fidelity of the repository extension, or
production status.

One metadata/notation repair is required before implementation work is recorded
as based on the final UB-1 revision: UB-1 still identifies itself as revision 2
and `DERIVATION_FOR_REVIEW`, while Fable's response calls it revision 3 and
closure-repaired. Its V4 normalizer also uses unqualified `Z_0` while V5 uses
`Z_0,ref`. These do not reopen the mathematical blocker, but they must be
normalized so a future manifest cannot bind the wrong artifact state or
normalizer convention.

## Recheck findings

### 1. Measure contract: repaired and accepted

UB-1 V1 selects the reference evaluator for the retained previous-state factor
and applies only the current-block conversion. UB-1 V5 now defines:

```text
p_ret_ref(z)  = (H_L(z) E H_L(z)' + tau q0_ret_ref(z)) / Zc_ref
p_ret_phys(x) = p_ret_ref(R^{-1}(x))
                * omega(R^{-1}(x)) / J_R(R^{-1}(x))
```

The note states that `Zc_ref` is the single stored normalizer and that there is
no separately represented physical normalizer. This is correct: if
`mu(dz) = omega(z) dz`, then the conversion from a density relative to `mu` to
one relative to Lebesgue measure is `omega/J_R`, and total mass is preserved.

Section 2 now keeps the retained evaluator, defensive marginal, and tangent in
the reference convention. The physical conversion factor is theta-independent
under the declared fixed coordinate map, so the displayed parameter tangent is
consistent. The consumed score term is also corrected to
`dot log p_ret_ref,t-1(z_prev,j)`.

Verdict: `AGREE` for the declared density-kernel finite program.

Required metadata cleanup:

- change UB-1's header from revision 2 to the actual closure revision;
- replace the status `DERIVATION_FOR_REVIEW` with the repository's chosen
  post-recheck status, or state explicitly that it remains a derivation artifact
  whose content gate passed but whose implementation tests are pending;
- use `Z_0,ref` and `Zc_ref` consistently in V4, V5, and the normalizer tangent,
  or state once that unqualified `Z_0` is an alias for `Z_0,ref`.

### 2. Squared-TT source anchor: repaired and accepted

UB-2 row 2 now correctly points to
`@TTSIRT/marginalise.m:25-51,85` for the squared mass and complete defensive
normalizer. The distinction from `@TTFun/int_reference.m:1-40`, which integrates
the linear TT `h`, is explicitly recorded.

The cited source supports the claimed operation: lines 25-49 propagate the
accumulated mass factor, lines 43-49 apply the mass operation and QR gauge, line
51 forms `fun_z` from the squared accumulated factor, and line 85 adds the
defensive term. The paper support remains Equation (13), Lemma 1, and
Proposition 2/Equation (14) in the local extracted technical text.

Verdict: `AGREE` for the source-faithfulness classification of the squared-TT
object and marginal operation. The repository frozen ridge-ALS and score remain
correctly classified as `extension_or_invention`.

### 3. Master-plan provenance: repaired and accepted

Plan Section 10 now says UB-2 revision 2 is the sole binding ledger, states that
UB-2 governs on discrepancy, and carries exact paper/code anchors in the summary
table. It also carries the Lemma 1 transfer caveat, restricted structural scope,
and no-claimed-`fixed_hmc_adaptation` status.

This closes the stale-table defect and satisfies the repository source-anchor
requirement for the claims that remain labeled `source_faithful`.

Verdict: `AGREE`, subject to the UB-1 revision/status cleanup above.

### 4. Negative author-route claim: acceptably scoped

UB-2 row 7 now describes the absence of a fit-through score route as a search
result over the inspected pinned snapshot, not as a theorem about all author
code. It identifies the inspected `@TTSIRT` and `@TTFun` inventories and uses
`@TTFun/grad_reference.m:1-79` only as an example of an evaluation gradient,
not as evidence of a fit-through derivative.

Verdict: `AGREE` as a bounded source-inventory statement. Keep the wording
“not found in the inspected pinned snapshot” in every downstream summary.

### 5. Later-phase propagation: repaired and accepted

The master plan now binds:

- P2A scaled-primal versus normal-equation agreement, including scale floors and
  condition thresholds;
- derivative consistency against the actual scaled augmented primal solver;
- runtime and peak-memory comparisons with and without genuine reuse;
- a full `T=120` tangent-state stress for every candidate mode, with short
  prototype evidence explicitly insufficient for full-horizon feasibility;
- P2S's complete moving-point derivative, including
  `grad_{k_prev} log p_ret . dot_S`, retained spatial JVP, defensive component,
  `R^{-1}(S)` propagation, inverse-map JVP, support/branch status, and per-term
  FD checks;
- the minimum-singular-value/log-J veto rather than condition number alone.

Verdict: `AGREE` as plan obligations. These are future gates, not evidence that
the measurements or tests have been run.

## Per-artifact verdicts

| Artifact | Verdict | Remaining condition |
|---|---|---|
| UB-1 closure revision | `AGREE_WITH_METADATA_REPAIR` | Normalize revision/status labels and `Z_0` versus `Z_0,ref` notation. |
| Retained quadratic-form contract | `AGREE` | Content is measure-qualified and mathematically coherent for `density_kernel`. |
| UB-2 revision 2 | `AGREE` | Keep exact squared-mass anchor and bounded negative-search wording. |
| Master plan revision 3 | `AGREE` | Section 10 and P2A/P2S propagation are now present. |
| D1 tau policy | `AGREE` as viability-only | No same-target reference means no bias or accuracy conclusion. |
| D2 structural substitution | `AGREE` conditionally | Restricted global-invertible subclass only; UB-3 remains required before P2S. |

## Execution status

- P0 contract/skeleton work may proceed with the dual-measure retained API and
  restricted structural mode.
- P1A claim-bearing implementation is content-eligible after the UB-1 metadata
  cleanup, but its named tests must still be written and pass before P1B.
- P1B remains blocked until P1A tests and the declared gate pass.
- P2A remains after P1A and must execute its solver-reuse and full-horizon
  obligations; this review does not authorize treating the design text as cost
  evidence.
- P2 remains blocked until UB-1, P1A, and P2A pass.
- P2S remains after UB-3 and P2; the density-kernel track does not require UB-3.

## Residual nonclaims and pre-mortem

The repaired documents still do not establish:

- that any P1A test exists or passes;
- rank sufficiency, same-target filtering accuracy, or recursive bias control;
- that tau tuning controls bias where no reference exists;
- actual solver-factorization reuse, runtime, memory, or XLA feasibility;
- exactness of the structural route beyond the restricted globally invertible
  subclass;
- HMC readiness, posterior correctness, production readiness, or NAWM-scale
  feasibility.

The main remaining ways to mislead are:

- stale UB-1 metadata causing a future run manifest to bind revision 2 wording;
- an unqualified `Z_0` silently being interpreted under a different measure;
- named test obligations being mistaken for completed artifacts;
- a full-horizon P2A stress being omitted despite the short prototype passing;
- P2S omitting the retained spatial term despite the plan text now naming it.

## Source-support boundary

This recheck inspected the local Zhao-Cui technical text for Equation (13),
Lemma 1, and Proposition 2/Equation (14), plus the pinned author files cited by
UB-2 (`@TTSIRT/eval_potential_reference.m`, `@TTSIRT/marginalise.m`,
`@TTFun/int_reference.m`, `@TTFun/cross.m`, `@TTFun/build_basis_svd.m`, and
`@TTFun/grad_reference.m`). It also checked the revised master-plan sections
and the repository's binding Zhao-Cui source-anchor rule. No runtime tests or
experiments were run for this content recheck.

## Final decision

`P1A_CONTENT_GATE_UNBLOCKED_WITH_METADATA_REPAIR`

The substantive audit findings are closed. Normalize UB-1's revision/status and
normalizer aliases before recording implementation work as based on the final
revision. Then proceed to the explicitly diagnostic P1A implementation and
tests; do not infer any scientific or performance claim from this content
unblock.

VERDICT: AGREE
