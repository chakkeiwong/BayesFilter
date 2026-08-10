# Squared-TT retained-block derivation and certificate execution plan

- **Date:** 2026-08-08
- **Target:** `docs/fable-rewrite/monograph/`
- **Scope:** repair and certify only the retained-prefix squared-TT value,
  derivative, and saved-evaluator formulas in Chapters 36b and 37.
- **Authority:** the user's 2026-08-08 instruction to plan, review, and execute
  the squared-TT handoff.

## Research intent ledger

| Field | Declaration |
|---|---|
| Main question | For a TT square-root field whose first `m` coordinates are retained, do the displayed right contractions equal direct integration of the same squared field, and does their differentiated recursion equal direct differentiation? |
| Mechanism under test | Retained-prefix product `G_m=H_1\cdots H_m`, right mass contraction `M_{>m}`, and their product-rule derivatives. |
| Expected failure mode | A scalar-only formula stops at `M_{>1}` and is silently generalized to `m>1`, integrating out retained coordinates `2,...,m`. |
| Promotion criterion | Exact equality for both `m=1,D=2` and `m=2,D=4` value and directional-derivative certificates, consistent `Q_t,\dot Q_t` saved evaluators, successful MathDevMCP audits with no reported mismatch, and a clean LaTeX build. |
| Promotion veto | Any exact-arithmetic inequality, MathDevMCP `mismatch`/refutation, dimension inconsistency, hidden change of measure, unresolved left/right mismatch, or LaTeX build failure. |
| Continuation veto | The target cannot be represented by a retained-prefix contraction under the declared reference measure, or the deterministic certificate is not independent of the contraction being tested. |
| Repair trigger | MathDevMCP abstention, missing assumptions, notation drift, or a certificate mismatch with a localizable algebra/implementation cause. These trigger repair and re-audit, not automatic abandonment. |
| Explanatory diagnostics | MathDevMCP `unverified`/`inconclusive` substatuses, source excerpts, and decimal renderings of exact values. |
| Nonclaims | This work does not certify the TT fitting algorithm, adaptive branch selection, implementation/source parity, posterior correctness, HMC readiness, or the entire monograph. |

## Evidence contract

The claimed target is the retained-reference-measure marginal

\[
  a_t(z_{1:m})=
  \int e^{-c_t}\{\phi_t(z_{1:D})^2+\tau_t\lambda_t(z_{1:D})\}
  \,\mathrm d\mu_{m+1:D}.
\]

The comparator is direct polynomial expansion and exact integration under the
same product reference measure. The candidate is the right-contraction formula

\[
  e^{-c_t}G_mM_{>m}G_m^\top
  +e^{-c_t}\tau_t\lambda_{t,\mathrm{ret}}.
\]

The primary pass criterion is equality in exact rational arithmetic, not a
floating-point tolerance. The derivative comparator is exact integration of
`2 phi dot(phi)` on a fixed branch. MathDevMCP CLI audits every displayed
derivation added or materially changed, but its abstentions are diagnostic only
and cannot replace the direct certificate.

Artifacts:

- exact checker:
  `docs/plans/artifacts/fable-rewrite-squared-tt-certificate-20260808/check_squared_tt_retained_block_certificate.py`
- exact checker output:
  `docs/plans/artifacts/fable-rewrite-squared-tt-certificate-20260808/certificate.json`
- MathDevMCP ledger:
  `docs/plans/bayesfilter-fable-rewrite-squared-tt-mathdevmcp-audit-2026-08-08.md`
- result note:
  `docs/plans/bayesfilter-fable-rewrite-squared-tt-block-certificate-result-2026-08-08.md`
- documentation amendment handoff:
  `docs/plans/bayesfilter-fable-rewrite-squared-tt-documentation-agent-amendment-handoff-2026-08-08.md`

## Default and assumption audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|---|
| Retained coordinates form the prefix `1:m` | Current rewrite ordering | Matches `(x_t,x_{t-1})` and its concatenated vector form | Non-prefix ordering invalidates the simple right contraction | Inspect coordinate order and stop index | Reviewed branch assumption |
| Product reference measure | Current basis-mass construction | Makes `B_j` and repeated right integration well-defined | Mixing Lebesgue/reference measures changes every mass | State measure on every certificate | Reviewed branch assumption |
| Bases, mass matrices, domains, maps, `c_t`, `tau_t`, and defensive density are frozen | Chapters 36b/37 fixed-branch convention | Defines the same-branch directional derivative being claimed | Omitted product-rule terms if any are parameter-dependent | Explicit zero-derivative ledger | Reviewed branch assumption |
| Basis `b_j=(1,z_j)` on `[-1,1]` for certificates | Minimal exact test choice | Gives exact rational mass matrix and exposes cross-core products | Too-simple rank/basis could hide index errors | Use nontrivial TT ranks and both scalar/vector retained cases | Diagnostic fixture |
| Uniform normalized defensive density | Existing concrete branch suggestion | Its retained marginal and derivative are exact | Unnormalized convention changes `tau_t` contribution | Check full and retained masses exactly | Diagnostic fixture |
| Exact Python `Fraction` arithmetic | Repository standard-library preference | Independent, deterministic, and free of tolerance choices | A checker could accidentally reimplement only the candidate formula | Direct path expands multivariate polynomials; candidate path uses contractions | Reviewed diagnostic default |
| MathDevMCP `audit-derivation-v2-label`, typed audit, and SymPy scalar shadows | Monograph MathDevMCP workflow | Covers label context, shapes, and bounded algebra | Tool abstention could be misreported as proof | Preserve raw statuses and use direct certificate as primary evidence | Diagnostic only |

## Skeptical pre-execution audit

1. **Wrong baseline found and repaired:** the handoff asks for a two-coordinate
   scalar example while also claiming a vector analogue. That example cannot
   detect integration of retained coordinates `2,...,m`. The plan therefore
   requires a genuinely vector-retained `m=2,D=4` case.
2. **Proxy promotion blocked:** MathDevMCP status is not the promotion
   criterion. Exact direct-versus-contraction equality is primary.
3. **Measure fairness:** direct and contraction paths must use the same declared
   product reference measure and the same defensive-density normalization.
4. **Hidden derivative assumptions surfaced:** all frozen objects receive an
   explicit zero-derivative declaration. If the documentation instead elects a
   parameter-dependent object, its product-rule term must be added before the
   certificate can pass.
5. **Artifact adequacy:** the direct path expands TT entries into multivariate
   polynomials before integration; it must not call the right-contraction code.
6. **Stop conditions:** do not remove the existing nonclaim if a veto remains.
   An `unverified` MathDevMCP result without mismatch is recorded as a tool
   boundary and requires the exact/manual evidence to carry the claim.
7. **Scope control:** no implementation, fitting, or unrelated monograph files
   are changed unless the derivation exposes a direct dependency.

**Pre-execution verdict:** pass after the mandatory vector-block amendment. The
revised plan's artifacts answer the stated question and its vetoes distinguish a
failed candidate formula from a tool limitation.

## Execution steps

1. Write the documentation-agent amendment explaining why the scalar-only
   handoff must be strengthened.
2. Add the exact standard-library certificate with independent direct-polynomial
   and right-contraction paths for `m=1,D=2` and `m=2,D=4`.
3. Run the certificate and stop on any exact inequality.
4. Repair Chapter 36b so the general formula stops the right recursion at
   `m+1`, leaves `G_m=H_1\cdots H_m` explicit, and differentiates `G_m`,
   `M_{>m}`, and the retained numerator on the same frozen branch.
5. Present the scalar formula explicitly as the `m=1,D=2` specialization.
6. Repair Chapter 37 so `Q_t` and `\dot Q_t` are derived from retained-prefix
   coefficient matrices and include the fixed defensive retained marginal.
7. Add a compact reader-facing certificate table using the exact checker output.
8. Index the final LaTeX tree and run MathDevMCP label and typed-obligation
   audits on every added or materially changed derivation label. Run SymPy
   `check-proof-obligation` checks for all scalar product/quotient shadows.
9. If an audit reports a mismatch, repair and repeat. Record abstentions and
   limitations without upgrading them to proof.
10. Build `docs/fable-rewrite/monograph/main.tex` with `latexmk` and inspect
    undefined references, undefined citations, and affected-label duplication.
11. Write the final MathDevMCP ledger and result note. Update the persistent
    audit ledger only if every promotion criterion passes; otherwise retain the
    blocker and state the smallest remaining gap.

## Plan-change rule

Any material change to the mathematical target, retained-coordinate ordering,
frozen-object ledger, certificate cases, or release criteria requires both an
update to this plan and an amendment in the documentation-agent handoff. A local
formatting or command-path correction that leaves the evidence contract
unchanged is recorded in the result note but is not a plan amendment.
