# Fable Focused Re-audit Handoff: Revised Direct-Factor SR-UKF Plan

Date: 2026-08-16
Status: `FOCUSED_REAUDIT_REQUESTED_NO_IMPLEMENTATION_AUTHORITY`
Requestor: BayesFilter / Codex
Reviewer: Fable, using the local MathDevMCP CLI
Revised plan: `docs/plans/bayesfilter_direct_factor_srukf_execution_plan_2026_08_15.md`
Prior reply: `docs/plans/bayesfilter_direct_factor_srukf_fable_audit_reply_to_codex_2026_08_15.md`

## 1. Purpose and boundary

Audit only whether the revised plan closes Fable findings FS-1 through FS-9.
The earlier audit already accepted the untouched architecture and derivations;
do not repeat a repository-wide or full-plan audit. This is read-only review.
Do not edit files, implement the filter, install packages, use the network,
modify MacroFinance, run NeuTra/HMC, change a default, or make a production or
scientific claim.

The first review prompt must follow the repository's one-path rule exactly:

```text
READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the
file itself explicitly asks you to inspect a cited line:
docs/plans/bayesfilter_direct_factor_srukf_fable_focused_reaudit_handoff_2026_08_16.md.
Do not edit, run commands, launch agents, or review the whole repo. Question:
Does the specified focused re-audit adequately test closure of FS-1 through
FS-9? End with VERDICT: AGREE or VERDICT: REVISE.
```

After accepting this handoff's scope, inspect only the exact revised-plan
anchors listed below. Request another exact path or line range if one is needed.

## 2. Disposition of prior findings

### FS-1: gain solve orientation, blocking

Revised-plan anchor: Section 3.5, `Gain and filtered mean`.

The normative route is now

\[
  U S_y'=P_{xy},\qquad U'=S_y^{-1}P_{xy}',
  \qquad K=US_y^{-1},
\]

with

\[
  dU'=S_y^{-1}(dP_{xy}'-dS_yU'),
  \qquad dK=dU S_y^{-1}-K dS_y S_y^{-1}.
\]

The plan identifies \(U=KS_y\) with the downdate matrix \(V\), avoiding a
second solve. Closing tests compare the gain with a dense
\(P_{xy}P_y^{-1}\) oracle and check affine filtered mean and factor, not only
likelihood and score.

Required decision: confirm the triangular-solve orientation and derivative by
direct substitution, dimensions, and the prior counterexample.

### FS-2: rank-one vector update, blocking

Revised-plan anchor: Section 3.7, scalar-pivot recurrence and derivative order.

The normative recurrence is now

\[
  a_{new}=(a_{old}-s u_{old})/c,
  \qquad u_{new}=c u_{old}-s a_{new},
\]

with the equivalent all-old form recorded. The plan explicitly requires value
evaluation in the order \(a_{new}\), then \(u_{new}\), and derivative evaluation
in the order \(da_{new}\), then \(du_{new}\). All scalar pivot derivative
right-hand sides use pre-update pivot values.

Required decision: confirm the value recurrence, equivalence of the two forms,
and derivative evaluation order against the prior counterexample.

### FS-3: route guard boundary, material

Revised-plan anchors: Sections 4.1, 4.2, and 7.

The one-time legacy covariance conversion is isolated in the non-admitted
`bayesfilter/nonlinear/factor_srukf_compat.py`. The closed admitted set is:

```text
bayesfilter/linear/stack_qr_tf.py
bayesfilter/linear/lower_rank_downdate_tf.py
bayesfilter/nonlinear/factor_srukf_tf.py
```

The standalone QR kernel may not import the legacy mixed-purpose
`qr_factor_tf.py`. The route guard uses case-insensitive substring matching for
the exact token list in Section 7, extends
`FORBIDDEN_SRUKF_ROUTE_PATTERNS`, asserts the closed file set, and complements
the lexical check with runtime downdate provenance and an import-boundary test.

Required decision: confirm this is implementable by the existing scanner model
and does not falsely ban the separately named rank-downdate primitive.

### FS-4: SPD versus roundoff attribution, material

Revised-plan anchor: Section 3.7, feasibility and failure attribution.

The plan now states

\[
  P^{(k)}=P_f+\sum_{j>k}v_jv_j'\succeq P_f,
\]

so exact SPD \(P_f\) makes every intermediate target SPD for every column
order. Original runtime failure evidence is retained before an offline
eigenvalue comparator classifies it as `downdate_target_indefinite` or
`downdate_roundoff_or_implementation_suspected`. The comparator is diagnostic
only and never a fallback. Phase 3 includes both failure classes.

Required decision: confirm the lemma and that the classification does not
convert floating-point evidence into an exact mathematical proof.

### FS-5 through FS-9: advisory hardening

Revised-plan anchors: Sections 3.4, 3.7, 4.3, Phase 1 through Phase 5, Section
6, and Section 7.

The plan now includes:

- the dimensionally explicit zero-extended all-additive comparator
  \(\widetilde A_f=A_x^+-KA_y\), its derivative, and its diagnostic-only role;
- centered, per-parameter scaled finite differences at \(h\) and \(h/2\), with
  both actual step sizes retained and frozen before rare-row inspection;
- named failure codes and explicit NaN/Inf rejection for both QR and rank
  downdate primitives;
- relative QR-pivot and downdate-margin conditioning diagnostics while strict
  positivity remains the hard gate; and
- the corrected observation-stack derivative multiplication
  \([\sqrt{w_i^{(c)}}\,d_p\delta y_i]_i\Vert d_pS_r\).

Required decision: confirm each advisory item is mathematically and
operationally well specified.

## 3. Bounded MathDevMCP protocol

MathDevMCP evidence is diagnostic, not a proof certificate. Run from
`/home/ubuntu/workspace/BayesFilter` without network or package mutation. Put
new output under the unique root
`docs/plans/artifacts/direct-factor-srukf-fable-focused-reaudit-20260816/` and
do not overwrite the prior audit artifacts.

Environment check:

```bash
PYTHONPATH=/home/ubuntu/workspace/MathDevMCP/src \
python -m mathdevmcp.cli doctor
```

Record the current commit, dirty-worktree status, and SHA-256 of exactly the
revised plan, this handoff, the prior Fable reply, and prior
`counterexample_checks.txt`. Do not interpret hashes as approval authority.

The symbolic router previously abstained on noncommutative matrix products. Do
not repeat broad symbolic calls that cannot represent the obligations. Use the
prior counterexamples, bounded human algebra, and small independent numerical
checks for only these identities:

1. corrected versus old gain solve on Fable's nonsymmetric lower-triangular
   \(S_y\), including a directional derivative check for \(U,K\);
2. corrected versus old rank-downdate recurrence on Fable's \(L,x\), including
   reconstruction and a centered directional derivative at \(h,h/2\);
3. the zero-extended block comparator identity and its derivative on one fixed
   small SPD fixture; and
4. the partial-covariance feasibility identity after each sequential column.

Use TensorFlow or Python standard-library arithmetic for new checks where
practical. A diagnostic test script under the audit artifact root may use NumPy
because it is explicitly independent reference evidence and cannot affect the
candidate runtime. Record commands, seeds, dtype, step sizes, tolerances,
elapsed times, raw output, and any abstention or tool limitation.

## 4. Acceptance questions

Answer each with `SUPPORTED`, `REVISE`, `COUNTEREXAMPLE`, or `ABSTAIN`:

1. Does Section 3.5 now compute exactly
   \(P_{xy}(S_yS_y')^{-1}\), and is its derivative consistent with that value
   program?
2. Does Section 3.7 now implement a valid lower-factor rank-one downdate, with
   unambiguous old/new value and derivative ordering?
3. Is \(V=U\) correct, including \(dV=dU\), so the gain solve feeds the
   downdate directly?
4. Is the explicit admitted file boundary mechanically enforceable without
   banning a legitimate compatibility conversion outside it?
5. Does the feasibility lemma justify the exact-arithmetic statement for every
   column order, and is floating-point attribution kept diagnostic?
6. Is the block comparator dimensionally valid when observation-noise columns
   are present, and is its derivative complete?
7. Are finite-difference, nonfinite-input, conditioning, failure-code, unit,
   and integration-test obligations sufficient to close FS-6 through FS-9?
8. Did any revision introduce a new hidden inverse, covariance refactorization,
   runtime eigendecomposition/SVD, fallback, nonlinear orientation-equivalence
   claim, or implementation/production authority?

## 5. Required artifact and verdict

Return one concise report at:

```text
docs/plans/artifacts/direct-factor-srukf-fable-focused-reaudit-20260816/
fable_focused_reaudit_report.md
```

The report must map FS-1 through FS-9 to the revised plan anchors, give each
acceptance answer, cite any numerical evidence, record tool limitations, and
end exactly with:

```text
VERDICT: AGREE
```

or

```text
VERDICT: REVISE
```

`AGREE` authorizes only the bounded implementation phases in the revised plan.
It does not approve implementation completion, MacroFinance integration,
NeuTra, HMC/NUTS, a backend switch, production use, or scientific claims.
