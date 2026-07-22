# Phase 7 Subplan: Contract E LaTeX Reconciliation

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Status: `CLOSED_RESULT_WRITTEN`

## Phase Objective

Reconcile `docs/chapters/ch32c_entropic_ot_sinkhorn.tex` and
`docs/chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex` with the Phase 1
normative Contract E--Chol mathematics and the checked Phase 3-5 code. Correct
the filtering time order, row quotient, population moments, fixed residual
design, fixed-ridge Cholesky chart, complete direct-plus-transport pullback,
active/inactive reset branches, and same-scalar versus Kalman-oracle roles.

## Entry Conditions

- Phases 0-6 are closed at their narrow engineering gates.
- Contract E--Chol is the only canonical-eligible reset semantics.
- Raw-barycentric routes are historical diagnostics only and cannot be used as
  documentation authority or fallback.
- The production v2 factory remains empty and no admission claim is allowed.
- The continuation clock remains `2026-07-14T01:32:19+08:00` through
  approximately `2026-07-14T09:32:19+08:00`.

## Authority And Required Artifacts

Authority order:

1. `docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase1-normative-mathematics-spec-2026-07-13.md`;
2. checked owned code in `ledh_contract_e_reset_tf.py`,
   `ledh_contract_e_streaming_tf.py`, and
   `ledh_contract_e_canonical_lgssm_tf.py`;
3. Phase 3-6 result artifacts; and
4. existing LaTeX only as text to audit, never as authority over the above.

Required outputs are the two corrected chapters, a code-anchor/label audit, a
successful `docs/main.tex` build log, a Phase 7 result, and a Phase 8 subplan or
an explicit margin-decision blocker.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Do the two chapters describe the same finite Contract E program and total derivative as the normative spec and checked code? |
| Comparator | Phase 1 equations plus exact Phase 3-5 implementation anchors |
| Primary criterion | Every required semantic item has an equation and code anchor; no contradictory historical/canonical claim remains; full monograph builds |
| Hard vetoes | Missing direct moment/weight term, transported-only gradient called total, unnormalized row output called a cloud, adaptive ridge called canonical, raw fallback/default, historical v1 admission, broken LaTeX/reference |
| Explanatory only | Prose organization and notation preferences |
| Artifact | Corrected chapters, anchor audit, build log, Phase 7 result |
| Not concluded | Numerical adequacy, Kalman equivalence, nonlinear validity, admission, HMC, leaderboard, or release readiness |

## Research Intent Ledger

| Field | Binding intent |
| --- | --- |
| Main question | Is the published mathematical description equal to the implemented canonical target? |
| Expected failure | Existing text omits row quotient or direct reset dependence and overstates raw paths |
| Promotion criterion | Documentation consistency only |
| Promotion veto | Any mathematical contradiction or broken build |
| Continuation veto | Correct text would require changing the scientific target or code rather than documenting it |
| Repair trigger | Equation/anchor/build audit failure |
| Must not conclude | Any Phase 8/9 scientific result |

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early check | Status |
| --- | --- | --- | --- | --- | --- |
| Population covariance (`1/N`) | Phase 1 spec/code | Same finite reset program | Sample covariance silently changes target | Equation/code anchor | Required |
| Fixed residual design | Route identity/spec | Same differentiable finite scalar | Fresh randomness changes program | State explicitly | Required |
| Fixed realized ridge | Phase 1 policy | Differentiable same chart | Adaptive escalation changes branches | Search for adaptive wording | Required |
| Total pullback | Owner policy/spec | Contract E depends on source moments and transport | Transport-only partial called total | Term-by-term equation | Required |
| Full monograph build | Existing repo workflow | Catches label/citation/syntax drift | Chapter-only compile misses integration | `latexmk` on `main.tex` | Required |

## Skeptical Plan Audit

Decision: `PASS`.

- The baseline is the normative specification and checked code, not stale
  chapter prose.
- No numerical proxy or threshold is used as documentation promotion evidence.
- The phase cannot repair a code/math mismatch by changing prose silently; such
  a mismatch is a continuation veto and must be recorded.
- The full build answers integration correctness; targeted searches and the
  anchor table answer semantic correctness.
- Phase 8's equivalence margin remains deliberately unresolved and cannot be
  chosen here without scientific justification.

## Required Checks And Review

1. Diff the two chapters against every Phase 1 discrepancy item.
2. Add explicit equations for `Y=Q/M`, weighted source moments, equal-weight
   transported moments, Contract E affine reset, and
   `G_X=G_X^moments+G_X^transport`,
   `G_w=G_w^moments+G_w^transport` including residual/ridge dependence.
3. State transition-first filter order, fixed prepared randomness, active reset
   and inactive carry branches, and same finite value/gradient program.
4. Mark compact/manual raw routes and all v1 artifacts historical-only.
5. Audit labels, references, symbols, code paths, and forbidden claims.
6. Build with
   `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` from `docs/`.
7. Run scoped `git diff --check` and targeted contradiction searches.
8. Write the result and draft Phase 8 statistical-design subplan. Do not launch
   Phase 8 unless its equivalence criterion is scientifically justified and
   frozen or the owner resolves the explicit decision.

## Forbidden Claims And Actions

- No code, schema, route registration, numerical threshold, or experiment
  change in this documentation phase.
- No claim that Contract E is already Kalman-equivalent, nonlinear-valid,
  HMC-ready, leaderboard-complete, or releasable.
- No adaptive ridge, fresh residual randomness, raw reset fallback, or
  transported-only derivative may be described as canonical.
- Do not silently reuse the historical `1%` margin or treat
  `0.05*sqrt(p)` as a Kalman-equivalence criterion.

## Exact Handoff Conditions

Phase 8 planning may begin only if both chapters, the Phase 1 spec, and checked
code agree; the full monograph builds; every required code anchor resolves; no
forbidden canonical/raw ambiguity remains; and the result preserves every
scientific blocker. Phase 8 execution additionally requires a defensible frozen
equivalence design or an explicit owner decision.

## Stop Conditions

Stop if documentation reconciliation reveals an actual target/code mismatch,
requires a scientific choice, cannot build after focused repairs, or the
campaign clock expires. A fixable LaTeX or stale-prose error is a repair trigger,
not a reason to abandon the phase.
