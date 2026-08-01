# Phase 7 Code-Anchor And Contradiction Audit

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

## Authority

The audit compared the two edited chapters with the Phase 1 normative
mathematics and the checked owned TensorFlow modules. Existing chapter prose was
not treated as authority when it disagreed with those sources.

## Semantic Crosswalk

| Required semantic item | Chapter anchor | Owned-code anchor | Verdict |
| --- | --- | --- | --- |
| Transition and flow precede the observed-data increment; reset follows it | `ch32c2_ledh_pfpf_ot_custom_gradient.tex:74`; `ch32c_entropic_ot_sinkhorn.tex:1448` | `ledh_contract_e_canonical_lgssm_tf.py:690`, `:724`, `:730`, `:749` | Agree |
| Active reset carries uniform weights; inactive branch carries normalized weights | `ch32c_entropic_ot_sinkhorn.tex:1452` | `ledh_contract_e_canonical_lgssm_tf.py:760-766`; tangent branch `:944-961` | Agree |
| Population covariance denominator is `N` | `ch32c_entropic_ot_sinkhorn.tex:1333-1349`; `ch32c2_ledh_pfpf_ot_custom_gradient.tex:140-145` | `ledh_contract_e_reset_tf.py:18-32` | Agree |
| Executed transport output is `Y=Q/M`, with no denominator floor and both quotient cotangents | `ch32c2_ledh_pfpf_ot_custom_gradient.tex:117-133`; companion anchor `ch32c_entropic_ot_sinkhorn.tex:1433-1446` | streaming quotient forward/JVP/VJP cores in `ledh_contract_e_streaming_tf.py`; composition at `:338-368` | Agree |
| Contract E uses a fixed realized residual design and fixed prepared ridge | `ch32c_entropic_ot_sinkhorn.tex:1317-1344`; `ch32c2_ledh_pfpf_ot_custom_gradient.tex:135-165` | reset arguments at `ledh_contract_e_reset_tf.py:452-467`; zero tangents at `ledh_contract_e_canonical_lgssm_tf.py:827-830`, `:934-935` | Agree |
| Cholesky affine reset is in row-vector orientation and uses a solve | `ch32c_entropic_ot_sinkhorn.tex:1351-1376`; `ch32c2_ledh_pfpf_ot_custom_gradient.tex:146-165` | reset forward core `ledh_contract_e_reset_tf.py:44-101` | Agree |
| Total source pullback includes direct moment and streaming-transport terms | `ch32c2_ledh_pfpf_ot_custom_gradient.tex:166-178` | `ledh_contract_e_streaming_tf.py:349-381` | Agree |
| Probability-coordinate moment adjoint is converted before normalized-log-weight composition | `ch32c2_ledh_pfpf_ot_custom_gradient.tex:179-191` | `ledh_contract_e_streaming_tf.py:369-381` | Agree |
| Canonical code packages separate primal and manual-JVP traversals without claiming one-trace generation | `ch32c2_ledh_pfpf_ot_custom_gradient.tex:646-657` | `_canonical_primal_core` at `ledh_contract_e_canonical_lgssm_tf.py:658`; `_canonical_manual_jvp_core` at `:804`; packaging at `:971-993` | Agree |
| Raw compact/manual routes are historical-only and never a fallback | `ch32c2_ledh_pfpf_ot_custom_gradient.tex:107-112`, `:698-706` | Phase 6 fail-closed result and inventory | Agree |

Line anchors are audit-time anchors for the current worktree. The source hashes
in the manifest bind the exact audited files.

## Contradiction Search

- Canonical candidate-dependent ridge escalation appears only in explicit
  rejection language.
- The canonical row quotient and absence of a row-mass floor are explicit.
- Both direct and transport cloud/weight terms are explicit.
- Probability and normalized-log-weight coordinates are not mixed.
- Raw barycentric routes are described only as historical diagnostics.
- No new Kalman-equivalence, HMC-readiness, admission, leaderboard-completeness,
  default-readiness, nonlinear-validity, or release claim appears.

One notation ambiguity found during the audit was repaired before closeout: the
Contract E equations now redefine source and transported moments locally in
row-vector orientation. The reset API's OT-settings bundle was renamed from
`rho` to `eta_OT`, reserving `rho=1` for the frozen Contract E residual amplitude.

## Labels And References

All seven labels added by Phase 7 occur exactly once. The full build reports
four duplicate labels and eleven unresolved citation occurrences. The four
duplicate label names already occur more than once in `HEAD`; no Phase 7 label
is among them. The eight unresolved bibliography keys are absent from the
`HEAD` bibliography and Phase 7 added no citation. These warnings are preserved
as pre-existing monograph debt, not claimed as repaired or introduced by this
phase.

Verdict: `PASS`.
