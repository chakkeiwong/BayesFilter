# Source-Faithfulness Route Ledger: Generic Squared-TT Filtering Program (UB-2)

Date: 2026-08-15 (revision 2, 2026-08-16: exact author-code anchors added
per focused re-audit Finding 7)
Status: `EXACT_ANCHORS_RECORDED`
Program plan:
`docs/plans/bayesfilter-zhao-cui-generic-highdim-analytic-score-program-plan-2026-08-15.md`
Required by: Codex audit finding `BLOCK_SOURCE_UNGROUNDED` and revision item 5.

Primary source: Y. Zhao and T. Cui, "Tensor-Train Methods for Sequential
State and Parameter Learning in State-Space Models," JMLR 25 (2024).
Local copies:
`.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.pdf`
(+ extracted text). Author code snapshot:
`third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/`
(pinned, Octave-compatible; paths below relative to that root).

## Operation-level classification

| # | Operation | Classification | Paper anchor | Author-code anchor (exact) | Notes |
|---|---|---|---|---|---|
| 1 | Squared-TT nonnegative density `p = (h^2 + tau q0)/Z` | `source_faithful` | Eq. (13), Lemma 1 (squared-Rosenblatt construction) | `@TTSIRT/eval_potential_reference.m:21,33` (`log(obj.z) - log(fx + obj.tau) + mlogw`: squared amplitude + tau, complete normalizer, reference log-weight) | Structural nonnegativity by squaring; defensive tau in the evaluated potential |
| 2 | Exact Gram-chain normalizer of `h^2` | `source_faithful` | Prop. 2 / Eq. (14) and the mass-matrix construction of Sec. 3 (extracted text `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt:549-626`) | `@TTSIRT/marginalise.m:25-51` (accumulated squared-mass factor propagated core-by-core: `mass_r` + QR gauge lines 43-49; `obj.fun_z = sum(sum(Ligeqk.^2))` line 51) and `:85` (complete defensive mass) | Repo implementation `bayesfilter/highdim/squared_tt.py:164-175`; audited correct. (`@TTFun/int_reference.m:1-40` is the LINEAR TT integral of `h` — a different operation, retained here only as that separate reference) |
| 3 | End-block marginalization as quadratic form / sum of squares | `source_faithful` | Proposition 2, Eq. (14) | `@TTSIRT/marginalise.m:25-85` (retains rk / rk-1 functions per direction, lines 35-37 and 63-65 comments; accumulated mass line 85) | The retained object keeps multiple functions + accumulated mass; scalar-square closure is NOT source-supported (audit F1) |
| 4 | Adaptive TT construction (TT-cross / interpolation / SVD ranks) | `source_faithful` but **excluded from runtime** | Sec. 3 algorithmic description | `@TTFun/cross.m:1-60` (random init ranks lines 20-28, kick-rank enrichment lines 38-53); SVD rank truncation `@TTFun/build_basis_svd.m:31` (`TTFun.local_truncate`) | Excluded by fixed-variant policy V1: data-dependent discrete choices (pivots, kick ranks, truncation) are non-differentiable |
| 5 | Frozen global weighted ridge ALS on repository-declared rows/designs | `extension_or_invention` | none (paper route is TT-cross, not weighted ridge ALS) | n/a (no author counterpart; nearest author construction is the cross/AMEN basis build at `@TTFun/build_basis_amen.m` / `build_basis_svd.m`, which this program does NOT copy) | Repository construction; must never be described as the author algorithm. "Zhao-Cui" is family provenance only |
| 6 | Freezing an author-matching discrete branch | `fixed_hmc_adaptation` | per-operation | per-operation; NONE CLAIMED in v1 (no current engine operation freezes an author-specific discrete branch; row retained for future use) | Only claimable with a specific paper+code anchor pair per operation |
| 7 | Ordered total-derivative tangent replay of the frozen ALS program (analytical score) | `extension_or_invention` | none (paper provides no HMC score route) | negative claim scoped to the inspected pinned snapshot: no fit-through score route FOUND in `deep-tensor.dev/src/` (`@TTSIRT`, `@TTFun` inventories inspected; `@TTFun/grad_reference.m:1-79` is an evaluation-gradient example, not a fit-through derivative). Repo donor: `bayesfilter/highdim/zhao_cui_moment_teacher_als.py:403-475`; FD-gated per Method A rules | Not a theorem about all author code — a recorded search result over the pinned snapshot. Solver-reuse caveat per re-audit Finding 2 recorded in UB-1 Sec. 3.2 |
| 8 | RetainedQuadraticForm runtime type (prefix cores + suffix Gram + tangent state + dual measure evaluators) | `extension_or_invention`, structure-anchored to #3 | Prop. 2 / Eq. (14) for the mathematical object | `@TTSIRT/marginalise.m:25-85` for structure | The *type, tangent state, and measure API* are repository constructions; the *mathematical identity* is source-faithful |
| 9 | Batched all-parameter tangent stack / multi-RHS organization | `extension_or_invention` | none | n/a | Pattern precedent: repository UKF/SGQF/LEDH analytic scores, not Zhao-Cui |
| 10 | Structural substitution mode (Dirac-integrated deterministic completion) | `extension_or_invention` (relative to Zhao-Cui) | n/a — grounded in Ch18b (`docs/chapters/ch18b_structural_deterministic_dynamics.tex`, structural split, pushforward assumptions incl. the explicit no-invertibility-required general identity near lines 1616-1628, and validation gates) | n/a | v1 implements the RESTRICTED globally-invertible-completion subclass (plan 3.6, V13); the chapter's general pushforward (no invertibility required) is broader than v1's route — recorded per re-audit Finding 5 |
| 11 | Defensive-mass complete normalizer in likelihood increments | `source_faithful` (object) / repo-enforced (policy) | Eq. (13) normalization | `@TTSIRT/marginalise.m:85` (`obj.z = obj.fun_z + obj.tau`) and `@TTSIRT/eval_potential_reference.m:21` (`log(obj.z)`) | Audit F4: increments must use `log(Z_h + tau Z_0)`; tau per-scope tuned (owner decision D1) |
| 12 | Physical<->reference measure conversion in target assembly | repo construction, standard change of variables | n/a (standard) | n/a (author reference-domain weighting visible as `mlogw` in `@TTSIRT/eval_potential_reference.m:21,33`) | Dual-evaluator contract per re-audit Finding 1, UB-1 Sec. 1(V1); pattern `filtering.py:2312-2322` |

## Lemma 1 transfer caveat (re-audit Finding 4)

Zhao-Cui Lemma 1 relates the defensive constant to the square-root
approximation error FOR THE SOURCE construction. No transfer of that lemma
to the repository's frozen-ALS program is claimed; any future use requires
a separately checked transfer argument. The D1 tau-tuning step is a
viability screen (`viability_tuning_only` label where no same-target
reference exists), not an accuracy or bias-control theorem.

## Forbidden claims (binding, from the audit's source-support boundary)

- that the repository weighted ridge ALS is the author algorithm;
- that Zhao-Cui supplies the analytical HMC score route;
- that fixed ranks remain small for NAWM-class models;
- that the route is HMC-ready or posterior-correct (separate campaigns);
- that internal parity proves physical-likelihood accuracy;
- that Lemma 1 controls the frozen-ALS program's defensive error without a
  checked transfer argument;
- that the v1 structural substitution route closes the general
  (non-invertible) structural case.

## Naming rule

Public names for the engine and leaderboard rows use "squared-TT
(Zhao-Cui-family)" or "ZC-family fixed-variant squared-TT"; documentation
must not attribute the frozen-ALS/score runtime to the source authors.
