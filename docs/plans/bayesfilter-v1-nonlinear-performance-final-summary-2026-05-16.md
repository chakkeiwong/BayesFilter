# BayesFilter V1 Nonlinear Performance Final Summary

## Date

2026-05-16

## Governing artifacts

- Master program: `docs/plans/bayesfilter-v1-nonlinear-performance-master-program-2026-05-15.md`
- NP7 plan: `docs/plans/bayesfilter-v1-nonlinear-performance-np7-consolidation-default-policy-plan-2026-05-15.md`
- NP0 result: `docs/plans/bayesfilter-v1-nonlinear-performance-np0-inventory-result-2026-05-16.md`
- NP1 result: `docs/plans/bayesfilter-v1-nonlinear-performance-np1-benchmark-harness-result-2026-05-16.md`
- NP2 result: `docs/plans/bayesfilter-v1-nonlinear-performance-np2-value-fastpath-result-2026-05-16.md`
- NP3 result: `docs/plans/bayesfilter-v1-nonlinear-performance-np3-score-fastpath-result-2026-05-16.md`
- NP4 result: `docs/plans/bayesfilter-v1-nonlinear-performance-np4-xla-gate-result-2026-05-16.md`
- NP5 result: `docs/plans/bayesfilter-v1-nonlinear-performance-np5-cpu-gpu-ladder-result-2026-05-16.md`
- NP6 result: `docs/plans/bayesfilter-v1-nonlinear-performance-np6-reference-path-decision-2026-05-16.md`
- NP1 benchmark artifact: `docs/benchmarks/bayesfilter-v1-nonlinear-performance-np1-smoke-2026-05-16.md`
- NP5 benchmark artifact: `docs/benchmarks/bayesfilter-v1-nonlinear-performance-cpu-gpu-2026-05-16.md`

## Phase purpose

Consolidate NP0-NP6 into a single final policy statement covering what is accepted as exact tested-cell evidence, what remains optional or deferred, what is blocked, and whether any production default change or optimization promotion is justified now.

## Skeptical plan audit

Audit target: whether the NP0-NP6 record supports any production default change, optimization promotion, or broader readiness claim rather than only a conservative consolidation.

Findings:

- Wrong-baseline risk: NP1 is a schema-and-smoke artifact, not a promotion-grade performance baseline. It cannot be treated as a sufficient value or score benchmark ladder for production acceptance.
- Proxy-metric risk: NP1 tiny smoke timings, branch-precheck success, and Model A affine parity are explanatory or schema-gating evidence only. They do not by themselves establish production performance, derivative correctness for optimized score paths, or broad numerical validity for Models B-C.
- Narrow-cell risk: NP4 certifies only fixed-static-shape CPU XLA support for Model B/C value cells, and NP5 times only exact Model B `T=3`, `return_filtered=False`, `tf.float64` value cells. Promoting any broader XLA, GPU, or default-policy claim would exceed the tested support boundary.
- Deferred-optimization risk: NP2 and NP3 explicitly deferred value and score fast-path promotion because the required ladder, proof, parity, branch-contract, and derivation obligations are still missing.
- Reference-scope leak risk: NP6 explicitly keeps NumPy reference paths reference-only; NP4 and NP5 evidence about existing production TensorFlow value cells cannot be recycled into a TensorFlow rewrite or reference-path promotion claim.
- Hidden-readiness risk: nothing in NP0-NP6 certifies score XLA readiness, Hessian readiness, HMC convergence/readiness, broad GPU speedup, client switch-over readiness, production deployment readiness, or exact nonlinear likelihood quality for Models B-C.
- Mandatory-cell risk: the phase can close only if the final summary states unresolved cells explicitly and converts any missing promotion evidence into repair or continuation labels rather than silently filling them with optimistic defaults.

Audit outcome: pass for a conservative consolidation only. The record justifies no production default change and no new optimization promotion. The admissible output is a bounded final summary that preserves the ledger separation between engineering correctness, numerical validity, and performance evidence.

## Evidence contract

Question:

- What nonlinear performance policy is justified by NP0-NP6 today?

Baseline:

- Original production and reference code inventory as classified in NP0.
- NP1 benchmark-harness smoke artifact and schema result.
- NP2 value-fast-path deferral result.
- NP3 score-fast-path deferral/blocker result.
- NP4 CPU-only fixed-static-shape value XLA support matrix.
- NP5 trusted CPU/GPU timing rows for exact Model B `T=3` value cells.
- NP6 reference-path decision keeping NumPy reference paths NumPy-only.

Primary promotion criterion:

- A production default change or optimization promotion would require aligned evidence across the relevant ledgers: engineering correctness, numerical validity, and performance, with no active veto diagnostics for the claimed scope.

Veto diagnostics:

- promoting optional or deferred paths to defaults without medium-shape or broader promotion-grade evidence;
- treating timing rows as proof of correctness or numerical validity;
- generalizing GPU or XLA results beyond the exact tested cells;
- treating NP1 branch-precheck evidence as derivative-proof evidence for score optimizations;
- implying Hessian readiness, HMC readiness, exact Model B/C nonlinear likelihood quality, client switch-over readiness, production deployment readiness, or reference-path TensorFlow rewrite readiness.

Explanatory diagnostics only:

- NP1 tiny CPU-only smoke timings;
- NP1 Model A affine parity;
- NP1 branch-precheck linkage for score rows;
- NP5 warmup and first-call timings;
- NP5 RSS/device-environment notes;
- dense one-step projection diagnostics for Models B-C already bounded in NP0.

What will not be concluded even if this summary passes:

- any production default change;
- any new optimization promotion for NP2 or NP3 candidates;
- broad GPU speedup across models, shapes, or backends;
- broad XLA readiness, score XLA readiness, or dynamic-horizon XLA readiness;
- HMC convergence, HMC readiness, or Hessian readiness;
- exact nonlinear likelihood quality for Models B-C;
- client switch-over readiness;
- production deployment readiness outside this bounded evidence program;
- TensorFlow rewrite readiness for the NumPy reference paths.

Artifact:

- `docs/plans/bayesfilter-v1-nonlinear-performance-final-summary-2026-05-16.md`

## Program-level decision

The justified NP7 policy is:

1. **No production default change.**
2. **No new optimization promotion.**
3. **Retain all existing non-claims outside the exact tested cells.**
4. **Keep NumPy reference paths reference-only.**
5. **Carry NP4 and NP5 forward only as exact-cell support/timing evidence, not as broader readiness or switch-over evidence.**

This follows directly from the current record:

- NP2 is deferred because value fast-path candidates lack promotion-grade ladder evidence and, for higher-risk candidates, required derivation artifacts.
- NP3 is deferred or blocked because score fast-path candidates lack proof obligations, finite-difference/reference score evidence for optimized candidates, and preserved branch/fixed-support contract evidence.
- NP4 certifies only a narrow CPU static-shape XLA value support matrix.
- NP5 records only a narrow trusted Model B `T=3` CPU/GPU timing ladder and explicitly blocks any broad GPU/default narrative.
- NP6 keeps `StructuralSVDSigmaPointFilter.filter` and `particle_filter_log_likelihood` as NumPy-only reference paths because no concrete unmet TensorFlow downstream need is shown.

## Accepted / optional / deferred / blocked policy summary

### Accepted as exact tested-cell evidence

- NP1 benchmark harness schema and manifest changes are accepted as enabling artifact structure for later work, not as promotion-grade timing evidence.
- NP4 CPU static-shape XLA support is accepted only for the six value rows explicitly certified there: Model B/C, cubature/UKF/CUT4, `return_filtered=False/True`, fixed observations shape `(3, 1)`, CPU only.
- NP5 timing evidence is accepted only for the exact Model B value rows at `T=3`, `return_filtered=False`, `tf.float64`, backends `tf_svd_cubature`, `tf_svd_ukf`, and `tf_svd_cut4`, under matched CPU/GPU graph/XLA comparisons.
- NP6 reference-path decision is accepted: keep `StructuralSVDSigmaPointFilter.filter` and `particle_filter_log_likelihood` as NumPy reference paths with no TensorFlow rewrite in this phase.

### Optional but not promoted

- Using NP4-supported CPU XLA value cells as optional exact tested cells in future controlled experiments is admissible.
- Using NP5 backend-specific rows as shape-specific explanatory evidence in future plans is admissible.

These remain optional because no evidence in NP0-NP6 justifies changing defaults or broadening claims.

### Deferred

- All NP2 lower-risk value-fast-path candidates remain deferred pending promotion-grade value ladders, focused parity, wrapper-contract evidence, and any needed XLA gate evidence.
- All NP3 lower-risk score-path candidates remain deferred pending branch-contract preservation evidence, focused score parity/reference checks, and any required precheck-lineage evidence.

### Blocked

- NP2 higher-risk algebraic rewrites remain blocked pending derivation artifacts in project notation and executable preconditions.
- NP3 derivative-structure-changing candidates remain blocked pending explicit proof obligations, branch-preservation evidence, and structural fixed-support evidence where required.
- Any score XLA readiness claim remains blocked because NP4 did not certify score cells.
- Any default switch-over to GPU, XLA, new value fast path, new score fast path, or TensorFlow reference rewrite remains blocked by missing promotion evidence.

## Engineering correctness ledger

This ledger addresses implementation and contract evidence only. It does not promote numerical validity or performance by itself.

| Item | Status | Evidence | Boundary / non-claim |
| --- | --- | --- | --- |
| Surface inventory and classification | passed | NP0 fully classifies public nonlinear exports and separates production TF, helper, and NumPy reference paths | classification alone does not imply performance or readiness |
| Benchmark-harness row schema and manifest | passed for NP1 scope | NP1 records row roles, branch-precheck linkage, CPU visibility policy, skip rows, and manifest metadata | schema readiness is not a value/score promotion claim |
| Value-path eager/graph/XLA parity for tested CPU static-shape value cells | passed for NP4-listed value cells | NP4 certifies eager vs `jit_compile=True` parity on CPU for Model B/C value rows, separate for `return_filtered=False/True` | no score XLA claim, no dynamic-horizon claim, no GPU claim from NP4 |
| Branch artifact preservation for score timing harness rows | passed for NP1 harness scope only | NP1 links score timing rows to branch-precheck rows | branch linkage is not score-derivative proof |
| Production-code optimization acceptance | not passed | NP2/NP3 explicitly produce deferral/blocker results rather than accepted production changes | no value/score fast path is engineering-accepted for production |
| Reference-path rewrite admissibility | not passed for rewrite; passed for no-rewrite decision | NP6 finds no concrete unmet TensorFlow downstream need for either NumPy reference path | no TensorFlow reference rewrite readiness |
| Source-map/reset-memo update status | passed after Codex supervisor finalization | `docs/source_map.yml` includes the NP0-NP7 execution entry and `docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md` records the final NP7 no-default-change state | metadata records handoff state only; it does not promote any optimization or default-policy claim |

Engineering correctness conclusion:

- The program established schema readiness and a narrow CPU value-XLA support boundary, but it did **not** establish engineering acceptance for any new production optimization or default change.

## Numerical validity ledger

This ledger addresses parity, branch correctness boundaries, and what numerical claims are or are not admissible. It does not promote performance claims.

| Item | Status | Evidence | Boundary / non-claim |
| --- | --- | --- | --- |
| Model A affine value parity | passed for NP1 smoke scope | NP1 reports machine-precision affine parity on tiny CPU-only value rows | explanatory/schema evidence only |
| Score branch-precheck linkage | passed for NP1 smoke scope | NP1 branch-precheck rows pass for tiny Model A score rows across backends | not finite-difference/reference score proof for optimized candidates |
| Value fast-path parity for new optimized paths | unresolved / not produced | NP2 explicitly records no new parity artifact for any candidate | no value optimization promotion |
| Score fast-path correctness for new optimized paths | unresolved / not produced | NP3 explicitly records no proof artifact or focused optimized score parity/reference evidence | no score optimization promotion, no Hessian/HMC implication |
| CPU XLA parity for existing tested value cells | passed for NP4-listed cells | NP4 shows eager-vs-XLA parity on CPU for static Model B/C value rows, including filtered-state parity when returned | no score XLA validity claim, no dynamic-horizon claim |
| Branch/fixed-support preservation for score optimization candidates | unresolved / blocked | NP3 keeps branch-assertion preservation and Model C structural fixed-support evidence as required future proof/evidence | no score optimization readiness |
| Exact nonlinear likelihood quality for Models B-C | not established | NP0 and all later phases preserve this as a non-claim | no exact Model B/C likelihood claim |
| Monte Carlo uncertainty / stochastic reference-path rewrite basis | not established for rewrite purposes | NP6 notes particle-filter rewrite would require separate randomness/resampling evidence contract | no TensorFlow particle-filter readiness |

Numerical validity conclusion:

- Existing production value cells have narrow tested parity boundaries, but optimized value/score paths remain numerically unvalidated for promotion. No part of the record supports Hessian readiness, HMC readiness, or exact nonlinear likelihood quality claims for Models B-C.

## Performance ledger

This ledger addresses timing evidence only. It does not promote correctness or numerical validity by itself.

| Item | Status | Evidence | Boundary / non-claim |
| --- | --- | --- | --- |
| NP1 tiny CPU-only smoke timings | explanatory only | NP1/benchmark artifact record tiny Model A CPU value and score timings with repeats=1 | not promotion-grade ladder evidence |
| CPU XLA support boundary | passed for support, not speed | NP4 certifies CPU static-shape value-cell support, not a benchmark ladder | no runtime-improvement claim from NP4 alone |
| Trusted CPU/GPU exact-cell timing rows | passed for exact tested cells | NP5 records matched CPU/GPU graph/XLA timings for exact Model B `T=3` value rows | no broad GPU or default-policy claim |
| Backend-specific winner consistency | mixed / backend-specific only | NP5 shows cubature and UKF near-ties between CPU XLA and GPU XLA, while CUT4 slightly favors CPU graph over both XLA rows | this mixed result blocks any single device/backend default story |
| Broad GPU speedup | not established | NP5 explicitly disallows this conclusion; CUT4 exact-cell row contradicts a simple “GPU wins” narrative | no broad GPU claim |
| Broad XLA readiness or score XLA readiness | not established | NP4 is value-only and static-shape CPU only; score rows remain untested | no score/dynamic-horizon/general XLA claim |
| Model C performance | not established in NP5 | NP5 times Model B only | no Model C timing conclusion |
| Production deployment or client switch-over readiness | not established | no phase produced the required breadth of validated performance evidence | no deployment/switch-over claim |

Performance conclusion:

- The performance record is intentionally narrow and mixed. It supports only exact tested-cell statements. It does not justify a global device/mode/backend default change.

## Required decision table

| Decision | Affected function/backend | Accepted, optional, deferred, or blocked | Correctness status | Numerical/branch status | XLA status | CPU/GPU status | Primary timing result | Memory status | Default-policy result | Uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Keep current production defaults unchanged | program-wide | accepted | passed for conservative policy scope | passed because it respects all active vetoes and non-claims | preserves NP4 narrow boundary only | preserves NP5 narrow boundary only | not based on a single timing winner; based on mixed and narrow evidence | no separate memory claim | no default change | whether broader ladders would eventually justify a narrower per-cell or per-backend default | keep current defaults; reopen only with promotion-grade evidence across ledgers | no production deployment, client switch-over, or global backend claim |
| Accept NP1 schema artifact as harness readiness only | nonlinear benchmark harness | accepted | passed for schema scope | branch linkage passed for smoke rows only | not claimed | CPU-only hidden-GPU smoke only | tiny smoke rows recorded, explanatory only | no memory claim | no policy change | whether the same schema remains sufficient for broader ladders | use as artifact structure for future ladders only | no performance promotion, no score-proof promotion |
| Keep NP2 value fast-path candidates off the default path | value paths including cubature/UKF/CUT4 wrappers and low-level value helpers | deferred / blocked by candidate risk | not passed for production acceptance | not passed for optimized-path parity/derivation obligations | not promotion-certified beyond existing cells | no new CPU/GPU runtime evidence for candidates | none admissible for promotion beyond existing baseline artifacts | no memory claim | no optimization promotion | whether any lower-risk candidate survives future parity and ladder evidence | produce promotion-grade tiny+small value ladders and required derivations before reopening | no value fast-path acceptance, no broad XLA/GPU/default claim |
| Keep NP3 score fast-path candidates off the default path | score paths including `tf_svd_cubature_score`, `tf_svd_ukf_score`, `tf_svd_cut4_score` | deferred / blocked by candidate risk | not passed for production acceptance | not passed because proof, branch/fixed-support preservation, and score-reference evidence are missing | score XLA untested | no new CPU/GPU runtime evidence for candidates | none admissible for promotion beyond NP1 branch-linked smoke rows | no memory claim | no optimization promotion | whether any candidate survives future proof and branch-contract evidence | write proof artifact for the narrowest candidate, then collect focused score parity/reference evidence | no score fast-path acceptance, no Hessian claim, no HMC claim, no score XLA claim |
| Keep CPU XLA support narrow and value-only | Model B/C value cells listed in NP4 | accepted as exact tested-cell support | passed for listed value cells | passed for eager-vs-XLA value parity on CPU | supported only for listed CPU static-shape value rows | CPU only; GPU non-claim from NP4 | NP4 is support/parity evidence, not promotion timing | no memory claim | no default change | whether other horizons, score cells, or trusted GPU XLA preserve parity | if needed, add focused NP4-grade artifacts for untested cells | no score XLA, no dynamic-horizon, no GPU speed claim |
| Carry NP5 timing rows forward as exact-cell evidence only | Model B `T=3` value cells for cubature/UKF/CUT4 on CPU/GPU graph/XLA | accepted as exact tested-cell evidence | passed for comparator admissibility | branch metadata matched across rows | inherits NP4 support boundary for value cells only | trusted CPU/GPU visible run for exact rows | cubature: GPU XLA ≈ CPU XLA; UKF: GPU XLA ≈ CPU XLA; CUT4: CPU graph slightly fastest | no memory claim | no default change | repeat depth is tiny and shape scope is single-cell only | widen ladder only under a new plan if broader claims are needed | no broad GPU speedup, no Model C performance claim, no score/HMC claim |
| Keep NumPy reference paths NumPy-only | `StructuralSVDSigmaPointFilter.filter`; `particle_filter_log_likelihood` | accepted as no-rewrite decision | passed for decision-only scope | rewrite admissibility not established | not applicable | not applicable | no timing basis for rewrite policy | no memory claim | no rewrite/default change | whether a future workflow exposes a concrete unmet TF need | reopen only with a new plan and separate evidence contract | no TensorFlow rewrite readiness, no GPU claim for reference paths |

## Unresolved mandatory cells and repair plan

There is no unresolved mandatory cell for the **final consolidation decision itself**, because the justified final policy is conservative: no default change and no promotion.

There **are** unresolved mandatory cells for any future promotion attempt, and this summary records them as required repair items rather than silently filling them in:

1. **NP2 promotion-grade value ladder missing.**
   - Repair label: `NP7_REPAIR_VALUE_LADDER_REQUIRED_BEFORE_PROMOTION`
2. **NP2 higher-risk derivation artifacts missing.**
   - Repair label: `NP7_REPAIR_VALUE_DERIVATIONS_REQUIRED_BEFORE_PROMOTION`
3. **NP3 proof obligations for derivative-structure-changing candidates missing.**
   - Repair label: `NP7_REPAIR_SCORE_PROOF_REQUIRED_BEFORE_PROMOTION`
4. **NP3 focused score parity/reference evidence missing for optimized candidates.**
   - Repair label: `NP7_REPAIR_SCORE_REFERENCE_EVIDENCE_REQUIRED_BEFORE_PROMOTION`
5. **NP3 branch/fixed-support preservation evidence missing for optimized candidates.**
   - Repair label: `NP7_REPAIR_BRANCH_AND_FIXED_SUPPORT_EVIDENCE_REQUIRED`
6. **Broader XLA support matrix cells missing, especially score and dynamic-horizon cells.**
   - Repair label: `NP7_REPAIR_SCORE_AND_DYNAMIC_XLA_GATES_REQUIRED_FOR_BROADER_CLAIMS`
7. **Broader CPU/GPU ladder evidence missing beyond exact Model B `T=3` value cells.**
   - Repair label: `NP7_REPAIR_BROADER_CPU_GPU_LADDER_REQUIRED_FOR_POLICY_CHANGE`
8. **Reference-path rewrite unmet-use-case contract absent.**
   - Repair label: `NP7_REPAIR_REFERENCE_REWRITE_REQUIRES_CONCRETE_USE_CASE`

Repair-plan conclusion:

- These unresolved cells do **not** block closing NP7 because the phase is not promoting a new default or optimization. They do block any future attempt to claim broader readiness or to change defaults without a new planned evidence program.

## Commands run

No shell commands were run for NP7. This bounded worker only read governing artifacts and wrote this final summary.

## Files changed

- `docs/plans/bayesfilter-v1-nonlinear-performance-final-summary-2026-05-16.md`

## Run manifest

| Field | Value |
| --- | --- |
| Git commit | `81c647b157612a233dde1a9fbd847647a5b03b8f` |
| Dirty worktree | `true` |
| Command | `N/A` |
| Environment | `N/A` |
| Python | `N/A` |
| TensorFlow | `N/A` |
| TensorFlow Probability | `N/A` |
| CPU/GPU status | `no runtime command executed in NP7` |
| GPU intentionally hidden | `N/A` |
| Device visibility | `N/A` |
| Data version | `N/A` |
| Random seeds | `N/A` |
| Wall time | `N/A` |
| Output artifact paths | `docs/plans/bayesfilter-v1-nonlinear-performance-final-summary-2026-05-16.md` |
| Plan file | `docs/plans/bayesfilter-v1-nonlinear-performance-np7-consolidation-default-policy-plan-2026-05-15.md` |
| Result file | `docs/plans/bayesfilter-v1-nonlinear-performance-final-summary-2026-05-16.md` |
| Files read | `docs/plans/bayesfilter-v1-nonlinear-performance-np7-consolidation-default-policy-plan-2026-05-15.md`; `docs/plans/bayesfilter-v1-nonlinear-performance-np0-inventory-result-2026-05-16.md`; `docs/plans/bayesfilter-v1-nonlinear-performance-np1-benchmark-harness-result-2026-05-16.md`; `docs/plans/bayesfilter-v1-nonlinear-performance-np2-value-fastpath-result-2026-05-16.md`; `docs/plans/bayesfilter-v1-nonlinear-performance-np3-score-fastpath-result-2026-05-16.md`; `docs/plans/bayesfilter-v1-nonlinear-performance-np4-xla-gate-result-2026-05-16.md`; `docs/plans/bayesfilter-v1-nonlinear-performance-np5-cpu-gpu-ladder-result-2026-05-16.md`; `docs/plans/bayesfilter-v1-nonlinear-performance-np6-reference-path-decision-2026-05-16.md`; `docs/benchmarks/bayesfilter-v1-nonlinear-performance-np1-smoke-2026-05-16.md`; `docs/benchmarks/bayesfilter-v1-nonlinear-performance-cpu-gpu-2026-05-16.md` |

## Commands/files changed ledger

- Commands run: none
- Files changed: `docs/plans/bayesfilter-v1-nonlinear-performance-final-summary-2026-05-16.md`

## Final continuation / repair labels

### Continuation labels

- `NP7_CONTINUE_WITH_NO_DEFAULT_CHANGE`
- `NP7_CONTINUE_WITH_EXACT_TESTED_CELL_EVIDENCE_ONLY`
- `NP7_CONTINUE_WITH_REFERENCE_ONLY_NUMPY_PATHS`
- `NP7_CONTINUE_WITH_NP4_CPU_VALUE_XLA_BOUNDARY_ONLY`
- `NP7_CONTINUE_WITH_NP5_MODEL_B_T3_BACKEND_SPECIFIC_TIMING_ONLY`

### Repair labels

- `NP7_REPAIR_VALUE_LADDER_REQUIRED_BEFORE_PROMOTION`
- `NP7_REPAIR_VALUE_DERIVATIONS_REQUIRED_BEFORE_PROMOTION`
- `NP7_REPAIR_SCORE_PROOF_REQUIRED_BEFORE_PROMOTION`
- `NP7_REPAIR_SCORE_REFERENCE_EVIDENCE_REQUIRED_BEFORE_PROMOTION`
- `NP7_REPAIR_BRANCH_AND_FIXED_SUPPORT_EVIDENCE_REQUIRED`
- `NP7_REPAIR_SCORE_AND_DYNAMIC_XLA_GATES_REQUIRED_FOR_BROADER_CLAIMS`
- `NP7_REPAIR_BROADER_CPU_GPU_LADDER_REQUIRED_FOR_POLICY_CHANGE`
- `NP7_REPAIR_REFERENCE_REWRITE_REQUIRES_CONCRETE_USE_CASE`

## Non-implications

This final summary does not imply:

- broad GPU speedup;
- broad XLA readiness;
- score XLA readiness;
- dynamic-horizon XLA readiness;
- HMC convergence or HMC readiness;
- Hessian readiness;
- exact nonlinear likelihood quality for Models B-C;
- production deployment readiness;
- client switch-over readiness;
- TensorFlow rewrite readiness for NumPy reference paths;
- any new production default or optimization promotion.

## Post-run red-team note

Strongest alternative explanation for the observed accepted speed evidence:

- The apparent wins in some NP5 rows may be dominated by exact-cell shape effects, small repeat depth, and backend-specific constant factors rather than a stable device/mode advantage that would survive broader ladders.

What result would overturn the default-policy decision:

- A new plan producing promotion-grade, multi-cell evidence across the relevant ledgers could overturn it: for example, a broader matched ladder showing consistent steady-state benefit without new correctness or numerical-validity vetoes, or a score/value optimization that survives proof, parity, branch, and performance gates.

Weakest part of the current evidence:

- The weakest part is the breadth of the performance evidence: NP5 is only exact Model B `T=3` value timing, and NP1 is only tiny smoke/schema evidence. This weakness belongs primarily to the **performance** ledger, with linked consequences for whether a broader default-policy promotion could ever be justified.

Ledger attribution of the remaining uncertainty:

- Engineering correctness uncertainty: production optimization candidates were not accepted, so contract-preservation questions remain open for future work.
- Numerical validity uncertainty: optimized score/value candidates still lack proof/parity evidence, and exact Model B/C nonlinear likelihood quality remains unclaimed.
- Performance uncertainty: timing breadth is narrow and mixed, preventing any global default change.

## Phase exit label

`NP7_COMPLETE_NO_DEFAULT_CHANGE_NO_OPTIMIZATION_PROMOTION`
