# LEDH Compact Score Default Visible Execution Ledger

Date: 2026-07-08

Status: `LAUNCH_PACKAGE_DRAFT`

## Program

- Master program:
  `docs/plans/bayesfilter-ledh-compact-score-default-master-program-2026-07-08.md`
- Runbook:
  `docs/plans/bayesfilter-ledh-compact-score-default-visible-gated-execution-runbook-2026-07-08.md`
- Stop handoff:
  `docs/plans/bayesfilter-ledh-compact-score-default-visible-stop-handoff-2026-07-08.md`

## Ledger

### 2026-07-08 - Launch Package - DRAFTED

Evidence contract:

- Question: Can every LEDH leaderboard score row be moved to compact
  forward-sensitivity by default, with reverse/manual-total-VJP routes demoted
  to historical and blocked from admission?
- Baseline/comparator: LGSSM compact score route, existing score validator,
  existing score scripts, admitted value artifacts, and historical reverse
  diagnostics as comparators only.
- Primary criterion: launch package includes a master program, visible runbook,
  Phase 0 subplan, review bundle, explicit demotion policy, and stop
  conditions before implementation edits.
- Veto diagnostics: old route remains described as admissible default, missing
  compact-only validator phase, missing model-by-model subplans, missing Claude
  review protocol, or unsupported score/scientific claims.
- Non-claims: no score admitted, no model port complete, no leaderboard
  rebuilt, no HMC/posterior/scientific claim.

Actions:

- Drafted compact-score-default master program and visible runbook.
- Drafted Phase 0 route-demotion subplan.

Artifacts:

- `docs/plans/bayesfilter-ledh-compact-score-default-master-program-2026-07-08.md`
- `docs/plans/bayesfilter-ledh-compact-score-default-visible-gated-execution-runbook-2026-07-08.md`
- `docs/plans/bayesfilter-ledh-compact-score-default-phase0-route-demotion-subplan-2026-07-08.md`

Gate status:

- `IN_PROGRESS`

Next action:

- Run local launch-package checks, then Claude read-only review.

### 2026-07-08 - Launch Review - CLAUDE_AGREE_WITH_PATCHABLE_NOTES

Evidence contract:

- Question: Is the launch package safe to start Phase 0 route demotion?
- Baseline/comparator: current score validator allowlist, LGSSM compact route,
  and actual-SV reverse-record implementation.
- Primary criterion: read-only review agrees Phase 0 may start and identifies
  no material boundary blocker.
- Veto diagnostics: old routes still code-admissible without a hard procedural
  stop; overbroad correctness wording; missing stop conditions.
- Non-claims: no implementation correctness, no score admission, no model port
  completion.

Actions:

- Claude review gate first returned `probe_timeout`.
- Direct tiny probe returned `CLAUDE_PROBE_OK`.
- Narrow packet-only Claude review returned `VERDICT: AGREE`.
- Patched launch package to record:
  - before Phase 1 validator/static guards, old-route blocking is procedural,
    so no `--admit-full` or full-admission command may run;
  - one-point/tested-coordinate FD evidence must not be described as broad
    all-parameter mathematical correctness;
  - old test filename phase suffixes are stale and must be mapped by model
    content and row ID.

Artifacts:

- `.claude_reviews/20260708-140327-bayesfilter-ledh-compact-score-default-launch-review-2026-07-08/status.json`
- `docs/reviews/bayesfilter-ledh-compact-score-default-launch-review-bundle-2026-07-08.md`

Gate status:

- `PATCHED_AFTER_AGREE_READY_FOR_PHASE0`

Next action:

- Launch Phase 0 precheck and route inventory.

### 2026-07-08 - Phase 0 - PASSED_ROUTE_DEMOTION_POLICY_GATE

Evidence contract:

- Question: Does the repository have a clear boundary that old
  reverse-record/manual-total-VJP routes are historical and wrong for
  leaderboard score admission?
- Baseline/comparator: current validator allowlist, active score scripts,
  LGSSM compact route, actual-SV/predator-prey reverse-record paths.
- Primary criterion: inventory old-route admission surfaces and draft Phase 1
  enforcement work.
- Veto diagnostics: any full-admission command before Phase 1 guards, old
  route still called future admissible default, target scalar ambiguity.
- Non-claims: no compact port, no new score admission, no full score run.

Actions:

- Ran static inventory.
- Ran focused score contract tests.
- Wrote Phase 0 result and Phase 1 subplan.
- Claude read-only review returned `VERDICT: AGREE`.

Artifacts:

- `docs/plans/bayesfilter-ledh-compact-score-default-phase0-route-demotion-result-2026-07-08.md`
- `docs/plans/bayesfilter-ledh-compact-score-default-phase1-contract-subplan-2026-07-08.md`
- `docs/reviews/bayesfilter-ledh-compact-score-default-phase0-result-phase1-subplan-review-bundle-2026-07-08.md`

Gate status:

- `PASSED`

Next action:

- Execute Phase 1 validator/static-guard enforcement.

### 2026-07-08 - Phase 1 - PASSED_COMPACT_ONLY_FULL_ADMISSION_GUARD

Evidence contract:

- Question: Does the shared score contract prevent `manual_total_vjp*` and
  reverse-record routes from full LEDH leaderboard score admission?
- Baseline/comparator: Phase 0 inventory and current validator allowlist.
- Primary criterion: old routes fail full admission while LGSSM compact remains
  valid.
- Veto diagnostics: old route still admitted, compact route broken,
  directional-only FD promoted, target scalar changed.
- Non-claims: no non-LGSSM compact port, no full score run, no HMC/posterior
  claim.

Actions:

- Split validator provenance into compact-admissible and historical-diagnostic
  route sets.
- Added full-admission rejection for historical `manual_total_vjp*` routes.
- Added focused tests for old-route-plus-memory-pass blocking.
- Ran focused local checks.

Artifacts:

- `bayesfilter/highdim/ledh_score_contract.py`
- `tests/highdim/test_ledh_score_contract_phase1.py`
- `tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py`
- `tests/highdim/test_ledh_actual_sv_score_phase5_contract.py`
- `tests/highdim/test_ledh_predator_prey_score_phase4_contract.py`
- `docs/plans/bayesfilter-ledh-compact-score-default-phase1-contract-result-2026-07-08.md`
- `docs/plans/bayesfilter-ledh-compact-score-default-phase2-lgssm-reference-subplan-2026-07-08.md`

Local checks:

- `58 passed, 2 warnings`

Gate status:

- `PASSED_PENDING_REVIEW`

Next action:

- Review Phase 1 result and Phase 2 subplan.

### 2026-07-08 - Phase 1 Review - AGREED_AFTER_PATCH

Evidence contract:

- Question: Did Phase 1 close the full-admission gap for `manual_total_vjp*`
  and is Phase 2 safe to start?
- Primary criterion: Claude/Codex review finds no unresolved material blocker
  after wording patches.
- Veto diagnostics: overclaiming historical diagnostic representability or
  transferring authority to Claude.
- Non-claims: no score admission or model port.

Actions:

- Claude first returned `VERDICT: REVISE`.
- Patched Phase 1 result to distinguish raw diagnostics from validator-admitted
  score artifacts.
- Patched handoff wording so Codex remains executor and Claude advisory only.
- Focused re-review returned `VERDICT: AGREE`.

Gate status:

- `PASSED`

Next action:

- Execute Phase 2 LGSSM compact reference freeze.

### 2026-07-08 - Phase 2 - PASSED_LGSSM_COMPACT_REFERENCE_FREEZE

Evidence contract:

- Question: Is LGSSM frozen as compact reference, with historical reverse route
  blocked from current admission?
- Baseline/comparator: LGSSM compact source/tests, Phase 1 validator, and
  historical LGSSM reverse diagnostics.
- Primary criterion: compact route validates; historical route remains
  diagnostic; stale `T=2` evidence is not current admission.
- Veto diagnostics: manual reverse admitted, compact route broken, stale memory
  overclaim, target scalar drift.
- Non-claims: no non-LGSSM compact port, no fresh GPU memory rerun, no HMC or
  posterior claim.

Actions:

- Inspected LGSSM route constants and compact forward-sensitivity loop.
- Confirmed compact transport uses streaming transport value+JVP.
- Ran focused LGSSM and score-contract tests.
- Wrote Phase 2 result and Phase 3 actual-SV compact subplan.

Local checks:

- `38 passed, 2 warnings`

Artifacts:

- `docs/plans/bayesfilter-ledh-compact-score-default-phase2-lgssm-reference-result-2026-07-08.md`
- `docs/plans/bayesfilter-ledh-compact-score-default-phase3-actual-sv-subplan-2026-07-08.md`
- `docs/reviews/bayesfilter-ledh-compact-score-default-phase2-result-phase3-subplan-review-bundle-2026-07-08.md`

Gate status:

- `PASSED_PENDING_REVIEW`

Next action:

- Review Phase 2 result and Phase 3 actual-SV compact subplan.

### 2026-07-08 - Phase 2 Review - REVISE_PATCHED

Evidence contract:

- Question: Is LGSSM correctly frozen as compact reference, and is the
  actual-SV compact port subplan safe and concrete?
- Primary criterion: review finds no material target-scalar or route-boundary
  issue before Phase 3 execution.
- Veto diagnostics: wrong actual-SV gamma map, missing artifact provenance
  replacement, stale memory overclaim, Claude authority transfer.
- Non-claims: no actual-SV implementation yet.

Actions:

- Claude returned `VERDICT: REVISE`.
- Patched the actual-SV subplan to preserve the current `_gamma_beta` mapping:
  `gamma` is the standard normal CDF transform of `gamma_unconstrained`.
- Patched the actual-SV subplan to require artifact-builder provenance and
  default dispatch replacement away from `ACTUAL_SV_MANUAL_SCORE_ROUTE_ID`.

Gate status:

- `PATCHED_PENDING_FOCUSED_REVIEW`

Next action:

- Focused review of the patched Phase 3 subplan.

### 2026-07-08 - Phase 2 Focused Review - AGREED_AFTER_PROMPT_FIX

Evidence contract:

- Question: Is the patched actual-SV Phase 3 subplan safe to execute after
  correcting the gamma-map and provenance-replacement issues?
- Primary criterion: read-only review finds no unresolved material blocker.
- Veto diagnostics: wrong actual-SV parameter map, missing provenance
  replacement, Claude authority transfer, or stale memory overclaim.
- Non-claims: no actual-SV implementation yet.

Actions:

- Focused Claude review returned `VERDICT: AGREE` after the subplan patch.

Gate status:

- `PASSED`

Next action:

- Execute Phase 3 actual-SV compact score port.

### 2026-07-08 - Phase 3 - PASSED_TINY_COMPACT_ACTUAL_SV_GATE

Evidence contract:

- Question: Can actual-SV compute the same transformed finite-`N` LEDH
  `log_likelihood` score using compact forward sensitivity instead of the
  historical reverse-record/manual-total-VJP route?
- Baseline/comparator: admitted actual-SV value artifact, same-scalar finite
  differences, LGSSM compact style, and historical actual-SV reverse route as
  diagnostic only.
- Primary criterion: compact route matches value scalar at the tiny gate,
  avoids tape/autodiff and reverse records in default dispatch, passes
  coordinate FD, and emits compact provenance.
- Veto diagnostics: wrong target scalar, old route default/admission, reverse
  transport pullback as default, tape/autodiff, stopped partial derivative,
  nonfinite score, or FD failure.
- Non-claims: no full `N=10000,T=1000` admission, no memory evidence, no HMC
  or posterior claim.

Actions:

- Added actual-SV compact provenance to shared score contract.
- Added actual-SV compact forward-sensitivity score route.
- Kept the historical actual-SV manual-total-VJP route diagnostic only.
- Updated artifact provenance to use computed compact route.
- Added focused tests for same scalar, no reverse records, no autodiff, FD, and
  old-route blocking.
- Ran required local checks and tiny same-scalar diagnostic.

Artifacts:

- `bayesfilter/highdim/ledh_score_contract.py`
- `docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py`
- `tests/highdim/test_ledh_actual_sv_score_phase5_contract.py`
- `docs/plans/bayesfilter-ledh-compact-score-default-phase3-actual-sv-tiny-compact-score-2026-07-08.json`
- `docs/plans/bayesfilter-ledh-compact-score-default-phase3-actual-sv-result-2026-07-08.md`
- `docs/plans/bayesfilter-ledh-compact-score-default-phase4-fixed-sir-subplan-2026-07-08.md`
- `docs/reviews/bayesfilter-ledh-compact-score-default-phase3-actual-sv-review-bundle-2026-07-08.md`

Local checks:

- `python -m py_compile ...` passed.
- `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest tests/highdim/test_ledh_actual_sv_score_phase5_contract.py tests/highdim/test_ledh_score_contract_phase1.py -q`
  passed: `38 passed, 2 warnings`.

Tiny artifact:

- `score_derivative_provenance = compact_forward_sensitivity_no_autodiff_same_scalar_actual_sv_ledh_pfpf_ot`
- `score_admission_status = tiny_score_diagnostic_not_admitted`
- `max_abs_error = 6.049662710727599e-06`
- `max_rel_error = 2.1665722010856176e-05`

Gate status:

- `PASSED_PENDING_REVIEW`

Next action:

- Run Claude read-only review of Phase 3 result and Phase 4 subplan.

### 2026-07-08 - Phase 3 Review - AGREED_AFTER_PROMPT_REPAIR

Evidence contract:

- Question: Did Phase 3 avoid overclaiming and is Phase 4 safe to start?
- Primary criterion: read-only review finds no material blocker after local
  checks and result/subplan artifacts.
- Veto diagnostics: full-admission overclaim, memory overclaim, inconsistent
  raw-value/JVP boundary, fixed-SIR relabeling loophole.
- Non-claims: Claude does not authorize full runs, scientific claims, or
  product/release boundaries.

Actions:

- Claude review gate returned `REVIEW_STATUS=probe_timeout`.
- Direct tiny probe returned `CLAUDE_PROBE_OK`.
- First narrowed direct review returned procedural `VERDICT: REVISE` because
  the prompt forbade file inspection.
- Repaired prompt to allow read-only inspection of the packet and cited fixed
  paths.
- Focused Claude review returned `VERDICT: AGREE`.

Artifacts:

- `.claude_reviews/20260708-172950-bayesfilter-ledh-compact-score-default-phase3-actual-sv-review-2026-07-08/status.json`
- `docs/reviews/bayesfilter-ledh-compact-score-default-phase3-actual-sv-review-bundle-2026-07-08.md`

Gate status:

- `PASSED`

Next action:

- Start Phase 4 fixed-SIR compact score precheck.

### 2026-07-08 - Phase 4 - PASSED_TINY_COMPACT_FIXED_SIR_GATE

Evidence contract:

- Question: Can fixed-SIR compute the same finite-`N` LEDH `log_likelihood`
  score in `sir_log_scale_theta` coordinates using compact forward sensitivity
  instead of the historical p8p reverse/manual-total-VJP route?
- Baseline/comparator: admitted fixed-SIR value artifact, p8p historical
  same-target diagnostic, LGSSM/actual-SV compact style, and tiny same-scalar
  finite differences.
- Primary criterion: compact route carries particles, log weights, tangents,
  and log-likelihood tangents forward; emits compact provenance; passes
  all-coordinate tiny FD; old route cannot full-admit.
- Veto diagnostics: wrong scalar, old route default/admission,
  `manual_total_vjp*` full admission, tape/autodiff, stopped partial
  derivative, reverse-record default, parameterized diagnostic row promotion,
  nonfinite score, or FD failure.
- Non-claims: no full `N=10000,T=20` fixed-SIR score admission, no HMC or
  posterior claim, no Zhao-Cui source-faithfulness claim.

Actions:

- Added fixed-SIR compact score provenance to the shared contract.
- Implemented compact fixed-SIR forward sensitivities for transition, LEDH
  flow, densities, log-weight increments, and streaming finite Sinkhorn
  transport value+JVP.
- Kept the old p8p/manual-total-VJP route historical and blocked from full
  admission.
- Added focused tests for no autodiff, no reverse records, same-scalar FD,
  compact artifact schema, full-admission memory/shape guard, and historical
  route blocking.
- Repaired two Phase 4 implementation issues found by the runbook:
  tangent flattening of SIR components and FD-helper dtype configuration.
- Generated a tiny compact score JSON artifact.
- Wrote Phase 4 result and Phase 5 predator-prey subplan.

Local checks:

- `python -m py_compile ...` passed.
- `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py tests/highdim/test_ledh_score_contract_phase1.py -q`
  passed: `37 passed, 2 warnings`.
- Tiny artifact readback validated with compact provenance and
  `tiny_score_diagnostic_not_admitted`.

Tiny artifact:

- `score_derivative_provenance = compact_forward_sensitivity_no_autodiff_same_scalar_fixed_sir_logscale_ledh_pfpf_ot`
- `score_admission_status = tiny_score_diagnostic_not_admitted`
- `max_abs_error = 0.0005332655891479021`
- `max_rel_error = 1.7359599675765744e-05`

Artifacts:

- `docs/plans/bayesfilter-ledh-compact-score-default-phase4-fixed-sir-result-2026-07-08.md`
- `docs/plans/bayesfilter-ledh-compact-score-default-phase4-fixed-sir-tiny-compact-score-2026-07-08.json`
- `docs/plans/bayesfilter-ledh-compact-score-default-phase5-predator-prey-subplan-2026-07-08.md`
- `docs/reviews/bayesfilter-ledh-compact-score-default-phase4-fixed-sir-review-bundle-2026-07-08.md`

Gate status:

- `PASSED_PENDING_REVIEW`

Next action:

- Run Claude read-only review of Phase 4 result and Phase 5 subplan.

### 2026-07-08 - Phase 4 Review - BOUNDED_PACKET_AGREE

Evidence contract:

- Question: Did Phase 4 close only a tiny fixed-SIR compact-score gate, avoid
  relabeling the historical p8p/manual-total-VJP route, and draft a safe Phase
  5 predator-prey compact migration subplan?
- Primary criterion: read-only review finds no material blocker while local
  checks and artifacts carry the evidence burden.
- Veto diagnostics: full-admission overclaim, historical-route relabeling,
  missing Phase 5 stop conditions, or Claude authority transfer.
- Non-claims: packet-only review is not full code inspection, not proof of
  correctness, and not full-row score admission.

Actions:

- Claude review gate returned `REVIEW_STATUS=probe_timeout`.
- Direct health probe returned `CLAUDE_PROBE_OK`.
- A narrowed file-inspection prompt hung and was interrupted.
- Packet-only Claude review returned `VERDICT: AGREE`.

Review limitation:

- This was a bounded packet-only review. It supports continuing the runbook
  because local tests and artifacts passed, but it is weaker than a full
  file-inspection material review.

Gate status:

- `PASSED_WITH_BOUNDED_PACKET_REVIEW`

Next action:

- Start Phase 5 predator-prey compact score precheck.

### 2026-07-08 - Phase 5 - PASSED_TINY_COMPACT_PREDATOR_PREY_GATE

Evidence contract:

- Question: Can predator-prey compute the same finite-`N` LEDH
  `log_likelihood` score in physical `(r,K,a,s,u,v)` coordinates using compact
  forward sensitivity instead of the historical reverse/manual-total-VJP route?
- Baseline/comparator: admitted predator-prey value artifact, existing
  historical score diagnostic, fixed-SIR compact port pattern, LGSSM compact
  reference, and tiny same-scalar finite differences.
- Primary criterion: compact route carries particles, log weights, tangents,
  and log-likelihood tangents forward; emits compact provenance; passes
  all-coordinate tiny FD; old route cannot full-admit.
- Veto diagnostics: wrong scalar, old route default/admission,
  `manual_total_vjp*` full admission, tape/autodiff, stopped partial
  derivative, reverse-record default, non-predator-prey row promotion,
  nonfinite score, or FD failure.
- Non-claims: no full `N=10000,T=20` predator-prey score admission, no HMC or
  posterior claim, no Zhao-Cui source-faithfulness claim.

Actions:

- Ran predator-prey baseline tests before migration: `35 passed, 2 warnings`.
- Added predator-prey compact score provenance to the shared contract.
- Implemented compact predator-prey forward sensitivities for transition,
  LEDH flow, densities, log-weight increments, and streaming finite Sinkhorn
  transport value+JVP.
- Updated default across-seed predator-prey score dispatch to compact route.
- Kept the old predator-prey manual-total-VJP route historical and blocked from
  full admission.
- Added focused tests for no autodiff, no reverse records, same-scalar value
  match, same-scalar FD, compact artifact schema, full-admission memory guard,
  and historical route blocking.
- Generated a tiny compact score JSON artifact.
- Wrote Phase 5 result and Phase 6 generalized-SV subplan.

Local checks:

- `python -m py_compile ...` passed.
- `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest tests/highdim/test_ledh_predator_prey_score_phase4_contract.py tests/highdim/test_ledh_score_contract_phase1.py -q`
  passed: `38 passed, 2 warnings`.
- Tiny artifact readback validated with compact provenance and
  `tiny_score_diagnostic_not_admitted`.

Tiny artifact:

- `score_derivative_provenance = compact_forward_sensitivity_no_autodiff_same_scalar_predator_prey_ledh_pfpf_ot`
- `score_admission_status = tiny_score_diagnostic_not_admitted`
- `max_abs_error = 1.4451367178480723e-06`
- `max_rel_error = 7.776625302494132e-09`

Artifacts:

- `docs/plans/bayesfilter-ledh-compact-score-default-phase5-predator-prey-result-2026-07-08.md`
- `docs/plans/bayesfilter-ledh-compact-score-default-phase5-predator-prey-tiny-compact-score-2026-07-08.json`
- `docs/plans/bayesfilter-ledh-compact-score-default-phase6-generalized-sv-subplan-2026-07-08.md`
- `docs/reviews/bayesfilter-ledh-compact-score-default-phase5-predator-prey-review-bundle-2026-07-08.md`

Gate status:

- `PASSED_PENDING_REVIEW`

Next action:

- Run Claude read-only review of Phase 5 result and Phase 6 subplan.

### 2026-07-08 - Phase 5 Review - CODEX_SUBSTITUTE_AGREE

Evidence contract:

- Question: Did Phase 5 close only a tiny predator-prey compact-score gate,
  avoid relabeling the historical manual-total-VJP route, and draft a safe
  Phase 6 generalized-SV subplan?
- Primary criterion: review finds no material blocker while local checks and
  artifacts carry the evidence burden.
- Veto diagnostics: full-admission overclaim, historical-route relabeling,
  generalized-SV target substitution, or missing Phase 6 stop conditions.
- Non-claims: substitute review is not a Claude review, not proof of
  correctness, and not full-row score admission.

Actions:

- Claude review gate escalation was rejected by the local policy reviewer as
  external data disclosure risk.
- Codex did not route around the rejection.
- Spawned a fresh Codex read-only substitute reviewer.
- The reviewer did not return within the wait window.
- Codex wrote a bounded substitute review artifact with explicit limitations.

Artifacts:

- `docs/reviews/bayesfilter-ledh-compact-score-default-phase5-predator-prey-codex-substitute-review-2026-07-08.md`

Gate status:

- `PASSED_WITH_CODEX_SUBSTITUTE_REVIEW`

Next action:

- Start Phase 6 generalized-SV compact score precheck.

### 2026-07-08 - Phase 6 - PASSED_TINY_COMPACT_GENERALIZED_SV_GATE

Evidence contract:

- Question: Can generalized-SV compute the same finite-`N` source-route
  prior-mean raw-y LEDH `log_likelihood` score in active transformed
  coordinates using compact forward sensitivity?
- Baseline/comparator: admitted generalized-SV value artifact, generalized-SV
  value runner, previous compact model ports, and tiny same-scalar finite
  differences.
- Primary criterion: compact route carries particles, log weights, tangents,
  and log-likelihood tangents forward; emits compact provenance; matches the
  value route; passes all-coordinate tiny FD; does not substitute actual-SV or
  KSC semantics.
- Veto diagnostics: wrong target scalar, actual-SV/KSC substitution,
  log-square proposal promoted as target likelihood, wrong coordinate order,
  stopped partial derivative, tape/autodiff, reverse-record default, nonfinite
  score, or tiny FD failure.
- Non-claims: no full `N=10000,T=1008` generalized-SV score admission, no HMC
  or posterior claim, no SP500 validity claim, no author-default truth claim.

Actions:

- Ran generalized-SV value-route compile precheck.
- Added generalized-SV compact score provenance to the shared contract.
- Implemented compact generalized-SV forward sensitivities for transformed
  parameters, stationary initialization, AR(1) transition, parameterized LEDH
  flow, raw-y target observation likelihood, normalized log weights, and
  streaming finite Sinkhorn transport value+JVP.
- Added focused tests for no autodiff, no reverse records, same-scalar value
  match against the actual value core, same-scalar FD, compact artifact schema,
  full-admission memory guard, and target-substitution rejection.
- Generated a tiny compact score JSON artifact.
- Wrote Phase 6 result and Phase 7 KSC-SV subplan.

Local checks:

- `python -m py_compile ...` passed.
- `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest tests/highdim/test_ledh_generalized_sv_score_phase6_contract.py tests/highdim/test_ledh_score_contract_phase1.py -q`
  passed: `34 passed, 2 warnings`.
- Tiny artifact readback validated with compact provenance and
  `tiny_score_diagnostic_not_admitted`.

Tiny artifact:

- `score_derivative_provenance = compact_forward_sensitivity_no_autodiff_same_scalar_generalized_sv_ledh_pfpf_ot`
- `score_admission_status = tiny_score_diagnostic_not_admitted`
- `max_abs_error = 4.1007384898997246e-05`
- `max_rel_error = 0.001305349464063811`

Artifacts:

- `docs/plans/bayesfilter-ledh-compact-score-default-phase6-generalized-sv-result-2026-07-08.md`
- `docs/plans/bayesfilter-ledh-compact-score-default-phase6-generalized-sv-tiny-compact-score-2026-07-08.json`
- `docs/plans/bayesfilter-ledh-compact-score-default-phase7-ksc-sv-subplan-2026-07-08.md`
- `docs/reviews/bayesfilter-ledh-compact-score-default-phase6-generalized-sv-review-bundle-2026-07-08.md`

Gate status:

- `PASSED_PENDING_REVIEW`

Next action:

- Run read-only review of Phase 6 result and Phase 7 subplan.

### 2026-07-08 - Phase 6 Review - CODEX_NARROW_SUBSTITUTE_AGREE

Evidence contract:

- Question: Did Phase 6 close only a tiny generalized-SV compact-score gate,
  preserve the raw-y source-route target, and draft a safe Phase 7 KSC-SV
  subplan?
- Primary criterion: read-only review finds no material blocker while local
  checks and artifacts carry the evidence burden.
- Veto diagnostics: full-admission overclaim, target substitution,
  historical-route relabeling, KSC/actual-SV boundary confusion, or missing
  Phase 7 stop conditions.
- Non-claims: substitute review is not Claude review, not proof of
  correctness, and not full-row score admission.

Actions:

- Claude review gate escalation was rejected by the local policy reviewer as
  external data disclosure risk.
- Codex did not route around the rejection.
- A broad Codex substitute reviewer timed out twice.
- Codex wrote a review limitation artifact.
- A narrowed packet-only Codex substitute reviewer returned
  `VERDICT: AGREE`.

Artifacts:

- `docs/reviews/bayesfilter-ledh-compact-score-default-phase6-generalized-sv-codex-substitute-review-limitation-2026-07-08.md`
- `docs/reviews/bayesfilter-ledh-compact-score-default-phase6-generalized-sv-review-bundle-2026-07-08.md`

Gate status:

- `PASSED_WITH_CODEX_NARROW_SUBSTITUTE_REVIEW`

Next action:

- Start Phase 7 KSC-SV compact score precheck.

### 2026-07-08 - Phase 7 - PASSED_TINY_COMPACT_KSC_SV_GATE

Evidence contract:

- Question: Can KSC-SV compute the same finite-`N` KSC finite-mixture
  surrogate LEDH `log_likelihood` score in `synthetic_unconstrained`
  coordinates using compact forward sensitivity?
- Baseline/comparator: admitted KSC-SV value artifact, KSC-SV value runner,
  previous compact model ports, and tiny same-scalar finite differences.
- Primary criterion: compact route carries particles, log weights, tangents,
  and log-likelihood tangents forward; emits compact provenance; matches the
  value route; passes all-coordinate tiny FD; does not substitute exact
  actual-SV, generalized-SV, or raw Gaussian semantics.
- Veto diagnostics: wrong target scalar, exact actual-SV target substitution,
  generalized-SV substitution, raw Gaussian callback promotion, wrong
  coordinate order, stopped partial derivative, tape/autodiff, reverse-record
  default, nonfinite score, exact native actual-SV overclaim, or tiny FD
  failure.
- Non-claims: no full `N=10000,T=1000` KSC-SV score admission, no exact native
  actual-SV likelihood, no HMC or posterior claim, no runtime ranking claim.

Actions:

- Ran KSC-SV value-route compile precheck.
- Added KSC-SV compact score provenance to the shared contract.
- Implemented compact KSC-SV forward sensitivities for transformed parameters,
  stationary initialization, SV transition, parameterized LEDH flow, KSC
  finite-mixture target likelihood, normalized log weights, and streaming
  finite Sinkhorn transport value+JVP.
- Added focused tests for no autodiff, no reverse records, same-scalar value
  match against the actual KSC value core, same-scalar FD, compact artifact
  schema, exact native overclaim rejection, and target-substitution rejection.
- Generated a tiny compact score JSON artifact.
- Wrote Phase 7 result and Phase 8 integration subplan.

Local checks:

- `python -m py_compile ...` passed.
- `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest tests/highdim/test_ledh_ksc_sv_score_phase7_contract.py tests/highdim/test_ledh_score_contract_phase1.py -q`
  passed: `33 passed, 2 warnings`.
- Tiny artifact readback validated with compact provenance and
  `tiny_score_diagnostic_not_admitted`.

Tiny artifact:

- `score_derivative_provenance = compact_forward_sensitivity_no_autodiff_same_scalar_ksc_sv_ledh_pfpf_ot`
- `score_admission_status = tiny_score_diagnostic_not_admitted`
- `max_abs_error = 1.688629603341374e-05`
- `max_rel_error = 6.364022943512981e-05`

Artifacts:

- `docs/plans/bayesfilter-ledh-compact-score-default-phase7-ksc-sv-result-2026-07-08.md`
- `docs/plans/bayesfilter-ledh-compact-score-default-phase7-ksc-sv-tiny-compact-score-2026-07-08.json`
- `docs/plans/bayesfilter-ledh-compact-score-default-phase8-integration-subplan-2026-07-08.md`

Gate status:

- `PASSED_PENDING_REVIEW`

Next action:

- Run read-only review of Phase 7 result and Phase 8 subplan.

### 2026-07-08 - Phase 7 Review - CODEX_SUBSTITUTE_AGREE

Evidence contract:

- Question: Can Phase 7 close as a tiny KSC-SV compact score gate and safely
  hand off to Phase 8 integration?
- Primary criterion: reviewer finds no material blocker in the Phase 7 result,
  shared score contract, KSC score route, or Phase 8 subplan.
- Veto diagnostics: full-row overclaim, exact native actual-SV overclaim,
  historical route admission, target substitution, or missing Phase 8 stop
  conditions.
- Non-claims: substitute review is not Claude review and is not proof of full
  score admission.

Actions:

- Attempted the Claude review gate:
  `bash /home/chakwong/python/claudecodex/scripts/claude_review_gate.sh ...`
- The local policy layer rejected the Claude call as external data disclosure
  risk.
- Codex did not route around the rejection.
- Spawned a fresh read-only Codex substitute review over the fixed Phase 7/8
  bundle.
- Substitute review returned `VERDICT: AGREE`.

Gate status:

- `PASSED_WITH_CODEX_SUBSTITUTE_REVIEW`

Next action:

- Execute Phase 8 integration policy patch and checks.

### 2026-07-08 - Phase 8 - PASSED_INTEGRATION_POLICY_GATE_FULL_SCORE_ROWS_BLOCKED

Evidence contract:

- Question: Does the leaderboard score workflow default to compact
  forward-sensitivity score routes and block historical `manual_total_vjp*`
  full admission?
- Baseline/comparator: shared score contract, compact per-model score ports,
  existing inclusive leaderboard merger, and score-memory candidate artifacts.
- Primary criterion: integration code and tests require a Phase 1 validated
  compact score artifact for admission; historical routes and raw legacy memory
  JSONs cannot be promoted.
- Veto diagnostics: historical route admitted, blocked row exposes admitted
  score provenance, tiny artifact promoted, target/value identity ignored, or
  full score completion claimed.
- Non-claims: no full score row is admitted in this phase.

Actions:

- Patched `docs/benchmarks/benchmark_two_lane_highdim_ledh_inclusive_results.py`
  so admitted score rows require
  `validate_ledh_score_artifact(..., require_admitted=True)`.
- Added candidate-only fields for legacy or non-admitted score evidence:
  `score_candidate_artifact`, `score_candidate_derivative_provenance`, and
  `score_candidate_admission_status`.
- Updated `tests/test_two_lane_highdim_ledh_leaderboard.py` to assert that:
  - LGSSM's raw July 6 score-memory JSON is not admitted because it lacks the
    Phase 1 score schema;
  - fixed-SIR's historical `manual_total_vjp` score-memory candidate is blocked
    as diagnostic-only;
  - blocked rows do not expose admitted `score_derivative_provenance`.
- Generated a Phase 8 integration candidate artifact.
- Wrote Phase 8 result and final review bundle.

Local checks:

- `python -m py_compile docs/benchmarks/benchmark_two_lane_highdim_ledh_inclusive_results.py`
  passed.
- `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest tests/test_two_lane_highdim_ledh_leaderboard.py tests/highdim/test_ledh_score_contract_phase1.py -q`
  passed: `34 passed, 2 warnings`.
- `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest tests/highdim/test_ledh_score_contract_phase1.py tests/highdim/test_ledh_generalized_sv_score_phase6_contract.py tests/highdim/test_ledh_ksc_sv_score_phase7_contract.py tests/test_two_lane_highdim_ledh_leaderboard.py -q`
  passed: `51 passed, 2 warnings`.
- Candidate generation command exited 0:
  `docs/plans/bayesfilter-ledh-compact-score-default-phase8-integration-candidate-2026-07-08.json`

Candidate artifact summary:

- LGSSM LEDH row: `executed_value_only_score_blocked`;
  candidate provenance
  `compact_forward_sensitivity_no_autodiff_same_scalar_lgssm_ledh_pfpf_ot`;
  candidate status `legacy_raw_score_memory_not_admitted`.
- Fixed-SIR LEDH row: `executed_value_only_score_blocked`;
  candidate provenance
  `manual_total_vjp_no_autodiff_same_scalar_fixed_sir_logscale_ledh_pfpf_ot`;
  candidate status `historical_diagnostic_not_admitted`.

Artifacts:

- `docs/plans/bayesfilter-ledh-compact-score-default-phase8-integration-result-2026-07-08.md`
- `docs/plans/bayesfilter-ledh-compact-score-default-phase8-integration-candidate-2026-07-08.json`
- `docs/plans/bayesfilter-ledh-compact-score-default-phase8-integration-candidate-2026-07-08.md`
- `docs/reviews/bayesfilter-ledh-compact-score-default-phase8-integration-review-bundle-2026-07-08.md`

Gate status:

- `PASSED_INTEGRATION_POLICY_GATE_FULL_SCORE_ROWS_BLOCKED`

Next action:

- Review Phase 8 result. Then create a new full-row compact score admission
  program for schema-valid `N=10000` score artifacts, starting with LGSSM and
  fixed-SIR.

### 2026-07-08 - Phase 8 Review - CODEX_SUBSTITUTE_AGREE

Evidence contract:

- Question: Did Phase 8 safely integrate compact score admission policy without
  promoting historical or legacy raw score-memory artifacts?
- Primary criterion: reviewer finds no material blocker in the result,
  candidate artifact, integration implementation, tests, or score contract.
- Veto diagnostics: historical route admitted, legacy raw artifact promoted,
  candidate fields mixed with admitted score fields, or full score leaderboard
  completion overclaimed.
- Non-claims: substitute review is not Claude review and cannot authorize full
  score admission or scientific claims.

Actions:

- Spawned a fresh read-only Codex substitute review over the Phase 8 fixed
  paths because Claude review remained blocked by the local policy layer.
- Substitute review returned `VERDICT: AGREE`.

Gate status:

- `PHASE8_REVIEW_PASSED_WITH_CODEX_SUBSTITUTE_REVIEW`

Final program status:

- Compact score default integration policy is complete.
- Full score leaderboard rows remain blocked until schema-valid compact
  `N=10000` score artifacts are produced and validated.
