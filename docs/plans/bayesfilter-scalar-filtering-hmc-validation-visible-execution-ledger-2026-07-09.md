# Scalar Filtering HMC Validation Visible Execution Ledger

Date: 2026-07-09

## Status

`PHASE_2AE_CURRENT_SEQUENTIAL_REFERENCE_BRANCH_BLOCKED_EXPANSION_REQUIRED`

## Ledger

### 2026-07-09 - Phase 0 - PRECHECK

Evidence contract:

- Question: Does the current scalar HMC route preserve native-divergence
  availability semantics strongly enough to launch Phase 1 without a
  zero-divergence claim?
- Baseline/comparator: 2026-07-08 scalar filtering closeout and current HMC
  diagnostics source.
- Primary criterion: governance review plus focused local tests confirm missing
  native divergence is not treated as zero.
- Veto diagnostics: failed tests, proxy divergence substitution, unsupported
  HMC readiness/convergence/posterior/default/source-faithfulness claim.
- Non-claims: no zero divergences, convergence, posterior correctness, HMC
  readiness, GPU/XLA readiness, default readiness, or Zhao-Cui source
  faithfulness.

Actions:

- Drafted master program, all phase subplans, visible runbook, ledger, stop
  handoff, and compact review bundle.

Artifacts:

- `docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md`
- `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase0-telemetry-policy-subplan-2026-07-09.md`
- `docs/plans/bayesfilter-scalar-filtering-hmc-validation-visible-gated-execution-runbook-2026-07-09.md`
- `docs/reviews/bayesfilter-scalar-filtering-hmc-validation-phase0-governance-review-bundle-2026-07-09.md`

Gate status:

- `IN_PROGRESS`

Next action:

- Run Claude review gate or documented fallback, then execute Phase 0 focused
  checks.

### 2026-07-09 - Phase 0 - REPAIR_LOOP

Evidence contract:

- Question: Does the current scalar HMC route preserve native-divergence
  availability semantics strongly enough to launch Phase 1 without a
  zero-divergence claim?
- Baseline/comparator: 2026-07-08 scalar filtering closeout and current HMC
  diagnostics source.
- Primary criterion: missing native divergence must remain unavailable, not
  zero, even when log-accept ratios are present.
- Veto diagnostics: proxy log-accept threshold counts driving native
  `zero_divergences`.
- Non-claims: no zero divergences, convergence, posterior correctness, HMC
  readiness, GPU/XLA readiness, default readiness, or Zhao-Cui source
  faithfulness.

Actions:

- Claude review gate was blocked by external-transfer policy.
- Fresh Codex substitute review returned `VERDICT: REVISE`.
- Patched `screen_hmc_diagnostics` so absent native `divergences` always makes
  `zero_divergences` unavailable/false.
- Added regression coverage for log-accept present with native divergence
  absent.

Artifacts:

- `bayesfilter/inference/hmc_diagnostics.py`
- `tests/test_common_inference_runtime_contracts.py`
- `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase0-telemetry-policy-result-2026-07-09.md`

Gate status:

- `PASSED_AFTER_REPAIR_WITH_CODEX_SUBSTITUTE_REVIEW`

Next action:

- Refresh and review Phase 1 subplan before runtime.

### 2026-07-09 - Phase 1 - PRECHECK

Evidence contract:

- Question: Does the scalar fixed-kernel route pass a modest CPU-hidden
  finite/acceptance short-chain validation screen under the Phase 0 telemetry
  policy?
- Baseline/comparator: 2026-07-08 three-seed finite-telemetry diagnostic.
- Primary criterion: all three 2026-07-09 seeds produce retained finite samples,
  finite target log probabilities, finite log-accept ratios, no runtime errors,
  per-seed acceptance strictly between `0.05` and `0.99`, and no positive
  native divergence if native divergence telemetry is available.  If native
  divergence is unavailable, it is recorded as unavailable and not used as a
  pass criterion.
- Veto diagnostics: runtime error, nonfinite required arrays, invalid artifact,
  missing seed, acceptance-screen failure, positive native divergence when
  available, unavailable divergence treated as zero, telemetry semantics
  mismatch, or unsupported claim.
- Non-claims: no HMC readiness, convergence, posterior correctness,
  zero-divergence claim when native divergence is unavailable, GPU/XLA
  readiness, default readiness, sampler superiority, or Zhao-Cui source
  faithfulness.

Actions:

- Refreshed the Phase 1 subplan with concrete settings, artifact paths, command,
  and skeptical plan audit.
- Claude review remains blocked by external-transfer policy from Phase 0; use a
  fresh Codex substitute review for this material subplan.

Artifacts:

- `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase1-cpu-short-chain-subplan-2026-07-09.md`

Gate status:

- `REVISE_FROM_CODEX_SUBSTITUTE_REVIEW_ROUND_1`

Next action:

- Patch the Phase 1 subplan findings and rerun the Codex substitute review
  before Phase 1 runtime.

### 2026-07-09 - Phase 1 - REVIEW_REPAIR_ROUND_1

Review verdict:

- `VERDICT: REVISE` from Codex substitute reviewer.

Findings:

- Stop conditions were narrower than the Phase 1 veto contract.
- The subplan needed to explicitly forbid using raw
  `HMCScreenResult.passed` as the Phase 1 pass flag when native divergence is
  unavailable.
- Serious-run manifest fields needed explicit artifact coverage.
- The quiet log directory precheck needed to be explicit before shell
  redirection.

Repair:

- Expanded stop conditions to all primary-criterion failures and veto
  diagnostics.
- Added an implementation boundary requiring the Phase 1 harness to derive its
  finite/acceptance gate separately from raw `HMCScreenResult.passed`.
- Added explicit manifest-field requirements.
- Added the `mkdir -p` log-directory precheck.

Artifacts:

- `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase1-cpu-short-chain-subplan-2026-07-09.md`

Gate status:

- `PENDING_CODEX_SUBSTITUTE_REVIEW_ROUND_2`

### 2026-07-09 - Phase 2AB Through 2AE - SEQUENTIAL_REFERENCE_BRANCH_CLOSEOUT

Evidence contract:

- Question: Can a CPU-hidden sequential-reference route produce a valid
  reference nomination for the scalar MAP-local target without overstating
  posterior/HMC/GPU/default claims?
- Baseline/comparator: final Phase 2AB, 2AC, and 2AD artifacts.
- Primary criterion: beta reaches `1.0`, finite target/base/log weights,
  terminal ESS ratio at least `0.50`, terminal max weight at most `0.08`,
  unique ancestor fraction at least `0.25`, and aggregate rejuvenation
  acceptance in `[0.10, 0.90]`.
- Nonclaims: no valid reference, no HMC-reference agreement, no posterior
  correctness, no HMC readiness/convergence, no zero-divergence claim, no
  sampler superiority/statistical ranking, no GPU/XLA/default readiness, and
  no Zhao-Cui source faithfulness.

Actions:

- Phase 2AA selected a sequential/transport branch after independent SNIS
  proposal attempts failed.
- Phase 2AB implemented a sequential tempering pilot and repaired a cumulative
  ESS scheduling bug before final runtime.
- Phase 2AC implemented fallback-boundary resampling after Claude review and
  focused local review.
- Phase 2AD implemented a projected root-ancestor diversity gate after Claude
  found ambiguous projected-diversity and terminal semantics.
- Phase 2AE wrote a no-runtime reference-method expansion decision.

Checks:

- Phase 2AC focused tests plus Phase 2AB/2Z regressions passed before runtime:
  `31 passed`.
- Phase 2AD focused tests plus Phase 2AC regression passed before runtime:
  `28 passed`.
- `python -m py_compile` passed for Phase 2AC and Phase 2AD harnesses.
- `git diff --check` passed before Phase 2AC and Phase 2AD runtime.

Runtime results:

- Phase 2AB failed nomination: beta stalled at `0.3419540270406287`.
- Phase 2AC failed nomination: beta reached `1.0`, terminal ESS ratio
  `0.9912539055044092`, terminal max weight `0.010002188339361427`, but unique
  ancestor fraction was `0.21875 < 0.25`.
- Phase 2AD failed nomination: unique ancestor fraction was preserved at
  `0.4140625`, but beta stalled at `0.9712250668187553`.
- Phase 2AE decision: current local fallback-resampling sequential-reference
  branch is blocked; continuing requires a materially different reviewed
  reference-method design or program closeout with reference agreement
  unresolved.

Artifacts:

- `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ab-transport-or-sequential-reference-result-2026-07-09.md`
- `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ac-sequential-resampling-repair-result-2026-07-09.md`
- `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ad-diversity-preserving-sequential-repair-result-2026-07-09.md`
- `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ae-reference-method-expansion-decision-result-2026-07-09.md`

Gate status:

- `CURRENT_SEQUENTIAL_REFERENCE_BRANCH_BLOCKED_EXPANSION_REQUIRED`

Next action:

- Do not proceed to Phase 3 GPU/XLA.  Draft a new reviewed reference-method
  design branch, or close out the scalar validation program with reference
  agreement unresolved.

### 2026-07-09 - Phase 2Y - FINAL_RUNTIME_RESULT

Plan/review:

- Patched the Phase 2Y subplan with explicit hypotheses H1-H6 after tracing
  the Phase 2S/2U/2W/2X math/code path.
- Claude review was attempted through the local review gate but blocked by the
  approval layer because it would transfer private repository planning and
  diagnostic context to an external Claude service.  No workaround was used.
- A fresh local Codex substitute review was used instead and is weaker than
  full Claude material review.
- The user's 2026-07-09 instruction cleared only the Phase 2Y CPU-hidden
  diagnostic boundary, not Phase 3 GPU/XLA or scientific/default claims.

Checks:

- `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement.py`
  passed before runtime: `20 passed`.
- `git diff --check` passed before runtime.

Runtime:

- Ran the reviewed Phase 2Y CPU-hidden diagnostic command.
- Exit status: `0`.
- Wall time: `54.949073967000004` seconds.
- Wrote JSON/Markdown artifacts:
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_cpu_hidden_2026-07-09.json`
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization_cpu_hidden_2026-07-09.md`

Gate:

- `phase2y_target_geometry_localization_passed=True`.
- Final vetoes: `[]`.
- Artifact bug indicated: `False`.
- Proposal-family mismatch indicated: `True`.
- Orientation replay matched the Phase 2U row-vector formula with max absolute
  error `4.440892098500626e-16`; the Phase 2S display string remains
  ambiguous but did not indicate a runtime transform bug.
- Phase 2W/2X proposal log densities replayed with max absolute delta `0.0`.

Interpretation:

- Phase 2Y supports, descriptively only, that the Phase 2W/2X failures are
  proposal-family/global-geometry mismatch rather than affine/proposal replay
  bugs.
- Top-weight anchor norms ranged from `3.1676400712527686` to
  `8.91162729981821`, far outside the Phase 2S trust radius `0.6`.
- Ray target-minus-quadratic residuals reached max absolute
  `104.96420467633178`.
- This does not establish a valid reference, posterior correctness, HMC
  readiness/convergence, zero divergences, statistical ranking, GPU/XLA
  readiness, default readiness, or Zhao-Cui source faithfulness.

Next action:

- Review Phase 2Z proposal strategy pilot subplan:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2z-proposal-strategy-pilot-subplan-2026-07-09.md`.
- Phase 3 GPU/XLA remains blocked until a later reviewed reference-repair or
  independent replication branch passes and explicitly authorizes it.

Gate status:

- `PHASE_2Z_REVIEWED_READY_PENDING_RUNTIME_CLEARANCE`

### 2026-07-09 - Phase 2Z - SUBPLAN_REVIEW

Review:

- Wrote Phase 2Z proposal strategy pilot subplan:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2z-proposal-strategy-pilot-subplan-2026-07-09.md`.
- Claude material review remained unavailable because the approval layer
  blocked external transfer of private repository context.
- Fresh local Codex substitute review returned `VERDICT: AGREE`; this is
  weaker than full Claude material review.

Gate:

- Phase 2Z is reviewed-ready for future implementation/runtime only if the
  user clears that next runtime boundary.
- Phase 3 GPU/XLA remains blocked.

Gate status:

- `PHASE_2Z_REVIEWED_READY_PENDING_RUNTIME_CLEARANCE`

### 2026-07-09 - Phase 2Z - FINAL_RUNTIME_RESULT

Plan/review:

- Patched the Phase 2Z subplan before implementation to pin exact mixture
  weights, local scales, deterministic component allocations, and seeds.
- Claude material review was not retried after the prior approval-layer block
  for external transfer of private repository context.
- Updated local Codex substitute review returned `VERDICT: AGREE`; this is
  weaker than full Claude material review.

Checks:

- `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization.py`
  passed before runtime and after the timeout repair: `11 passed`.
- `git diff --check` passed.

Runtime:

- Initial `4096` samples per candidate attempt timed out with exit code `124`
  before writing JSON/Markdown artifacts.  This is a runtime-plan flaw, not
  proposal evidence.
- Repaired the CPU-hidden pilot size to `1024` samples per candidate under the
  subplan timeout-feasibility escape hatch.
- Reran the same Phase 2Z command with the repaired harness.
- Exit status: `0`.
- Wall time: `188.37495324999327` seconds.
- Wrote JSON/Markdown artifacts:
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_cpu_hidden_2026-07-09.json`
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_cpu_hidden_2026-07-09.md`

Gate:

- `phase2z_proposal_strategy_pilot_passed=True` for artifact validity.
- Final vetoes: `[]`.
- Candidate nominated: `False`.
- No proposal passed ESS `>= 256`, ESS ratio `>= 0.05`, and max normalized
  weight `<= 0.05`.
- Candidate diagnostics:
  - `student_t_centered`: ESS `21.01706760314462`, ratio
    `0.02052448008119592`, max weight `0.10525616575524867`.
  - `student_t_shifted`: ESS `14.656246107251146`, ratio
    `0.014312740339112447`, max weight `0.2428262178992962`.
  - `anchor_mixture_student_t`: ESS `26.071556547207543`, ratio
    `0.025460504440632366`, max weight `0.1112365984074613`.
  - `ridge_line_student_t`: ESS `1.535816743965212`, ratio
    `0.0014998210390285273`, max weight `0.7997380494257067`.

Interpretation:

- Phase 2Z argues against another blind independent SNIS proposal tweak.
- It does not prove SNIS impossible, invalidate the target, or invalidate the
  Phase 2V HMC mechanics screen.
- It does not establish posterior correctness, HMC readiness/convergence, zero
  divergences, statistical ranking, GPU/XLA readiness, default readiness, or
  Zhao-Cui source faithfulness.

Next action:

- Review Phase 2AA reference-branch decision subplan:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2aa-reference-branch-decision-subplan-2026-07-09.md`.
- Phase 3 GPU/XLA remains blocked.

Gate status:

- `SUPERSEDED_BY_PHASE_2AE_CURRENT_SEQUENTIAL_REFERENCE_BRANCH_BLOCKED_EXPANSION_REQUIRED`

### 2026-07-09 - Phase 2W - REVIEW_REPAIR_AND_RUNTIME_RESULT

Review:

- Codex substitute review round 1 returned `VERDICT: REVISE`.
- Findings:
  - The subplan called a weighted variance divided by ESS an MCSE; this needed
    the square root.
  - Phase 2U artifact and selected-kernel handoff validity needed to be an
    explicit reference-validity veto because the Phase 2W target route reuses
    the Phase 2U MAP-local adapter.
- Repairs:
  - Patched the Phase 2W MCSE definition to
    `sqrt(weighted second-moment variance / ESS)`.
  - Added explicit Phase 2S/2U/2V artifact and Phase 2U selected-kernel
    handoff validity vetoes.
  - Harmonized the stop conditions with the repaired veto contract.
- Focused Codex substitute review round 2 returned `VERDICT: AGREE`.
- Review strength: Codex substitute review, weaker than full Claude material
  review.

Checks:

- `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen.py`
  passed: `15 passed`.
- `git diff --check` passed.

Runtime:

- Ran the reviewed Phase 2W command.
- Exit status: `0`.
- Wall time: `81.64405142096803` seconds.
- Wrote JSON/Markdown artifacts:
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_cpu_hidden_2026-07-09.json`
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_cpu_hidden_2026-07-09.md`

Gate:

- `phase2w_importance_reference_agreement_passed=False`.
- Final vetoes:
  `reference_ess_below_threshold`, `reference_ess_ratio_below_threshold`.
- Reference ESS: `22.894679726459746`.
- Reference ESS ratio: `0.022358085670370845`.
- Target log-probabilities and log weights were finite.
- HMC-vs-reference agreement was not evaluated or interpreted.
- Native divergence remained unavailable in the Phase 2V comparator; no
  zero-divergence claim.

Interpretation:

- Phase 2W is a reference-proposal failure for the fixed standard-normal
  proposal.
- It is not evidence against the Phase 2V HMC chain, the target, or the whole
  research direction.
- It does not establish posterior correctness, HMC readiness, convergence,
  zero divergences, statistical ranking, GPU/XLA readiness, default readiness,
  or Zhao-Cui source faithfulness.

Next action:

- Review Phase 2X shifted-mixture reference repair before runtime.  Phase 3
  GPU/XLA remains blocked until a later reviewed reference-repair or reference
  replication result explicitly authorizes a GPU/XLA reproduction subplan.

### 2026-07-09 - Phase 2X - REVIEW_AND_RUNTIME_RESULT

Review:

- Fresh Codex substitute review returned `VERDICT: AGREE`.
- Review strength: Codex substitute review, weaker than full Claude material
  review.
- Reviewer found no blocking findings and confirmed the proposal parameters
  used Phase 2W pilot diagnostics only, not Phase 2V HMC moments.

Checks:

- `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen.py`
  passed: `21 passed`.
- `git diff --check` passed.

Runtime:

- Ran the reviewed Phase 2X command.
- Exit status: `0`.
- Wall time: `113.19665156002156` seconds.
- Wrote JSON/Markdown artifacts:
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_cpu_hidden_2026-07-09.json`
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_cpu_hidden_2026-07-09.md`

Gate:

- `phase2x_shifted_mixture_reference_repair_passed=False`.
- Final vetoes:
  `reference_ess_below_threshold`, `reference_ess_ratio_below_threshold`.
- Reference ESS: `33.4215730897076`.
- Reference ESS ratio: `0.01631912748520879`.
- Target log-probabilities, proposal log densities, and log weights were
  finite.
- HMC-vs-reference agreement was not evaluated or interpreted.
- Native divergence remained unavailable in the Phase 2V comparator; no
  zero-divergence claim.

Interpretation:

- Phase 2X is a second reference-proposal failure, now for the shifted-mixture
  repair.
- The current importance-reference branch is blocked until target/proposal
  mismatch is localized.
- It is not evidence against the Phase 2V HMC chain, the target, or the whole
  research direction.
- It does not establish posterior correctness, HMC readiness, convergence,
  zero divergences, statistical ranking, GPU/XLA readiness, default readiness,
  or Zhao-Cui source faithfulness.

Next action:

- Review Phase 2Y target-geometry localization before any further runtime.
  Phase 3 GPU/XLA remains blocked.

### 2026-07-09 - Phase 2Y - HANDOFF_REVIEW

Review:

- Fresh Codex substitute review returned `VERDICT: AGREE`.
- Review strength: Codex substitute review, weaker than full Claude material
  review.
- Reviewer found no blocking findings and confirmed:
  - Phase 2Y is consistent with Phase 2W/2X ESS-only reference failures.
  - The phase is narrowed to diagnostic target/proposal localization.
  - It forbids reference-agreement, HMC-readiness, posterior-correctness,
    GPU/XLA, default-readiness, and source-faithfulness claims.
  - Phase 2W/2X JSON artifacts contain enough proposal and log-probability
    data to reconstruct top-weight diagnostics.

Gate status:

- `PHASE_2Y_REVIEWED_READY_PENDING_BLOCKER_BOUNDARY_CLEARANCE`

Next action:

- Do not run Phase 2Y automatically from the Phase 2X blocker.  If continuing,
  execute Phase 2Y as a diagnostic localization phase only, with the reviewed
  evidence contract and nonclaims preserved.  Phase 3 GPU/XLA remains blocked.

### 2026-07-09 - Phase 2V - REVIEW_ROUND_2_PASS

Review:

- Focused Codex substitute review returned `VERDICT: AGREE`.
- Reviewer confirmed the initial-state blocker is fixed across primary
  criterion, veto diagnostics, fixed runtime settings, and stop conditions.
- Reviewer confirmed the next handoff is exact: Phase 2V pass can only draft a
  reviewed scalar reference/posterior-agreement subplan, while GPU/XLA remains
  blocked until a later reviewed reference/posterior-agreement result.

Gate status:

- `PASSED_FOR_PHASE_2V_IMPLEMENTATION`

### 2026-07-09 - Phase 2V - FINAL_RUNTIME_RESULT

Checks:

- `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen.py`
  passed: `13 passed`.
- `git diff --check` passed.

Runtime:

- Ran the reviewed Phase 2V command.
- Exit status: `0`.
- Wall time: `178.8504172930261` seconds.
- Wrote JSON/Markdown artifacts:
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_cpu_hidden_2026-07-09.json`
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2v_longer_selected_map_local_screen_cpu_hidden_2026-07-09.md`

Gate:

- `phase2v_longer_selected_map_local_screen_passed=True`.
- Final vetoes: `[]`.
- Selected kernel: `L=2`, `step_size=0.785`, trajectory length `1.57`.
- Initial state: `u_new=[0, 0, 0, 0]`.
- Acceptance: `0.40625`.
- Retained samples: `128` finite, `0` nonfinite.
- Native divergence unavailable; no zero-divergence claim.

Interpretation:

- Phase 2V supports drafting a scalar reference/posterior-agreement diagnostic
  subplan.
- It does not establish posterior correctness, HMC readiness, convergence,
  zero divergences, statistical ranking, GPU/XLA readiness, default readiness,
  or Zhao-Cui source faithfulness.

Next action:

- Review Phase 2W MAP-local importance reference agreement subplan before
  runtime.  Phase 3 GPU/XLA remains blocked.

Next action:

- Rerun focused Codex substitute review before runtime.

### 2026-07-09 - Phase 1 - RUNTIME_RESULT

Evidence contract:

- Question: Does the scalar fixed-kernel route pass a modest CPU-hidden
  finite/acceptance short-chain validation screen under the Phase 0 telemetry
  policy?
- Primary criterion: all three seeds finite, no runtime errors, acceptance in
  `(0.05, 0.99)`, and no positive native divergence if available.
- Non-claims: no HMC readiness, convergence, posterior correctness,
  zero-divergence claim when native divergence is unavailable, GPU/XLA
  readiness, default readiness, sampler superiority, or Zhao-Cui source
  faithfulness.

Actions:

- Ran the Phase 1 CPU-hidden benchmark with output redirected to the planned
  quiet log.
- Wrote the Phase 1 JSON/Markdown artifacts.
- Wrote the Phase 1 result and drafted a Phase 1R repair subplan.

Artifacts:

- `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1_cpu_hidden_2026-07-09.json`
- `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1_cpu_hidden_2026-07-09.md`
- `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase1_cpu_short_chain.log`
- `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase1-cpu-short-chain-result-2026-07-09.md`
- `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase1r-acceptance-envelope-repair-subplan-2026-07-09.md`

Result:

- Phase 1 failed by predeclared acceptance screen:
  `seed_2_acceptance_outside_phase1_screen`.
- Acceptance rates were `[0.9375, 0.75, 1.0]`.
- Samples, target log probabilities, and log-accept ratios were finite.
- Native divergence was `not_exposed_by_kernel` for all seeds and was not
  treated as zero divergences.

Gate status:

- `FAILED_ACCEPTANCE_SCREEN_REPAIR_TRIGGER`

Next action:

- Review the Phase 1R subplan before any repair runtime.

### 2026-07-09 - Phase 1R - REVIEW_REPAIR_ROUND_1

Review verdict:

- `VERDICT: REVISE` from Codex substitute reviewer.

Findings:

- Phase 1R was a valid repair direction in principle.
- Changing `num_burnin_steps` from 4 to 8 would confound the stated
  more-retained-draws-only repair.
- Serious-run manifest fields needed explicit artifact coverage.
- Phase 1 result wording should say native positive-divergence veto was
  unassessable/not applicable, not that no native positive divergence was
  observed from unavailable telemetry.

Repair:

- Kept `num_burnin_steps=4` and changed only `num_results` from 16 to 64.
- Added explicit manifest-field requirements to the Phase 1R subplan.
- Corrected the Phase 1 result decision-table wording around unavailable native
  divergence.
- Updated Phase 1R harness/tests to keep burn-in fixed.

Gate status:

- `PENDING_CODEX_SUBSTITUTE_REVIEW_ROUND_2`

Next action:

- Rerun focused Codex substitute review before Phase 1R runtime.

### 2026-07-09 - Phase 1R - RUNTIME_RESULT

Evidence contract:

- Question: Does the same scalar fixed-kernel route avoid the Phase 1
  all-accepted seed when the acceptance screen has more retained draws?
- Primary criterion: same kernel, same seeds, same burn-in, `num_results=64`,
  finite telemetry, acceptance in `(0.05, 0.99)`, and no positive native
  divergence if available.
- Non-claims: no HMC readiness, convergence, posterior correctness,
  zero-divergence claim when native divergence is unavailable, GPU/XLA
  readiness, default readiness, sampler superiority, or Zhao-Cui source
  faithfulness.

Actions:

- Ran the Phase 1R CPU-hidden benchmark with output redirected to the planned
  quiet log.
- Wrote the Phase 1R JSON/Markdown artifacts.
- Wrote the Phase 1R result and refreshed Phase 2 subplan as a local quadratic
  reference-agreement check.

Artifacts:

- `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1r_longer_same_kernel_cpu_hidden_2026-07-09.json`
- `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase1r_longer_same_kernel_cpu_hidden_2026-07-09.md`
- `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase1r_longer_same_kernel.log`
- `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase1r-acceptance-envelope-repair-result-2026-07-09.md`
- `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2-reference-agreement-subplan-2026-07-09.md`

Result:

- Phase 1R passed.
- Acceptance rates were `[0.921875, 0.734375, 0.578125]`.
- Samples, target log probabilities, and log-accept ratios were finite.
- Native divergence was `not_exposed_by_kernel` for all seeds and was not
  treated as zero divergences.

Gate status:

- `PASSED_FINITE_ACCEPTANCE_SCREEN_WITH_BOUNDARIES`

Next action:

- Review the refreshed Phase 2 local quadratic reference-agreement subplan
  before runtime.

### 2026-07-09 - Phase 2 - REVIEW_REPAIR_ROUND_1

Review verdict:

- `VERDICT: REVISE` from Codex substitute reviewer.

Findings:

- The local quadratic reference formula omitted the general `C_u @` factor in
  `m_u = C_u @ F.T @ l_z`.
- Agreement thresholds were predeclared but under-justified.

Repair:

- Corrected the reference construction to `K_u = F.T @ K_z @ F`,
  `C_u = inv(K_u)`, and `m_u = C_u @ F.T @ l_z`.
- Added threshold rationale explaining that max mean error `0.5` and standard
  deviation ratio `[0.5, 2.0]` are loose engineering screens for a short
  `N=192` local-reference check, not statistical proof of posterior agreement.

Gate status:

- `PENDING_CODEX_SUBSTITUTE_REVIEW_ROUND_2`

Next action:

- Rerun focused Phase 2 subplan review before runtime.

### 2026-07-09 - Phase 2 - RUNTIME_RESULT

Evidence contract:

- Question: Do Phase 1R HMC marginal moments agree with the local quadratic
  Gaussian reference implied by the accepted scalar geometry/mass handoff in
  HMC `u` coordinates?
- Primary criterion: valid inputs/reference; max marginal mean error at most
  `0.5`; standard-deviation ratios in `[0.5, 2.0]`.
- Non-claims: no exact posterior correctness, HMC readiness/convergence,
  zero-divergence claim, GPU/XLA readiness, default readiness, sampler
  superiority, or Zhao-Cui source faithfulness.

Actions:

- Implemented the Phase 2 local quadratic reference harness and focused tests.
- Ran the Phase 2 CPU-hidden artifact comparison with output redirected to the
  planned quiet log.
- Wrote Phase 2 JSON/Markdown artifacts, Phase 2 result, and Phase 2R
  localization subplan.

Artifacts:

- `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_2026_07_09.py`
- `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_cpu_hidden_2026-07-09.json`
- `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_cpu_hidden_2026-07-09.md`
- `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2_local_quadratic_reference.log`
- `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2-reference-agreement-result-2026-07-09.md`
- `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2r-local-reference-localization-subplan-2026-07-09.md`

Result:

- Phase 2 failed the local reference screen.
- Max marginal mean error was `2.728680904681481` versus threshold `0.5`.
- Standard-deviation ratios were
  `[3.5135873864369875, 2.78503939933846, 2.0517916307643334, 2.266402929702481]`,
  outside `[0.5, 2.0]`.
- Native divergence remained unavailable and no zero-divergence claim was made.

Gate status:

- `FAILED_LOCAL_REFERENCE_SCREEN_REPAIR_REQUIRED`

Next action:

- Review Phase 2R localization before any further runtime.  Do not proceed to
  Phase 3 GPU/XLA until the local-reference mismatch is localized or repaired.

### 2026-07-09 - Phase 2R - REVIEW_REPAIR_ROUND_1

Review verdict:

- `VERDICT: REVISE` from Codex substitute reviewer.

Findings:

- Phase 2R was a valid next direction after Phase 2 failed.
- Required artifacts/checks were too underspecified for a governed research
  run.
- Evidence contract lacked a concrete localization decision criterion and exact
  handoff outcomes.

Repair:

- Added exact Phase 2R harness, JSON, Markdown, quiet log, result paths, focused
  tests, and planned command.
- Added a predeclared localization outcome table:
  `transform_bookkeeping_mismatch`, `outside_geometry_trust_region`,
  `local_quadratic_reference_center_weak`,
  `short_chain_transient_or_multimodality_possible`, or
  `inconclusive_needs_longer_cpu_chain`.
- Added concrete diagnostic thresholds and exact handoff conditions requiring
  one selected outcome or an inconclusive result.

Gate status:

- `CODEX_SUBSTITUTE_REVIEW_ROUND_2_AGREE`

Next action:

- Execute Phase 2R localization only after explicit continuation from this
  stop point.  Phase 3 GPU/XLA remains blocked until Phase 2R localizes or
  repairs the Phase 2 mismatch under a reviewed plan.

### 2026-07-09 - Phase 2R - RUNTIME_RESULT

Evidence contract:

- Question: What is the smallest discriminating explanation for the Phase 2
  local-reference mismatch?
- Primary criterion: valid localization artifact with one selected
  predeclared outcome or an inconclusive result.
- Non-claims: no posterior correctness, HMC readiness/convergence,
  zero-divergence claim, GPU/XLA readiness, default readiness, sampler
  superiority, or Zhao-Cui source faithfulness.

Actions:

- Ran the Phase 2R CPU-hidden localization benchmark with output redirected to
  the planned quiet log.
- Wrote the Phase 2R JSON/Markdown artifacts.
- Wrote the Phase 2R result and drafted Phase 2S geometry/centering repair.

Artifacts:

- `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_cpu_hidden_2026-07-09.json`
- `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_cpu_hidden_2026-07-09.md`
- `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2r_localization.log`
- `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2r-local-reference-localization-result-2026-07-09.md`
- `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2s-geometry-centering-repair-subplan-2026-07-09.md`

Result:

- Phase 2R passed as a localization diagnostic.
- Selected outcome: `outside_geometry_trust_region`.
- Transform identity max absolute error was `3.3306690738754696e-15`.
- Pooled HMC mean norm in `u` was `3.2079965478482895`, above the `0.6`
  outside-trust warning threshold.
- Seed mean norms were `1.9314939055758316`, `3.024038476914353`, and
  `5.718576264956027`.
- Seed 2 local quadratic drop was `15.838221711125513`, above threshold `10`.
- Target replay did not support a pooled-HMC-mean-higher-than-center claim:
  pooled minus center was `-0.2815211066872152`.

Gate status:

- `PASSED_LOCALIZATION_OUTCOME_OUTSIDE_GEOMETRY_TRUST_REGION`

Next action:

- Review Phase 2S geometry/centering repair before runtime.  Phase 3 GPU/XLA
  remains blocked.

### 2026-07-09 - Phase 2S - REVIEW_REPAIR_ROUND_1

Review verdict:

- `VERDICT: REVISE` from Codex substitute reviewer.

Findings:

- Sample-count wording incorrectly said `sample_count=90` was five times the
  required finite-sample count.
- Locator fallback/exception was not a Phase 2S veto.
- Nonzero holdout was not explicit even though the helper can mark zero
  holdout as passed when finite samples only meet the training floor.

Repair:

- Corrected sample-count wording to `90 = 10x` the parameter count and `2x`
  the finite-sample floor.
- Made locator fallback, locator exception, and
  `accepted_optimizer_position=False` a Phase 2S veto.
- Required nonzero holdout and made zero holdout a Phase 2S veto.

Gate status:

- `CODEX_SUBSTITUTE_REVIEW_ROUND_2_AGREE`

### 2026-07-09 - Phase 2S - RUNTIME_RESULT

Evidence contract:

- Question: Can a MAP-local SPD quadratic geometry, initialized from the Phase
  2R reference-mean neighborhood, produce a usable covariance/reference handoff
  for the scalar filtering target?
- Primary criterion: accepted initializer, accepted locator, finite
  gate-required target replays, SPD precision/covariance with condition number
  at most `1e5`, finite sample count at least five times regression parameter
  count, nonzero holdout, holdout accepted, and no hard vetoes.
- Non-claims: no certified global MAP, posterior covariance correctness, HMC
  readiness/convergence, zero-divergence claim, GPU/XLA readiness, default
  readiness, sampler superiority, or Zhao-Cui source faithfulness.

Actions:

- Implemented Phase 2S harness and tests.
- Ran focused pre-runtime tests.
- Ran the Phase 2S CPU-hidden diagnostic with output redirected to the planned
  quiet log.
- Wrote Phase 2S JSON/Markdown artifacts, result, and Phase 2T subplan.

Artifacts:

- `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_2026_07_09.py`
- `tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair.py`
- `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_cpu_hidden_2026-07-09.json`
- `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_cpu_hidden_2026-07-09.md`
- `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2s_geometry_centering_repair.log`
- `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2s-geometry-centering-repair-result-2026-07-09.md`
- `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2t-map-local-reference-handoff-subplan-2026-07-09.md`

Result:

- Phase 2S passed.
- Locator status: `tfp_lbfgs_locator_accepted`.
- Locator log probability: `-37.77528495512359`.
- Locator score norm: `1.017238315038726e-10`.
- Geometry finite samples: `90`; required finite samples: `45`;
  regression parameter count: `9`; holdout count: `22`.
- Holdout RMSE: `0.058211774395612294`; threshold:
  `0.3777528495512359`.
- Regularized precision condition number: `45.0073152832043`.
- Covariance condition number: `45.007315283204306`.
- Diagonal fallback was not used.

Gate status:

- `PASSED_MAP_LOCAL_GEOMETRY_HANDOFF_DIAGNOSTIC`

Next action:

- Review Phase 2T MAP-local reference handoff before runtime.  Phase 3
  GPU/XLA remains blocked.

### 2026-07-09 - Phase 2T - REVIEW_AND_RUNTIME

Review:

- Claude material review remains unavailable for repo context due the recovered
  external-transfer policy failure.  A fresh Codex substitute review was used
  and is weaker than full Claude material review.
- Round 1 returned `VERDICT: REVISE`.
- Repairs:
  - added explicit theta/z scale-transform checks;
  - excluded old Phase 1R summaries from every pass/fail field;
  - made the Phase 2U candidate grid, acceptance envelope, and selection policy
    exact.
- Focused round 2 returned `VERDICT: AGREE`.

Runtime:

- Implemented Phase 2T harness and tests.
- Initial runtime failed because the harness symmetrized `factor_z` before
  checking Cholesky reconstruction.  This was a harness bug, not scientific
  evidence against the MAP-local handoff.
- Repaired `_matrix(..., symmetrize=False)` for `factor_z` and added a
  non-diagonal factor test.
- Reran focused test:
  `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff.py`
  with `5 passed`.
- Final Phase 2T runtime passed and wrote:
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_cpu_hidden_2026-07-09.json`
  and
  `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_cpu_hidden_2026-07-09.md`.

Key diagnostics:

- `precision_z @ covariance_z` identity max error:
  `1.1254733403169899e-15`.
- `factor_z @ factor_z.T` reconstruction max error:
  `8.881784197001252e-16`.
- Precision theta scale-transform max error:
  `1.0000285044498014e-09`.
- Covariance theta scale-transform max error:
  `4.887283910903761e-10`.
- Phase 2U candidate contract:
  `(L, step) = (2, 0.785), (4, 0.3925), (8, 0.19625),
  (16, 0.098125)`, all with trajectory length `1.57`.

Gate status:

- `PASSED_MAP_LOCAL_REFERENCE_HANDOFF_DIAGNOSTIC`

Next action:

- Review Phase 2U retuned MAP-local HMC screen before runtime.  Phase 3
  GPU/XLA remains blocked.

### 2026-07-09 - Phase 2U - PRECHECK_AND_REVIEW

Evidence contract:

- Question: does the fixed equal-trajectory-length HMC grid in the MAP-local
  `u_new` coordinate produce at least one candidate with finite
  samples/telemetry and acceptance inside `(0.05, 0.99)`?
- Baseline/comparator: Phase 2S/2T MAP-local affine handoff only.  Old Phase 1R
  HMC summaries are excluded from pass/fail.
- Primary criterion: at least one candidate passes hard vetoes and acceptance;
  selection is the first passing candidate in the predeclared order.
- Vetoes: invalid Phase 2S/2T artifact, invalid MAP-local adapter, runtime
  error, nonfinite initial target/score, nonfinite samples/target/log accept,
  positive native divergence when available, no candidate passing acceptance,
  invalid artifact, or unsupported claim.
- Explanatory only: per-candidate acceptance, log-accept tails, target ranges,
  sample ranges, runtime, and native-divergence availability.
- Nonclaims: no posterior correctness, HMC readiness/convergence, zero
  divergences when telemetry is unavailable, statistical ranking, GPU/XLA
  readiness, default readiness, or Zhao-Cui source faithfulness.

Skeptical audit:

- Wrong baseline avoided: Phase 2U starts from Phase 2S/2T MAP-local geometry,
  not the old truth-free Phase 1R chain.
- Proxy promotion avoided: finite/acceptance evidence can select a next
  candidate only, not validate the posterior.
- Stop conditions and artifacts are explicit in the subplan.
- Environment mismatch controlled by CPU-hidden non-XLA manifest and nonclaims.
- Candidate grid and selection rule are fixed before runtime.

Review:

- Claude material review remains unavailable for repo context due the recovered
  external-transfer policy failure; a Codex substitute reviewer was used and is
  weaker than full Claude review.
- Substitute review returned `VERDICT: AGREE`.
- Reviewer checked transform orientation against
  `LatentAffineBatchValueScoreAdapter`: `factor = diag(scale) @ factor_z`
  follows the repository row-vector convention
  `theta = center + z @ factor.T`.  The subplan wording was tightened after
  review to avoid a misleading column-vector equivalence slogan.

Gate status:

- `PASSED_FOR_PHASE_2U_IMPLEMENTATION`

### 2026-07-09 - Phase 2U - IMPLEMENTATION_REPAIR_ROUND_1

Runtime attempt:

- The reviewed Phase 2U command ran the expensive HMC path but exited nonzero
  while writing the artifact.
- Quiet log showed `TypeError: Object of type EagerTensor is not JSON
  serializable`.
- No Phase 2U JSON/Markdown artifact was written.

Interpretation:

- This is a harness serialization bug, not candidate-gate evidence and not
  scientific evidence against the MAP-local target.

Repair:

- Patched Phase 2U `json_ready(...)` to convert TensorFlow tensors through
  `.numpy()`.
- Added a focused tensor-serialization unit test.
- Rerun focused tests before retrying the same reviewed Phase 2U runtime
  command unchanged.

### 2026-07-09 - Phase 2U - FINAL_RUNTIME_RESULT

Checks:

- `CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff.py`
  passed after serialization repair: `12 passed`.
- `git diff --check` passed.

Runtime:

- Reran the same reviewed Phase 2U command unchanged.
- Exit status: `0`.
- Wall time: `610.7140235070256` seconds.
- Wrote JSON/Markdown artifacts:
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_cpu_hidden_2026-07-09.json`
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_cpu_hidden_2026-07-09.md`

Gate:

- `phase2u_retuned_map_local_hmc_screen_passed=True`.
- Final vetoes: `[]`.
- Passed candidate count: `4 / 4`.
- Acceptance rates: `[0.34375, 0.546875, 0.96875, 0.984375]`.
- Selected candidate by first-passing policy:
  `(candidate_index=0, L=2, step_size=0.785, trajectory_length=1.57,
  acceptance=0.34375)`.
- Native divergence unavailable for all candidates; no zero-divergence claim.

Interpretation:

- Phase 2U nominates the selected candidate for a longer reviewed screen.
- It does not establish posterior correctness, HMC readiness, convergence,
  zero divergences, statistical ranking, GPU/XLA readiness, default readiness,
  or Zhao-Cui source faithfulness.

Next action:

- Review Phase 2V longer selected MAP-local screen subplan before runtime.
  Phase 3 GPU/XLA remains blocked.

### 2026-07-09 - Phase 2V - REVIEW_REPAIR_ROUND_1

Review:

- Codex substitute review returned `VERDICT: REVISE`.
- Findings:
  - Phase 2V did not explicitly fix the initial state to `u_new = 0`, the
    Phase 2U MAP-local center used by the selected candidate.
  - The next-phase handoff allowed either scalar reference/posterior agreement
    or Phase 3 GPU/XLA, which was too loose for a finite/acceptance screen.

Repairs:

- Added initial state `u_new = [0, 0, 0, 0]` to the fixed runtime settings,
  primary criterion, veto diagnostics, and stop conditions.
- Made the pass handoff exact: Phase 2V may only draft a reviewed scalar
  reference/posterior-agreement subplan.  GPU/XLA remains blocked until a later
  reviewed reference/posterior-agreement result explicitly authorizes a GPU/XLA
  reproduction subplan.

Gate status:

- `PENDING_CODEX_SUBSTITUTE_REVIEW_ROUND_2`
