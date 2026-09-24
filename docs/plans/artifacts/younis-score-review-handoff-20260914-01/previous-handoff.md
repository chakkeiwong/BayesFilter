# Claude handoff: full review of the KDM/LEDH model-score master program

Status: `READY_FOR_REVIEW`

Date: 2026-09-14

## Purpose

This memo requests a rigorous, read-only review of the master research program
for KDM, LEDH, UKF, GenUT, OT, smoothing, Fisher-score, IWSG, control-variate,
and degenerate-transition score estimators.

The review must determine whether the proposed program is mathematically sound,
scientifically motivated, internally consistent, executable in the stated
phase order, and covered by adequate tests. It must distinguish a missing
implementation from a bad research idea and a missing test from a failed test.

The central question is whether the program can investigate model-score
accuracy without silently replacing the observed-data score by a finite-program
derivative, a KDM expectation gradient, an unnormalised derivative, or a changed
filtering approximation.

## Role contract

Claude is a read-only reviewer. Claude must not edit files, run experiments,
launch agents, install packages, change configuration, or modify repository
state. Claude may read the exact paths requested in each stage and may request
one additional exact path or line range when a finding cannot be resolved from
the current stage.

Codex remains supervisor and executor. A Claude `REVISE` or `BLOCK` finding is a
repair specification; it is not permission to edit or execute. A Claude
`AGREE` verdict is valid only when the requested evidence was actually read.
Self-consistency of the plan is not enough for agreement.

## Primary review target

Read first, and initially read only:

`docs/plans/younis-kdm-score-master-program-2026-09-14.md`

This is the master program under review. Its current source is 791 lines. Do
not begin with a repository-wide search or a broad file bundle.

## Scientific boundary that must remain explicit

The review must preserve these distinctions:

1. The exact model score is
   \(S(\theta)=\nabla_\theta\log p_\theta(y_{1:T})\).
2. The finite-program derivative is the derivative of a declared finite scalar;
   it is not automatically the exact model score.
3. Younis--Sudderth IWSG is an expectation-gradient identity for
   \(\mathcal J(\theta)=\int F_\theta(z)m_\theta(z)\,dz\), with the current
   proposal held fixed during the local differentiation.
4. For \(F=\log g(y\mid z)\), the target is an expected conditional log
   likelihood, generally different from the log of the marginal likelihood.
5. Unnormalised likelihood and derivative estimators may each be unbiased while
   their finite ratio is biased.
6. A control variate requires a known or separately corrected centering
   expectation. Two biased estimators can be combined for lower MSE only after
   proving that they target the same score and estimating the combination on
   independent calibration data.
7. KDM filtering is not a smoothing theorem. Ordinary regular-transition
   smoothing and ambient Gaussian density ratios are not automatically valid for
   the degenerate DSGE target.
8. A better UKF/KDM covariance can improve proposal placement or variance; it
   does not by itself establish a better score or replace the filtering target.
9. OT can be a coupling, a deterministic cloud reset, a continuous resampling
   law, an explicit jitter proposal, or a model-changing finite operation. These
   roles require different mathematical claims and tests.

Any review finding that blurs one of these boundaries is a material finding.

## Review protocol

Use the following stages in order. Do not skip a stage because a later stage
appears easier. At the end of every stage, report findings in severity order and
end with exactly one verdict line.

### Stage 0: health and packet-read probes

Before substantive review, use the smallest probes:

```text
READ-ONLY HEALTH PROBE. Return exactly CLAUDE_PROBE_OK. Do not read files,
run commands, edit files, or launch agents.
```

Then:

```text
READ-ONLY PACKET-READ PROBE. Read exactly
docs/plans/younis-kdm-score-master-program-claude-review-handoff-2026-09-14.md
and return exactly P46_PACKET_READ. Do not read other files, run commands,
edit files, or launch agents.
```

If either probe fails, report the failure without attempting a broad review.

### Stage 1: master-program motivation and consistency review

Use this exact prompt after the probes:

```text
READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the
file itself explicitly asks you to inspect a cited line:
docs/plans/younis-kdm-score-master-program-2026-09-14.md
Do not edit, run commands, launch agents, or review the whole repo. Review the
program's motivation and internal consistency in detail. Check whether the
central research question is stated at the correct target, whether every option
in the matrix has a reason to exist and a declared failure mode, whether the
regular-transition and degenerate-transition tracks are genuinely separated,
whether the phase order and gates prevent target leakage, whether the program
distinguishes promotion criteria from explanatory diagnostics, whether the
heuristic-dominance and statistical-evidence requirements are usable, and
whether the proposed coordinator, registries, matrix generator, execution
lanes, metadata, and decision rules are sufficient to prevent unlabeled
comparisons. Check for duplicated baselines, missing baselines, unexamined
defaults, scope drift toward HMC or production claims, hidden tuning, unfair
compute comparisons, and conclusions stronger than the evidence contract.
For every finding give exact line references, severity, claimed intent,
problem, and a concrete repair. Explicitly distinguish a design omission from a
scientific contradiction. End with exactly VERDICT: AGREE or VERDICT: REVISE.
```

Required Stage 1 output:

| Field | Required answer |
|---|---|
| Research question | Is it a model-score question, a finite-program question, or both with labels? |
| Option coverage | Which discussed options are present, absent, duplicated, or misclassified? |
| Track separation | Are regular and degenerate claims isolated at every phase and artifact? |
| Gate logic | Can a candidate bypass an identity, support, tuning, heuristic, or uncertainty gate? |
| Scientific nonclaims | Are unsupported claims explicitly prohibited? |
| Architecture | Can the proposed registries and lanes enforce the target distinctions? |

### Stage 2: mathematical target and derivation review

Do not review all source material at once. Request these paths one at a time,
in this order, using the exact prompt shape below:

1. `docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex`
2. `docs/papers/ledh_younis_kdm_score/disturbance_score_proposal.tex`
3. `docs/plans/younis-model-score-analysis-2026-09-11.md`
4. `docs/plans/younis-degenerate-transition-correction-2026-09-12.md`
5. `docs/plans/degenerate-model-score-reassessment-2026-09-12.md`
6. `docs/plans/disturbance-score-proofs-2026-09-12.md`

For each path, use:

```text
READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the
file explicitly cites a line that must be checked:
<ONE EXACT PATH>
Do not edit, run commands, launch agents, or review the whole repo. Audit the
mathematical claims relevant to the KDM/LEDH score master program. Check
definitions, measures, conditioning, support, integrability, differentiation
under the integral, normalisation, dimensions, derivative scope, and whether
each proposition proves the target later assigned to it. In particular check:
(1) Fisher's identity and the initial-law term; (2) the fixed-proposal IWSG
expectation-gradient identity and its conditional scope; (3) the distinction
between expected log likelihood and log marginal likelihood; (4) unbiased
unnormalised value/derivative versus biased finite ratios; (5) linear
combinations of two biased estimators, covariance, coefficient fitting, and
centering; (6) physical transition numerators and evaluable proposal
denominators; (7) OT, jitter, Jacobian, and mixture-density requirements; (8)
UKF/KDM covariance interpretations; and (9) regular versus singular or
degenerate transition support. For every objection state the claimed target,
the quantity actually derived, and why they are equal, different, or not
checked. Do not treat a citation, MathDevMCP status, finite-difference check,
or numerical finiteness as a proof of a different claim. End with exactly
VERDICT: AGREE or VERDICT: REVISE.
```

The mathematical review must answer these discriminators explicitly:

- Does the IWSG proposition prove \(\nabla\mathcal J\), or the exact marginal
  score, under the stated assumptions?
- Does detaching the proposal describe a local sampling convention while
  retaining all upstream total derivatives required by the declared objective?
- Are proposal density, model transition density, mixture mass, and Jacobian
  kept separate in every corrected route?
- Is the finite score's random normalisation treated as a separate source of
  bias?
- Does a control variate preserve expectation only when its centering constant is
  known or separately corrected?
- Does a two-estimator combination require same-target matching before its MSE
  can be interpreted?
- Are smoothing and backward-pair arguments restricted to support classes where
  their conditional densities exist?
- Does the degenerate DSGE branch require a new base measure and score
  derivation rather than a full-rank Gaussian test?

### Stage 3: phase-prerequisite and gate audit

After Stages 1 and 2, review the master plan's phase sections one at a time.
Use one exact line range per prompt:

| Phase | Master-plan lines | Required focus |
|---|---:|---|
| Program architecture and registries | 60--152 | Are target, support, proposal, estimator, tuning, and artifact registries sufficient and enforceable? |
| Target and evidence contract | 153--330 | Are targets, primary criteria, vetoes, diagnostics, pre-mortem, and campaign envelope complete? |
| Option matrix | 331--384 | Are all options from the discussion represented with correct roles and scope? |
| Phase 0 | 385--420 | Are model catalogue, oracle ladder, and target metadata prerequisites concrete? |
| Phase 1 | 421--441 | Are identity and call-chain checks sufficient before quality comparisons? |
| Phase 2 | 442--497 | Are proposal, OT, covariance, and density-correction studies separated? |
| Phase 3 | 498--566 | Are fixed-cloud estimator, control-variate, and estimator-combination prerequisites complete? |
| Phase 4 and 4B | 567--609 | Are ratio, horizon, particle-count, and approximate-consistency studies identifiable? |
| Phase 5 | 610--629 | Are regular-transition smoothing and KDM claims properly restricted? |
| Phase 6 | 630--651 | Does the degenerate support program stop before invalid implementation? |
| Phase 7 | 652--666 | Are heuristic adversaries constructed and evaluated conditionally? |
| Phase 8 | 667--684 | Are tuning, replication, coupling, uncertainty, and compute fairness sufficient? |
| Phase 9 | 685--706 | Are implementation, provenance, backend, and production audits complete? |
| Exit gates and artifacts | 707--774 | Are prerequisites and terminal decisions explicit and non-circular? |
| First executable tranche | 775--791 | Does the first tranche answer the stated question and respect all prior gates? |

For each row, report a prerequisite ledger:

| Requirement | Must exist before phase | Exact planned artifact | Is it defined? | Is it executable? | Exit test | Status |
|---|---|---|---|---|---|---|
| Mathematical identity |  |  |  |  |  | `BUILT` / `NOT BUILT` / `INVALID` |
| Target and measure metadata |  |  |  |  |  |  |
| Oracle or oracle-status |  |  |  |  |  |  |
| Proposal density/support check |  |  |  |  |  |  |
| Derivative and call-chain check |  |  |  |  |  |  |
| Tuning artifact |  |  |  |  |  |  |
| Claim/validation split |  |  |  |  |  |  |
| Statistical uncertainty |  |  |  |  |  |  |
| Heuristic table |  |  |  |  |  |  |
| Manifest and immutable outputs |  |  |  |  |  |  |

The key question is whether each prerequisite is actually built before the
phase that consumes it. A sentence saying “implement an oracle” is not an
oracle artifact. A test filename is not evidence that the test covers the
required invariant. If a prerequisite is absent, mark it `NOT BUILT` and state
the smallest artifact and acceptance test needed before the phase can open.

### Stage 4: implementation and call-chain audit

Inspect the following exact paths one at a time, only after the plan and
mathematical reviews identify why each is needed:

- `bayesfilter/highdim/ledh_younis_kdm_tf.py`
- `bayesfilter/highdim/ledh_younis_kdm_resampling_tf.py`
- `bayesfilter/highdim/ledh_younis_kdm_integrated_tf.py`
- `bayesfilter/highdim/ledh_younis_kdm_lgssm_reference_tf.py`
- `bayesfilter/highdim/ledh_canonical_score_tf.py`
- `bayesfilter/highdim/ledh_canonical_filter_tf.py`
- `bayesfilter/highdim/ledh_ukf_lifecycle_tf.py`
- `bayesfilter/highdim/ledh_flow_perparticle_tf.py`
- `bayesfilter/highdim/ledh_contract_e_reset_tf.py`
- `bayesfilter/highdim/ledh_contract_e_streaming_tf.py`
- `bayesfilter/highdim/ledh_score_artifact.py`
- `bayesfilter/highdim/score_api.py`

Use this prompt for each path:

```text
READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless it
explicitly cites one required dependency:
<ONE EXACT PATH>
Do not edit, run commands, launch agents, or review the whole repo. Audit only
the call-chain obligations relevant to the KDM/LEDH score master program.
Determine which target the code actually computes, which measure and support it
uses, whether the proposal density and model numerator are both available,
whether all declared dependencies receive analytical total derivatives, and
whether batch/rank/dtype/device/XLA contracts are compatible with every claimed
consumer. Check for model-specific or lane-specific forks, identity placeholders,
missing UKF per-particle covariance lifecycle, omitted initial-law terms,
untracked OT/GenUT/reset dependence, and autodiff paths being mistaken for the
claim-bearing analytical route. Report exact symbol and line references. Mark
anything not checked as NOT CHECKED. End with exactly VERDICT: AGREE or
VERDICT: REVISE.
```

The audit must verify the complete call chain from each claim-bearing endpoint
to the claimed implementation. Existence of a capable helper function is not
evidence that the lane calls it.

### Stage 5: test-coverage audit

Inspect these exact test paths as relevant to each phase:

- `tests/highdim/test_ledh_younis_kdm_tf.py`
- `tests/highdim/test_ledh_younis_kdm_resampling_tf.py`
- `tests/highdim/test_ledh_younis_kdm_integrated_tf.py`
- `tests/highdim/test_ledh_younis_kdm_lgssm_reference_tf.py`
- `tests/highdim/test_ledh_canonical_score_recursion.py`
- `tests/highdim/test_ledh_canonical_score_stages.py`
- `tests/highdim/test_ledh_canonical_total_gaussian_tf.py`
- `tests/highdim/test_ledh_score_wiring_phase8_cross_model.py`
- `tests/highdim/test_ledh_tuning_scope.py`
- `tests/highdim/test_ledh_canonical_meta_governance.py`
- `tests/highdim/test_ledh_pfpf_genut_initialization_designs.py`
- `tests/highdim/test_transport.py`
- `tests/test_ledh_pfpf_ot_manual_adjoint_primitives.py`
- `tests/test_ledh_no_tape_total_sinkhorn_vjp_phase1.py`
- `tests/test_ledh_no_tape_total_sinkhorn_vjp_phase2.py`
- `tests/highdim/test_ledh_kalman_oracle_tf.py`
- `tests/highdim/test_model_agnostic_score_opg_lgssm_witness.py`

Do not infer coverage from filenames. For every phase, construct a test matrix
with these columns:

| Invariant | Required test type | Existing test/path and symbol | Direct or indirect? | Boundary case | Negative case | Stochastic replication | Missing test |
|---|---|---|---|---|---|---|---|

Use the following minimum coverage standard.

#### Phase 0 tests

- exact value and score oracle on linear-Gaussian models;
- independent directional derivative and finite-difference checks;
- parameter-dependent initial-law checks;
- metadata validation for target, measure, support, normalisation, and oracle;
- an oracle-free model marked diagnostic rather than silently admitted.

#### Phase 1 tests

- same-finite-scalar analytical derivative parity;
- omitted-term tests for initial, transition, observation, covariance,
  bandwidth, mixture, Jacobian, OT, and reset dependencies;
- complete call-chain/wiring tests;
- support and positivity failures;
- rank, batch, dtype, device, and XLA contract checks.

#### Phase 2 tests

- proposal sampler versus evaluated proposal-density parity;
- physical transition numerator and proposal denominator separation;
- UKF, KDM, LEDH, GenUT, and hybrid covariance lifecycle tests;
- OT marginal balance, coupling, deterministic reset, jitter, Jacobian, and
  evaluable-density cases as separate tests;
- changed-target detection for non-measure-preserving resets.

#### Phase 3 tests

- fixed-cloud IWSG expectation-gradient identity on a tractable mixture;
- expected-log versus log-expected-log distinction;
- unnormalised value/derivative versus finite ratio;
- forward Fisher and backward-pair agreement where valid;
- exact control-variate centering and deliberately wrong-centering failure;
- two-estimator combination with known biases, covariance, coefficient fitting,
  calibration/claim separation, and same-target mismatch rejection.

#### Phase 4 and 4B tests

- denominator positivity and reciprocal-tail diagnostics;
- particle-count and horizon ladders with coupled seeds;
- ratio-bias witness;
- diagnostic/bias correlation calibrated on oracle-bearing models and evaluated
  on held-out models;
- uncertainty intervals wide enough to prevent descriptive differences being
  reported as rankings.

#### Phase 5 tests

- regular-transition support and common-measure checks;
- adjacent-state pair and backward-weight identities;
- model-corrected mixture normalisation;
- proposal-only KDM versus changed-target positive-bandwidth KDM;
- untouched multi-dataset retest against the same oracle.

#### Phase 6 tests

- ancestor/innovation base-measure derivation;
- singular-support and ancestor-collapse fixtures;
- parameter-dependent deterministic-map/Jacobian checks;
- rejection of ambient Gaussian density ratios when invalid;
- explicit unresolved status when no support-aware oracle exists.

#### Phases 7--9 tests

- constructed heuristic adversaries evaluated by salient regime;
- scope-specific tuning artifact identity and data partition checks;
- paired uncertainty and replication checks;
- manifest, provenance, memory-growth, chunk-policy, and backend checks;
- rendered-document and claim/nonclaim consistency checks.

For every missing test, state whether it blocks implementation, blocks a phase
gate, or is an explanatory enhancement. “There are many tests” is not an
adequate coverage argument.

### Stage 6: integrated readiness and red-team review

After all prior stages, request a final synthesis using this exact prompt:

```text
READ-ONLY BOUNDED REVIEW. Review the previously returned findings for
docs/plans/younis-kdm-score-master-program-2026-09-14.md and the exact paths
that were subsequently requested. Do not edit, run commands, launch agents, or
review additional files. Produce an integrated red-team assessment. Reconcile
the mathematical target review, motivation review, phase prerequisites,
implementation call chain, and test-coverage matrix. Identify any contradiction
between the master plan and the supporting manuscript or existing code. State
whether each phase is READY, NOT READY, or INVALID, and name the exact missing
artifact or test for every NOT READY phase. Separate implementation failure,
tuning failure, diagnostic failure, finite-normalisation failure,
support/measure failure, and evidence insufficiency. Check that no regular-
transition result can be promoted as a DSGE result and that no KDM expectation
gradient can enter a marginal-score leaderboard without a target proof. State
the smallest safe first executable tranche and the conditions for opening each
later phase. End with exactly VERDICT: AGREE or VERDICT: REVISE.
```

## Required final report

The final Claude report must use this structure.

### 1. Verdict summary

State one of:

- `AGREE`: the program is mathematically and scientifically coherent, every
  phase has an adequate prerequisite/test path or an explicitly diagnostic
  status, and no material target or support contradiction remains;
- `REVISE`: the direction is viable but one or more concrete repairs are needed;
- `BLOCK`: a mathematical target, support, or evidence contradiction prevents
  execution until resolved.

The terminal line must be exactly:

```text
VERDICT: AGREE
```

or

```text
VERDICT: REVISE
```

Use `VERDICT: REVISE` for both ordinary revisions and a material blocker, while
labeling the blocker severity explicitly in the body.

### 2. Findings by severity

Use this table:

| ID | Severity | Exact path:line | Category | Claimed target | Actual issue | Required repair | Blocks phase? |
|---|---|---|---|---|---|---|---|

Severity categories:

- `BLOCK_MATH`: proposition, derivative, measure, support, or normalization is
  wrong relative to its stated target;
- `BLOCK_TARGET`: computed quantity and claimed quantity differ without a
  declared approximation;
- `BLOCK_SUPPORT`: regular-transition identity is used without a valid support
  or base-measure derivation;
- `BLOCK_PREREQ`: a phase consumes an artifact or proof that is not built;
- `BLOCK_COVERAGE`: required invariant lacks a direct test or negative case;
- `BLOCK_EVIDENCE`: comparison cannot support its proposed claim;
- `REVISE_DESIGN`: repairable program inconsistency or missing option;
- `ADVISORY`: useful improvement that does not block the declared tranche.

### 3. Phase readiness table

| Phase | Scientific question | Mathematical prerequisite | Implementation prerequisite | Required tests | Artifact | Status | Reason |
|---|---|---|---|---|---|---|---|

Use `READY`, `NOT READY`, or `INVALID`. Do not use “mostly ready.”

### 4. Target and option matrix

For every candidate, report:

| Candidate | Declared target | Actual target checked? | Support class | Proposal correction | Score identity | Variance method | Oracle/comparator | Promotion status |
|---|---|---|---|---|---|---|---|---|

### 5. Test-coverage matrix

Return the invariant-to-test matrix from Stage 5, including explicit missing
tests and whether each omission blocks a phase.

### 6. Minimal repair sequence

Give a numbered sequence of the smallest repairs needed before the first
executable tranche, then the conditions that open Phases 2--9. Do not recommend
launching a large factorial sweep before identity and oracle gates pass.

### 7. Explicit nonclaims

State what the review does not establish, including posterior correctness,
finite-sample unbiasedness beyond the proved identity, HMC readiness, default
readiness, DSGE validity, or superiority outside the tested scope.

## Review-specific nonclaims

Even if Claude returns `VERDICT: AGREE`, the review does not itself prove that
any estimator is statistically superior, that a KDM proposal improves the
score, that OT preserves a target measure, or that the degenerate DSGE score
has been solved. Those require the executable phases and their declared
artifacts.

## Evidence available before the review

The master plan is design-only. No new campaign has been executed under it.
Relevant existing documents are cited in the plan and listed in Stage 2. Any
existing numerical result must be treated as historical or scope-specific
unless the reviewer verifies its target, support, tuning scope, uncertainty,
and provenance. Do not use an old positive result as a prerequisite merely
because it has a finite score or a passing smoke test.

## Review recovery rule

If a broad stage hangs or returns no output, preserve the failed attempt and
split the stage by the exact path or line range that failed. Do not respond to a
timeout by sending the whole repository or a large pasted code bundle. A
reviewer timeout is a review-process limitation, not scientific evidence that
the program passes or fails.
