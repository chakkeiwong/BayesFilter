# HMC Tuning Guide and Policy Repair Plan

Date: 2026-09-01  
Status: `COMPLETED_WITH_SCOPED_RESIDUALS`  
Owner scope: BayesFilter HMC tuning documentation, fixed-transport tuning
policy, and focused regression tests.

## Purpose

The Phase 9A q=20 preflight exposed a policy-level failure: the active
fixed-transport tuner treated a short acceptance screen and a multiplicative
epsilon repair as if they were sufficient evidence for a useful HMC tuning
decision.  The endpoint target and Jacobian score were not shown to be wrong.
The problem is that the guide, executable policy, and tests do not enforce the
same distinction between mechanics validity and exploration efficiency.

This plan repairs the guide first and then makes the shared implementation obey
the repaired guide.  Existing Phase 9A artifacts remain preserved historical
or diagnostic evidence and are not overwritten or upgraded.

## Research and engineering intent

**Question.** Can BayesFilter prevent a high-acceptance, resonant, or otherwise
under-tested HMC candidate from being presented as a well-tuned sampler?

**Mechanism under repair.** The fixed-transport HMC tuning interface and its
candidate-selection route.

**Expected failure mode.** At fixed finite leapfrog count, acceptance can be
non-monotone in epsilon.  A nearly Gaussian endpoint can have acceptance near
one even when trajectory movement and effective sampling are poor.  Short,
central-start screens can make this look healthy.

**Primary pass criteria.**

1. The Markdown guide and normative LaTeX chapter state the correct target,
   acceptance semantics, resonance limitation, and authority boundaries.
2. The claim-bearing fixed-transport route cannot use unmeasured directional
   epsilon repair or a single short acceptance observation as its tuning
   decision.
3. The implementation records mean Metropolis probability and binary
   acceptance separately, and reports movement, trajectory time, energy/error
   and divergence provenance without conflating adaptive and fixed runs. In
   the measured route, acceptance is descriptive/a repair evidence rather than
   a hard validity veto; finite transition, telemetry, divergence, and retained
   movement checks own validity.
4. Analytic and adversarial fixtures demonstrate that the policy rejects or
   downgrades a resonant high-acceptance candidate.
5. Existing target and transport mathematics are unchanged; all focused tests,
   import/compile checks, and documentation-generation checks pass.

**Promotion vetoes.** Any target/value-score mismatch, non-finite transition,
missing identity or telemetry, hidden scalar/pfor path, test failure,
unreviewed API default change, or artifact written over an existing path stops
the relevant phase.

**Continuation vetoes.** Stop the campaign only for a corrupted worktree,
missing required dependency, exhausted bounded repair budget, or a change that
would alter the scientific target, privacy boundary, hardware class, or public
product direction.  A failed candidate is not a reason to reject the HMC
research direction.

**Nonclaims.** This repair does not establish posterior convergence, mode
discovery, sampler superiority, target correctness for every model, HMC
readiness, or high-dimensional scaling.  Those require later target-specific
campaigns.

## Evidence contract

| Item | Declared contract |
|---|---|
| Question | Does the guide and shared tuner prevent false tuning conclusions caused by high acceptance and fixed- L resonance? |
| Baseline | Current `tune_fixed_transport_hmc_kernel` behavior and the preserved Phase 9A attempt-05 record. |
| Primary criterion | Policy/documentation/tests agree that candidate selection is based on measured joint `(epsilon, L)` evidence and explicit efficiency/validity diagnostics, not directional inference. |
| Hard veto diagnostics | Target/score identity, finite values, transition telemetry, divergence provenance, reproducible candidate identity, test/build failures. |
| Explanatory diagnostics | Acceptance probability, binary acceptance, energy error, displacement, ESJD, trajectory time, retracing, runtime, and seed variation unless explicitly promoted by a later plan. |
| Not concluded | No posterior or scientific claim follows from this repair or its unit fixtures. |
| Preserved artifact | This plan, a repair result note, a reset/migration memo, updated guide/chapter, code diff, and focused test output. |

## Default and assumption audit

| Choice | Provenance | Justification | Failure mode | Earliest diagnostic | Status |
|---|---|---|---|---|---|
| Mean Metropolis probability is reported | Existing mechanics implementation | It is the quantity used by TFP dual averaging | It can be mistaken for binary acceptance or mixing | Compare both statistics on a known Gaussian | Existing semantic; documentation repair required |
| Target acceptance 0.70 | Existing repository policy | A tuning target, not a correctness theorem | It can become a false universal objective | Analytic Gaussian/resonance curve and efficiency comparison | Baseline hypothesis |
| Fixed- L directional repair | Existing tuner and guide | Local small-step intuition | Fails under non-monotone resonance | Harmonic-oscillator fixture | Retire for claim-bearing use |
| Joint `(epsilon, L)` measured candidates | Derived from the failure analysis and existing grid helpers | Directly observes the variables that determine trajectory phase | More compute and requires a bounded grid | Candidate-count and budget checks | Active policy after focused tests |
| Short CPU-hidden tests | Routine implementation verification | No scientific campaign or GPU claim is needed for policy repair | May miss device-specific behavior | TensorFlow import/compile and analytic fixtures | Diagnostic only |
| No GPU run in this plan | Scope decision | Guide/code policy can be tested without a model campaign | GPU-specific integration remains untested | Later GPU preflight under a separate plan | Explicit nonclaim |

## Allow-list and approval boundary

The user request authorizes this local repair campaign.  Under the repository
governance profile, no additional scientific approval, hash token, per-command
click, network access, package installation, external message, destructive Git
operation, or GPU launch is needed.

Commands in scope are read-only inspection, `apply_patch`, Python compile and
focused CPU-hidden tests, documentation rendering/checks, and `git diff`/
`git status`.  A narrow persistent command rule may be added for a repository
test wrapper if the host requires it:

`bash /home/ubuntu/python/BayesFilter/scripts/run_hmc_tuning_policy_tests.sh`

The relative form is equivalent when the working directory is the repository
root. For the later q=20 GPU campaign, the separate service rule should match
the exact repository launcher (with physical GPU 0 exposed):

`bash /home/ubuntu/python/BayesFilter/scripts/run_ssl_lstm_q20_tempered_rkl_phase8_gpu_default.sh`

No broad `bash`, `python`, `codex`, package-manager, or arbitrary GPU allow-list
entry is requested.  A future GPU campaign would require one separate trusted
approval for its bounded launcher, with `TF_FORCE_GPU_ALLOW_GROWTH=true` and
pre-import memory-growth verification; that campaign is outside this plan.

## Phases and between-phase repair

### Phase 0: Inventory and baseline (completed before execution)

- Read the guide, normative chapter, active tuner, registry, tests, and Phase 9A
  result.
- Confirm no process from the interrupted run remains.
- Preserve all unrelated dirty worktree changes.
- Record the specific mismatch: the guide acknowledges resonance, but the
  executable factor-of-two repair and monotone fake tests do not enforce that
  warning.

**Repair review.** Verify paths and line anchors still exist; if not, refresh
the plan before editing.

### Phase 1: Repair the guide and mathematical contract

- Define (a(\epsilon,L)) as an expected Metropolis probability and distinguish
  it from binary acceptance.
- State explicitly that (a(\epsilon,L)) is not generally monotone in
  epsilon for finite (L), with the harmonic-oscillator phase explanation.
- State that target acceptance is an efficiency heuristic, not convergence or
  correctness evidence.
- Require joint measured `(epsilon, L)` candidates, representative starts,
  independent fixed-kernel validation, uncertainty, and efficiency diagnostics.
- Separate `mechanics_validated`, `tuning_candidate`, and
  `posterior_ready` statuses.
- Mark short preflight and directional-repair routes diagnostic-only.

**Repair review.** Red-team every sentence against the current implementation;
remove any promise the code does not yet satisfy.  Run the documentation
contract only after the prose and LaTeX agree.

### Phase 2: Implement guide-directed policy guardrails

- Add `tuning_policy="measured_joint_grid_v1"` as the claim-bearing default,
  plus an explicit `legacy_directional_diagnostic_v1` escape hatch. The latter
  may emit a diagnostic record but cannot produce an artifact-authority
  handoff.
- Add an explicit `step_size_candidates` grid (with a bounded
  `max_joint_candidate_count`). The measured policy evaluates every declared
  `(epsilon, L)` pair, including all declared leapfrog values; it never treats
  a factor-of-two neighbor as observed evidence. The legacy fixed-grid fields
  remain readable only as a diagnostic migration path.
- Reject a measured grid containing a step above the configured hard cap before
  launching any chain. A pre-vetoed pair is not labelled measured, so the
  all-pairs evidence field remains truthful.
- Require the measured policy to have at least two distinct step sizes, at
  least two distinct leapfrog counts, replicated efficiency selection, and a
  disjoint held-out fixed-kernel verification. These are validity checks for
  the tuning decision, not posterior-convergence claims.
- Retain the old directional ladder only behind the explicit diagnostic policy
  and stamp its result `diagnostic_only_legacy_policy`; it must not be accepted
  by the verified-handoff builder or route authority.
- Add hard validation for candidate count, distinct leapfrog values, explicit
  diagnostic roles, and step-size caps; do not silently widen caps or alter the
  target.
- Correct movement telemetry so the initial-to-first-retained displacement from
  an adaptive run is not presented as fixed-kernel efficiency.
- Add explicit payload status fields for `mechanics_validated`,
  `tuning_candidate`, and `posterior_ready` (the last remains false here), and
  bump result/kernel/selection schemas so old artifacts cannot be silently
  interpreted under the repaired contract.

**Repair review.** Check API compatibility, source-closure identity, route
registry consistency, and that no consumer can silently receive a legacy
candidate as a claim-bearing handoff.

### Phase 3: Analytic and adversarial regression tests

- Add a standard-Gaussian affine-transport fixture.
- Add a harmonic-oscillator/fixed- L acceptance curve with a deliberate
  non-monotone region.
- Test mean-probability versus binary-acceptance semantics.
- Test that a high-acceptance, low-movement candidate is not promoted by
  acceptance alone.
- Test that a high-acceptance candidate with valid movement is not rejected
  solely for missing the target-acceptance band; efficiency and health remain
  the selection evidence.
- Test that a failed measured grid is reported as a tuning-candidate failure,
  not as target invalidity.
- Test that legacy directional configuration cannot create a verified handoff,
  and that a single-step or single-`L` measured grid fails before any chain
  runs.
- Retain existing target/score finite-difference and XLA parity tests.

**Repair review.** Ensure fixtures test real policy behavior rather than only
  fake monotone callbacks; verify no NumPy computation enters runtime paths.

### Phase 4: Integration and documentation consistency

- Update examples and route registry payloads to the new policy identity.
- Migrate the active SSL-LSTM tuning configuration to an explicit measured
  grid; leave the Phase 9B claim-bearing run closed until that fresh grid has
  passed its own target-specific evidence gates.
- Ensure the active q=20 preflight records the measured-grid policy and does
  not describe a cap-excluded pair as measured. Other historical callers are
  either migrated with an explicit target-specific grid or marked
  diagnostic-only; no implicit one-dimensional grid remains claim-bearing.
- Regenerate the route table and run the documentation contract.
- Add a migration note pointing old artifacts and callers to the new policy.
- Refresh the SSL-LSTM Phase 9A boundary so Phase 9B remains closed until a
  fresh policy-compliant repair is executed.

**Repair review.** Search for stale instructions, copied factor-of-two repairs,
  single-`L` serious configs, and claims that equate artifact authority with
  posterior readiness.

### Phase 5: Verification and closeout

- Run `git diff --check`, Python compilation, focused CPU-hidden tests, and
  documentation rendering/checks.
- Record exact commands, environment, seeds, wall time, and artifacts.
- State whether failures concern implementation, tuning, diagnostics, or the
  scientific idea.
- Write a result note and reset memo with the next smallest valid SSL-LSTM
  tuning subplan.

**Terminal review.** Confirm that no claim-bearing GPU/HMC campaign was launched
  from an unreviewed policy and that all prior evidence remains immutable.

## Pre-mortem

The repair could appear successful while still failing if the new tests use only
monotone synthetic acceptance, if the route registry labels a legacy branch
active, if the candidate grid is measured but selection still uses acceptance
alone, if adaptive and fixed telemetry remain mixed, or if a short CPU fixture
is mistaken for model validation.  The earliest checks are the harmonic-
oscillator fixture, a source scan for directional repair in active code, a
registry/status assertion, and an explicit nonclaim check in the result.

The repair could fail for ordinary engineering reasons (API consumers,
TensorFlow tracing, or documentation generation) rather than because the policy
is wrong.  Such failures trigger localized fixes and focused regressions under
the same budget; they do not justify relaxing the mathematical contract.

## Execution record

This file is the controlling plan for the repair.  Phase status, commands,
failures, repairs, and artifacts will be appended below as execution proceeds.

### Skeptical audit before execution

Audit status: `PASS_WITH_REQUIRED_REPAIRS_RECORDED`.

The plan uses the correct Phase 9A baseline, does not treat acceptance as a
posterior criterion, names the fixed- L non-monotonicity, preserves a separate
diagnostic role for failed candidates, and has explicit stop conditions.  The
main implementation risk is API migration; it is bounded in Phase 2 and will
be stopped if it would silently change unrelated routes.  No long experiment
or GPU action is authorized by this plan.

### Skeptical re-audit after the first implementation pass

The first pass exposed two material contract gaps before closeout. (A1) The
measured branch still treated the narrow acceptance band as a hard veto even
though the guide called acceptance a heuristic; this could reject an efficient
high-acceptance pair for the wrong reason. (A2) A pair above the step cap was
pre-vetoed but marked `measured`, making the all-pairs evidence field false.
The plan therefore remains open and is amended as follows: acceptance is hard
only when missing or non-finite (or on the explicitly legacy diagnostic branch),
over-cap measured grids fail configuration before execution, and schema/status
fields are made explicit. These are local repairs to the same scientific
contract, not a new target or campaign.

### Execution status after implementation

Phase 0: `COMPLETED`. The Phase 9A attempt-05 record was preserved as the
baseline failure; no prior result was overwritten or promoted.

Phase 1: `COMPLETED`. The Markdown guide and normative LaTeX chapter now state
the transformed-target contract, mean-probability versus binary-acceptance
semantics, fixed-L resonance, the measured joint-grid requirement, and the
authority/status boundaries.

Phase 2: `COMPLETED`. The shared fixed-transport tuner implements
`measured_joint_grid_v1`, validates the complete bounded grid before execution,
records all attempted pairs, uses replicated efficiency selection plus
disjoint held-out verification, and quarantines the legacy directional route.
Result/kernel/selection schemas were bumped to prevent silent interpretation of
old artifacts. The q=20 preflight and HNN caller were migrated; the July LGSSM
caller is explicitly legacy diagnostic-only.

Phase 3: `COMPLETED`. Harmonic-phase, high-acceptance, movement, over-cap,
legacy-handoff, target-health, and held-out fixtures were added or repaired.
The stale oracle fixture that treated out-of-band acceptance as a hard failure
was corrected to inject a genuine non-finite held-out target failure.

Phase 4: `COMPLETED_WITH_HISTORICAL_CALLER_NOTE`. The route table and docs
contract pass. The old LGSSM archive referenced by its historical test is
absent; that test remains an artifact-availability failure and is not used as
evidence for this repair. Its caller cannot issue a claim-bearing handoff.

Phase 5: `COMPLETED`. Exact commands, environment, test counts, failure
classification, decision tables, and the next q=20 action are recorded in
`bayesfilter-hmc-tuning-guide-policy-repair-result-2026-09-01.md` and the
corresponding reset memo.

The independent ordinary-tuner oracle regression was rerun during closeout.
It remains a scoped residual (`budget_exhausted` from the verification R-hat
cap, no hard veto) because its historical assertion expects `passed=True`.
The ordinary tuner was not modified here; investigate that fixture under a
separate plan rather than weakening this guide's measured fixed-transport
contract.

### Final skeptical review

Status: `PASS_WITH_EXPLICIT_NONCLAIMS`.

The active route no longer infers an unmeasured step from acceptance, and a
finite acceptance value outside the target band cannot by itself create a hard
failure. The over-cap pre-veto occurs before any chain and is not counted as a
measurement. Legacy records are rejected at the handoff boundary. No GPU,
posterior, or default-readiness claim was made. Remaining uncertainty is
target-specific q=20 behavior, which requires a separate bounded campaign.
