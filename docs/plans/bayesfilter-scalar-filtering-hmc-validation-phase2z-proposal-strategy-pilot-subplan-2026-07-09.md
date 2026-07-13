# Phase 2Z Subplan: Proposal Strategy Pilot

Date: 2026-07-09
Status: `REVIEWED_READY_PENDING_RUNTIME_CLEARANCE`

## Phase Objective

Use the Phase 2Y localization result to choose and pilot a small set of
importance proposal strategies that are plausible for the scalar SSL-LSTM
MAP-local `u_new` target.  This phase may nominate a proposal family for an
independent reference-replication phase or may recommend abandoning
self-normalized importance sampling for this target.

This phase is a proposal-strategy pilot.  It is not an HMC run, not a valid
posterior reference by itself, not posterior certification, not HMC readiness,
not convergence evidence, not a GPU/XLA/default-readiness phase, and not
Zhao-Cui source-faithfulness evidence.

## Entry Conditions

- Phase 2W failed only reference ESS/ESS-ratio gates with finite target and
  proposal evaluations.
- Phase 2X failed only reference ESS/ESS-ratio gates with finite target and
  proposal evaluations.
- Phase 2Y passed diagnostic validity and recorded:
  - no affine orientation or proposal-log-density replay bug;
  - top-weight anchors outside the Phase 2S trust radius;
  - strong target-minus-quadratic residuals along rays;
  - proposal-family mismatch indicated.
- Phase 3 GPU/XLA remains blocked.

## Required Artifacts

- Phase 2Z harness:
  `docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_2026_07_09.py`
- Phase 2Z tests:
  `tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot.py`
- Phase 2Z JSON/Markdown artifacts:
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_cpu_hidden_2026-07-09.json`
  - `docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_cpu_hidden_2026-07-09.md`
- Quiet log:
  `docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2z_proposal_strategy_pilot.log`
- Phase 2Z result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2z-proposal-strategy-pilot-result-2026-07-09.md`

## Required Checks, Tests, And Reviews

- Review this subplan before runtime.  Claude may be attempted only through the
  approved review gate.  If the approval layer blocks external Claude transfer,
  record that explicitly and use a fresh local Codex substitute review, weaker
  than Claude.
- Run focused tests before runtime:

```bash
CUDA_VISIBLE_DEVICES=-1 pytest -q tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot.py tests/test_scalar_ssl_lstm_filtering_hmc_validation_phase2y_target_geometry_localization.py
```

- Run `git diff --check`.
- Create the quiet log directory before redirected runtime.
- Planned runtime command:

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 timeout 420 python docs/benchmarks/benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_2026_07_09.py --json-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_cpu_hidden_2026-07-09.json --markdown-path docs/benchmarks/scalar_ssl_lstm_filtering_hmc_validation_phase2z_proposal_strategy_pilot_cpu_hidden_2026-07-09.md > docs/benchmarks/logs/bayesfilter_scalar_filtering_hmc_validation_2026-07-09/phase2z_proposal_strategy_pilot.log 2>&1
```

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Is there a proposal family worth taking to an independent reference-replication phase, or should the SNIS reference branch be abandoned for this target? |
| Baseline/comparator | Failed Phase 2W standard normal, failed Phase 2X shifted diagonal mixture, and Phase 2Y target-geometry diagnostics. |
| Primary pass criterion | The harness evaluates each predeclared proposal candidate with finite target/proposal/log-weight values and records ESS/ESS-ratio, top-weight concentration, and proposal provenance.  A candidate is only nominated if it passes pilot ESS gates without artifact bugs. |
| Pilot nomination screen | ESS `>= 256`, ESS ratio `>= 0.05`, max normalized weight `<= 0.05`, finite target/proposal/log weights, no proposal replay mismatch, and no use of HMC sample moments for tuning. |
| Veto diagnostics | Invalid Phase 2W/2X/2Y artifacts, nonfinite target/proposal/log weights, proposal sample shape mismatch, proposal density replay mismatch, using Phase 2V HMC moments to tune proposals, post-hoc candidate changes, unsupported reference/HMC/GPU/default/scientific claim, or timeout. |
| Explanatory diagnostics | Per-candidate ESS/ESS-ratio, max weight, weighted means/stds, top-weight coordinates, proposal component labels, target/proposal log-density summaries, runtime, and whether anchors from Phase 2Y explain remaining degeneracy. |
| Not concluded | A nominated proposal is not an independent valid reference, not HMC-vs-reference agreement, not posterior correctness, not HMC readiness/convergence, not zero-divergence evidence, not sampler superiority, not a statistical ranking, not GPU/XLA readiness, not default readiness, and not Zhao-Cui source faithfulness. |
| Artifact preserving result | Phase 2Z JSON, Markdown, quiet log, result file, and refreshed handoff. |

## Fixed Proposal Candidates

Candidates must be generated without using Phase 2V HMC moments.

1. `student_t_centered`: independent Student-t in `u_new`, centered at zero,
   degrees of freedom `4`, diagonal scales from the Phase 2X pilot standard
   deviations clipped to `[1.5, 4.0]`.
2. `student_t_shifted`: independent Student-t centered at the Phase 2W pilot
   mean, degrees of freedom `4`, same clipped scales.
3. `anchor_mixture_student_t`: mixture of centered Student-t, shifted Student-t,
   and low-weight Student-t components centered at Phase 2Y top-weight anchors.
   Anchor centers are allowed only because this is a pilot-nomination phase;
   any nominated result must be replicated in a later independent phase that
   does not reuse these same anchors as validation evidence.
4. `ridge_line_student_t`: mixture over line centers along the top Phase 2Y
   ray directions at radii `{0, 2.5, 5.0}` with Student-t local noise.  This is
   a pilot candidate only.

Each candidate must use deterministic seeds distinct from Phase 2W/2X and must
record proposal parameters, component counts, log-density formula, and sample
count.  Use a modest CPU-hidden pilot size.  The initial reviewed default was
`4096`, but the first runtime attempt timed out before writing an artifact.
Under the predeclared timeout-feasibility escape hatch, repair the pilot size to
`1024` per candidate for this CPU-hidden pilot.  This repair does not change
candidate families, seeds, proposal formulas, nomination thresholds, or claim
boundaries.

Fixed candidate parameterization:

- `student_t_centered` uses one component with weight `1.0`, center
  `[0, 0, 0, 0]`, scale `clip(phase2x_reference_std, 1.5, 4.0)`, degrees of
  freedom `4`, and seed `(20260709, 6701)`.
- `student_t_shifted` uses one component with weight `1.0`, center equal to the
  Phase 2W pilot mean, scale `clip(phase2x_reference_std, 1.5, 4.0)`, degrees
  of freedom `4`, and seed `(20260709, 6702)`.
- `anchor_mixture_student_t` uses weights `0.25` centered, `0.25` shifted, and
  `0.50` split equally across the unique Phase 2Y top-weight anchors from
  Phase 2W and Phase 2X.  It uses degrees of freedom `4`, local scale
  `clip(0.75 * phase2x_reference_std, 1.0, 3.0)`, and seed `(20260709, 6703)`.
- `ridge_line_student_t` uses weights split equally across centers formed by
  the unique Phase 2Y top-weight ray directions at radii `{0, 2.5, 5.0}`.  It
  uses degrees of freedom `4`, local scale
  `clip(0.75 * phase2x_reference_std, 1.0, 3.0)`, and seed `(20260709, 6704)`.
- Proposal sample counts are deterministic multinomial-by-rounding allocations:
  floor each `weight * 1024`, assign the remainder to largest fractional
  weights with stable component-index tie-breaks, and record the final component
  counts.  The log density must use the intended mixture weights, not the
  rounded sample fractions.

## Forbidden Claims And Actions

- Do not run HMC or GPU/XLA in Phase 2Z.
- Do not change defaults or public API behavior.
- Do not claim a valid independent reference from Phase 2Z.
- Do not interpret HMC-vs-reference agreement.
- Do not use Phase 2V HMC samples or moments to tune proposals.
- Do not treat unavailable native divergence telemetry as zero divergences.
- Do not claim posterior correctness, HMC readiness/convergence, sampler
  superiority, statistically supported ranking, default readiness, or
  Zhao-Cui source faithfulness.

## Exact Next-Phase Handoff Conditions

If at least one candidate passes the pilot nomination screen, write the Phase
2Z result and draft a Phase 2ZA independent reference-replication subplan.  The
replication subplan must use fresh seeds and must not reuse Phase 2Y top
anchors as validation evidence.

If no candidate passes but all evaluations are finite and artifact replay is
valid, write the Phase 2Z result and draft a reviewed decision subplan to
abandon SNIS reference agreement for this target or move to a transport or
sequential reference method.

If a proposal-log-density or transform replay bug appears, write a bug-repair
subplan before any further reference attempt.

## Stop Conditions

Stop for invalid source artifacts, nonfinite target/proposal/log weights,
proposal replay mismatch, use of HMC moments for proposal tuning, timeout,
review nonconvergence, or any need to cross HMC-runtime, GPU, default-policy,
model-file, source-faithfulness, product, or scientific-claim boundaries.

## Skeptical Plan Audit

| Risk | Audit finding |
| --- | --- |
| Wrong baseline | Baselines are the failed Phase 2W/2X proposals and Phase 2Y diagnostic evidence, not HMC success. |
| Proxy metrics promoted | Phase 2Z ESS screens can nominate only; independent replication is required before any reference validity or HMC agreement interpretation. |
| Missing stop conditions | Artifact validity, finite values, proposal replay, HMC-moment exclusion, review, timeout, and claim-boundary stops are explicit. |
| Unfair comparison | Candidate ranking is descriptive only unless a later replication with uncertainty justifies ranking. |
| Hidden assumptions | Student-t and anchor/ridge proposals are hypotheses driven by Phase 2Y, not default policies. |
| Stale context | Phase 2Z reloads current Phase 2W, 2X, and 2Y artifacts before runtime. |
| Environment mismatch | CPU-hidden non-XLA pilot cannot support GPU/XLA/default-readiness claims. |
| Artifact mismatch | JSON/Markdown/result/log paths and pilot-nomination contract are predeclared. |

Audit status: `PASSED_FOR_FUTURE_IMPLEMENTATION_REVIEWED_READY`.

Review status: Claude material review was unavailable because the approval
layer blocked external transfer of private repository context.  Fresh local
Codex substitute review returned `VERDICT: AGREE`; this is weaker than full
Claude material review.

Runtime repair status: initial `4096` samples per candidate runtime timed out
with no artifact, so the pilot size was repaired to `1024` per candidate under
the stated timeout-feasibility escape hatch.  This timeout is not proposal
evidence.
