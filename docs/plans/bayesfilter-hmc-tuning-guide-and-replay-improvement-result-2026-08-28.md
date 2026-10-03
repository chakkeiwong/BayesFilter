# BayesFilter HMC Tuning Guide And Replay Improvement Result

Date: 2026-08-28

Status: `COMPLETE`

Baseline: `1ef8876666ea05698b3fa4e46a1d6c10a721fad7`

Implementation commit: `0da727c49d898478b3259189b846a029280849d8`

BGS mechanics-handoff correction:
`2315e2b28d5c6b6520df84681b6cb358f9808e36`

Plan:
`docs/plans/bayesfilter-hmc-tuning-guide-and-replay-improvement-plan-2026-08-28.md`

## Outcome

The HMC tuning interface documentation and deterministic replay boundary have
been repaired. An ordinary tuning result initialized from nonidentity
covariance can now issue a durable JSON-ready mechanics payload and reconstruct
the same initial and adapted mass identities without rerunning target
evaluation, tuning, a transition, or HMC.

The agent-facing guide now explains the two active artifact-authority tuners,
all eight non-authoritative public-tuner records, geometry precedence and
identity fallback, a covariance-first handoff, fixed-transport scope, typed
neural-force mechanics, exact endpoint correction, artifact schemas, and the
separate retained-sampling boundary. The LaTeX manual contains the same
substance and executable examples.

## Feedback Disposition

The MacroFinance review is correct on every material point. In particular,
geometry is shared responsibility: the caller supplies a center and optional
negative-Hessian, covariance, or scale hypothesis, while the tuner validates
that hypothesis and owns construction, adaptation, candidate selection, repair,
and fresh verification. Its reported Phase 14 failure remains upstream of the
tuner: the tuner was called zero times, so this BayesFilter repair does not
resolve MacroFinance's incumbent-eligibility mismatch.

The Claude audit is substantially correct, with three corrections incorporated
into the final guidance:

1. The inventory has ten `public_tuner` records, not nine: two active and eight
   diagnostic or historical.
2. For the implemented symmetric, volume-preserving proposal, a poor or
   sign-reversed deterministic position-only field can reduce efficiency but
   does not by itself change the exact endpoint-corrected invariant target. A
   wrong endpoint potential or invalid proposal structure does.
3. A repository binding proves software identity and declared coordinate
   semantics; it cannot prove that a supplied endpoint function equals the
   claimed scientific target. That requires target-specific validation.

## Implemented Changes

- Result-based durable replay now consumes the validated geometry retained by
  `HMCKernelTuningResult`. Explicit geometry arguments remain supported for the
  older serialized-payload reconstruction path.
- The durable mechanics payload records both initial and adapted mass
  signatures. Replay validates geometry, adapter, dimension, target scope,
  execution settings, initial position, and both mass identities.
- The ordinary `1.01` tuning-handoff R-hat threshold has one named implementation
  constant.
- Runner-binding schema v2 records force semantics, endpoint-target identity
  and coordinates, affine Jacobian convention, and a tensor kernel factory.
- The public dispatcher is the sole inventoried ordinary-tuner entry point; the
  old module-level function is an explicit compatibility delegate.
- The TensorFlow route separates `diagnostic_only` from `candidate`. A candidate
  that passes its declared metric, divergence, and four-chain acceptance screen
  may hand the same frozen transition to retained execution. This is mechanics
  authority only: artifacts state `artifact_authority=False`,
  `posterior_admission_authority=False`, and `admission_supported=False`.
- The initial closeout's demand for fresh R-hat and XLA before any retained
  handoff was wrong relative to the BGS plan. R-hat and ESS are retained-run
  explanatory diagnostics; XLA qualification is irrelevant when execution is
  explicitly non-XLA. The one-ULP inclusive-boundary fix and two-ULP rejection
  regression remain.
- The diagnostic prototype no longer supplies numeric defaults for its exposed
  tuning controls or a bare `candidate` evidence role. Its fixed four-chain,
  `float64`, start-bank, metric-window, trajectory-grid, seed-offset, and TFP
  dual-averaging policies are serialized as unqualified diagnostic choices;
  the handoff-capable screen is labeled `candidate`.
- Importable covariance-first and neural-force examples exercise argument and
  binding contracts without launching HMC.
- Generated route tables, the Markdown reference, the LaTeX chapter,
  documentation contract tests, and dated downstream migration guidance have
  been updated together.

## Verification

All framework checks intentionally hid the GPU with
`CUDA_VISIBLE_DEVICES=-1`. No research or claim-bearing HMC run was performed.

| Check | Result |
|---|---|
| Complete scoped HMC interface suite | `242 passed, 6 failed`; all six failures reproduced on untouched baseline |
| Focused dispatcher, contract, geometry, public API, replay suite | `86 passed` |
| Typed neural-force binding subset | `6 passed` |
| Durable replay success and rejection checks after final policy repair | `3 passed` |
| Route inventory | 10 classified, 0 stale, 0 unclassified |
| Generated documentation drift | passed |
| Three non-HMC executable examples | passed |
| Changed-module byte compilation | passed |
| `git diff --check` | passed |
| `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` | passed, 529 pages |
| New chapter layout | zero overfull boxes; rendered pages inspected |

`ruff` was not installed, so no ruff result is claimed. Python compilation and
the focused behavioral tests supplied the available static and executable
checks.

The six baseline failures are:

- `test_operational_checkpoint_binds_complete_v2_lineage`
- `test_operational_outer_loop_accepts_fallback_and_applies_reserved_repair`
- `test_operational_phase5_selection_repairs_through_empirical_midpoint`
- `test_operational_exact_l_candidate_failure_is_budget_exhausted_not_runtime_error`
- `test_outer_loop_out_of_band_historical_acceptance_reenters_full_stages`
- `test_operational_outer_loop_runs_one_verified_midpoint_then_stops`

They fail because current operational Phase 7 requires the explicit P4-E
engineering probe bank or returns the associated hard-veto disposition. The
same six nodes fail on baseline `1ef88766e` with the same messages.

## Decision

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Accept the guide and replay repair | Nonidentity JSON replay preserves both mass signatures without runtime work | No task-specific veto fired | Target equality still requires downstream validation | Commit and migrate downstream callers against the recorded commit | No HMC convergence, posterior, performance, GPU, XLA, or default-readiness claim |
| Permit candidate mechanics handoff | Exact metric, final-divergence, and four-chain screen passes | Diagnostic-only and failed candidates remain closed | Posterior convergence is not established by tuning | Run the same frozen binding in a retained pilot and assess posterior diagnostics separately | No posterior admission, XLA, or scientific claim |
| Leave MacroFinance Phase 14 separate | Failure occurs before `tune_hmc_kernel` | Upstream wrapper veto remains | Correct incumbent-eligibility repair belongs to MacroFinance | Repair and retest in that repository | This work did not produce a MacroFinance kernel |

## Inference Status

| Item | Status |
|---|---|
| Hard veto screen | Candidate mechanics require a valid rank-eligible metric update, zero final-verification divergence, and the declared four-chain acceptance screen |
| Statistically supported ranking | Not applicable; no method comparison was run |
| Descriptive-only differences | None used for promotion |
| Default readiness | Not established |
| Next evidence needed | Target-specific equality checks downstream and retained multi-chain diagnostics; XLA evidence only if XLA is later selected |

## Post-Run Red Team

The strongest alternative explanation for the replay pass is that signatures
could agree while a downstream target callable computes the wrong density. The
test deliberately does not claim otherwise: it proves deterministic mechanics
identity and absence of runtime work, not scientific target correctness. A
target-specific value/score or endpoint-potential validation would overturn any
unsupported downstream equality claim.

The weakest remaining interface area is not the mechanics handoff but the later
posterior interpretation. The tuning screen does not establish convergence,
and a non-XLA run provides no evidence about XLA compatibility or performance.
