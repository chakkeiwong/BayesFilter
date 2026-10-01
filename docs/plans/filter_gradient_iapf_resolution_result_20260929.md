# Checked iAPF stopping-resolution result

The checked decision owner passes CPU04874 and GPU04875, with161 independent
readback/policy checks04876. On each device it preserves33 resolved decisions,
rejects the three adjacent-threshold cases explicitly and rejects seven invalid
histories. All propagated CV intervals contain the original Python CV, the raw
compiled CV and the independent80-digit Decimal result. The tests cover k1/2/3,
changed operands, shifts, monotonicity, particle growth/capacity and strict
iteration eligibility. Each signature has one trace and enclosing HLO without
host callbacks. This qualifies the checked primitive in the declared bounded
scope; the adaptive runtime and selection validator remain unchanged.

The original GPU mismatch is preserved, not reclassified as equivalence. The
new checked action-2 means the threshold is numerically unresolved. The raw
action/CV, interval endpoints and resolution flag remain available. A future
runtime caller must report a DiagnosticFailure, never choose a fallback action.
No tau, stopping inequality, particle rule or fitted numerical value changed.

The interval calculation is separate rejection diagnostics. It propagates
outward FP64 rounding through shifted exponentials, sequential sums, sample
variance, square root and division. Its explicit exp accuracy assumption is
documented in `filter_gradient_iapf_resolution_20260929.md` and the retrieved
source receipt `filter_gradient_iapf_accuracy_source_20260929.json`. Equal-log
windows retain the exact zero CV. The vendor accuracy table and36 histories
are not proof of a universal compiler/backend error bound; broader runtime
scopes still require applicable qualification.

04871/04872 stop before numerical execution on FP32 inference for Python zero
literals in TF selection/maximum operations. Explicit FP64 constants fix these
tracing defects without changing gates.04873 passes all36 numerical histories;
review then adds the missing wrapper-level seven invalid-input checks before
the final CPU/GPU workers. All attempts remain archived. New-source/harness
lint and whitespace checks pass. No allowlist exception was added; coverage
remains313 sources/1457 exact exceptions.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Nonclaim |
|---|---|---|---|---|---|
| Use checked primitive in the next implementation phase | All declared CPU/GPU containment/decision gates pass | Ambiguous histories explicitly refused | Full nested-controller compilation | Wire offline adaptive recurrence and validator | No adaptive endpoint admission yet |
| Preserve raw incompatibility | Original GPU branch flip remains documented | Unguarded primitive remains ineligible | Other near-boundary histories | Keep resolution diagnostics and refuse failures | No exact raw CPU/GPU equivalence claim |

| Inference status | Finding |
|---|---|
| Hard veto screen |33 resolved,3 unresolved and7 invalid cases handled as declared per device |
| Statistically supported ranking | None |
| Descriptive-only evidence | Bounded interval widths and rounding examples |
| Default readiness | Checked primitive only; adaptive public execution remains open |
| Next evidence | Full recurrence, shape dispatch, fitter/cast vetoes, work accounting and costs |

Six workers consumed31.693767551 CPU and8.982364673 GPU seconds. Remaining global
budget is24.999432 CPU/24.770371 GPU process-hours. The master program
still has broader numerical, cost/capacity, source-applicability and integration
gaps, so main remains unmerged. No training/HMC, scientific/canonical claim or
production performance ranking follows from this mechanics phase.

Post-run review: refusing an ambiguous branch intentionally narrows admissible
inputs and must be visible to callers. It does not excuse a changed resolved
decision or let one device pick a different scientific path. GPU nested-owner
compilation can transform arithmetic differently, so primitive HLO alone cannot
close full-controller compatibility; test the checked decisions again inside
the eventual actual offline owner. Retain the original failure as a regression
case, with explicit refusal as the checked runtime contract.
