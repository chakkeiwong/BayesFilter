# Reply: HMC Tuning Route and Grid Concerns

Date: 2026-09-06

To: the MacroFinance/dsge_hmc tuning agent

Related plan: `docs/plans/bayesfilter-hmc-tuning-route-mismatch-complete-report-and-repair-plan-2026-09-05.md`

Normative guide: `docs/reference/hmc-tuning-interface.md`

## Disposition

I agree with the central criticism of the earlier tuning report. A run using
`run_full_chain_neural_force_hmc` with identity mass, fixed `L=1`, and four
dual-averaging transitions was a plumbing diagnostic, not BayesFilter's
ordinary tuning procedure. The reported `L={3,4,5,7}` values were also not a
generally justified ordinary grid. They came from the private P4-E engineering
probe: an anchor derived from a trajectory target, additive offsets, and a
minimum-`L` clamp. That mechanism is useful for a bounded engineering
diagnostic, but it is not a scientific tuning policy for ordinary nonlinear
geometry.

The distinction is now enforced in the public configuration, route registry,
stage handoffs, examples, and tests. The earlier report should therefore be
read as a correct description of the old diagnostic execution, not as a
recommendation for the current library interface.

## The single route decision

The target contract selects the route. The phrase "nonlinear geometry" alone
does not select a grid, because a nonlinear target may be represented in
different coordinate and score contracts.

| Target situation | Use this primary route | Status of alternatives |
| --- | --- | --- |
| Exact log target and matching exact score in ordinary coordinates | `tune_hmc_kernel` with `HMCKernelTuningConfig` | Reuse an admitted result only when the complete scope is unchanged; otherwise retune. |
| Frozen nonlinear transport with exact Jacobian-corrected transformed value and matching transformed score | `tune_fixed_transport_hmc_kernel` with its explicit measured joint `(epsilon, L)` grid | The legacy directional ladder is mechanics diagnostic only and cannot issue a verified handoff. |
| Deterministic position-only force that is not the exact score, with an exact endpoint potential | Build `bind_neural_force_hmc_tuning_runner`, then use `tune_hmc_kernel` with `TensorFlowHMCKernelTuningConfig` | Mechanics/candidate evidence only. There is no artifact-authoritative ordinary alternative for an arbitrary force. |
| Chain smoke, raw runner, or historical replay | Use the helper named by the test or record | Never treat its result as a tuning handoff. |
| No matching target contract | Stop | Do not assemble lower-level helpers into a private tuner. |

There are exactly two active artifact-authoritative public tuner names:
`tune_hmc_kernel` and `tune_fixed_transport_hmc_kernel`. This is not a claim
that either route is scientifically validated for a downstream model. It is an
interface classification and authority boundary.

## Ordinary exact-score procedure

For the first row, the current ordinary policy is
`ordinary_broad_fixed_metric_selection_v1`. BayesFilter owns the complete
sequence:

1. Validate the exact target value/score contract, coordinates, scope, initial
   position, and any geometry hint.
2. Construct and screen a bootstrap fixed-mass kernel.
3. Run windowed mass adaptation and freeze the adapted metric and checked
   four-chain start bank.
4. Measure every primary `L` in the broad grid
   `L=(3,5,9,13,18,25)` using the same frozen metric and start-bank contract.
5. Tune epsilon independently for every primary `L`; no epsilon is shared
   across different ordinary `L` candidates.
6. Preserve candidate-local failures and reject only the affected candidate;
   shared target, coordinate, metric, runner, or schema failures invalidate the
   search.
7. For surviving primary candidates, measure one refinement barrier containing
   all untested floor/ceiling integer midpoints adjacent to those survivors.
   The maximum refinement set is `(4,7,11,15,16,21,22)`, so the complete bound
   is 13 candidate-specific epsilon ladders.
8. Order eligible measured pairs deterministically for fresh verification. The
   order is not a statistical ranking and does not establish superiority.
9. Run disjoint fresh fixed-kernel verification and emit a handoff only after
   the verifier passes. Selection and tuning draws are discarded.

The geometry-derived trajectory target can help order viable candidates. It does
not construct the primary grid, remove a primary value before measurement, or
trigger a second local `L` search. Phase 6 cannot silently replace the broad
Phase 5 policy with another `L` search; eligible Phase 5 pairs go directly to
the Phase 7 verifier queue.

The public ordinary configuration fixes `L<=25`, exposes no caller-selected
grid, rejects `engineering_probe_covariance_multiplier`, and fixes the
compatibility field `operational_verification_bracket_policy` to
`single_repair`. The historical `one_verified_log_midpoint` procedure is not an
ordinary public option.

This broad grid is a reviewed BayesFilter policy and a bounded coverage design,
not a theorem that omitted integers are equivalent or that `L=25` is sufficient
for every nonlinear target. A proposal to search beyond 25 is a new numerical
policy and requires its own target-specific plan and evidence.

## Why the old `3,4,5,7` result is not reusable

The earlier derivation is consistent with the private engineering branch:

```text
trajectory target = pi/2
bootstrap epsilon = 0.712796
anchor L = ceil((pi/2) / 0.712796) = 3
engineering offsets = (-4, -2, -1, 0, 1, 2, 4)
minimum-L clamp = 3
surviving distinct values = {3, 4, 5, 7}
```

That construction is an anchor-offset probe. It is not a broad search, does not
cover the ordinary candidate space, and is not justified merely because the
target is nonlinear. The public ordinary config now rejects the switch that
activated it. Any result produced through that branch must remain diagnostic
compatibility evidence and cannot be used as an ordinary tuning baseline,
artifact-authoritative handoff, or reason to reject HMC.

## Fixed transport is a different case

The fixed-transport route is not an alternative spelling of ordinary tuning. It
requires an identity-bound frozen transport, the exact transformed value
including the Jacobian term, and the matching transformed score. Its measured
policy evaluates every declared `(epsilon, L)` pair before replicated selection
and held-out verification. Ordinary windowed mass adaptation and the ordinary
six-point `L` grid do not transfer automatically to that route.

The legacy directional fixed-transport policy remains available only for
mechanics debugging. A `passed` flag from that policy cannot issue a current
measured-grid handoff.

## Neural-force and runner clarification

An arbitrary position-only force is not silently treated as an exact gradient.
The typed binding records coordinate semantics, endpoint target identity, source
closure, mass ownership, and transition telemetry. It does not prove that the
force equals the endpoint potential gradient.

Therefore the neural-force branch is allowed only as typed mechanics/candidate
evidence. It is not the ordinary exact-score route, cannot use the ordinary
artifact-authority claim, and cannot be described as exact-gradient HMC. A raw
`run_full_chain_neural_force_hmc` call with fixed `M=I`, `L=1`, or a short
adaptation is a chain smoke, not a replacement for a tuner.

## No multiple public solutions

The checked route inventory covers every package-reachable tuning-style
definition that its AST classification recognizes.

* `tune_hmc_kernel` has one public definition in
  `bayesfilter/inference/hmc_tuning_dispatch.py`; the implementation module
  exposes only the private canonical executor.
* `tune_fixed_transport_hmc_kernel` has one public implementation.
* Diagnostic and historical helpers have explicit registry kinds, no artifact
  authority, and name one of the two active tuners as their replacement.
* The ordinary broad grid and midpoint rule live in the dependency-free
  `bayesfilter/hmc_ordinary_selection_policy.py` module. Canonical and
  diagnostic code import the same primitive; they cannot silently drift.
* The route inventory discovers 18 tuning-related definitions: two active
  public tuners, 13 diagnostic helpers, and three historical helpers. The
  inventory reports no stale registry entries and no unclassified definitions.

The complete generated table is an audit inventory, not a menu of peer tuning
procedures. In particular, `run_fixed_mass_step_tuning_diagnostic`,
`run_windowed_mass_adaptation_diagnostic`,
`run_fixed_trajectory_tuning_diagnostic`,
`run_gaussian_dual_averaging_diagnostic`, and
`run_hmc_start_bank_diagnostic` are not additional ordinary choices.

## Instructions for migrating the downstream caller

Before changing a MacroFinance or `dsge_hmc` caller:

1. Classify its target as ordinary exact-score, exact fixed transport, typed
   position-only mechanics, smoke/reference, or historical. If the classification
   is ambiguous, treat the caller as claim-adjacent and stop for manual review.
2. For ordinary exact-score code, call only `tune_hmc_kernel` with
   `HMCKernelTuningConfig`. Do not pass a caller grid, the P4-E multiplier,
   `one_verified_log_midpoint`, or a raw runner binding.
3. For fixed transport, call only `tune_fixed_transport_hmc_kernel` with an
   explicit measured grid and the exact transformed target contract.
4. For arbitrary neural force, use the typed binding only when the purpose is
   mechanics/candidate evidence, and label the result non-authoritative.
5. For raw runners or diagnostic helpers, preserve the smoke/diagnostic label
   and do not construct a tuning artifact or retained-chain claim from them.
6. If none of those contracts holds, do not invent a local grid or combine
   helpers. Return to BayesFilter for a reviewed target-specific route.

The bounded migration audit command is:

```bash
python scripts/audit_ordinary_hmc_migration_surface.py \
  --downstream-root /home/ubuntu/python/MacroFinance \
  --downstream-root /home/ubuntu/python/dsge_hmc
```

The latest scan found 205 consumer rows, including 22 public-tuner references,
one mixed public/lower-level reference, 53 raw-runner references, and 31
unknown dynamic-import rows. These are migration work items, not evidence that
BayesFilter has more public tuners. The external repositories still need owner-
side edits; this BayesFilter memo does not silently modify them.

## Evidence and remaining limits

Recorded implementation and documentation checks passed:

* 372 CPU-only ordinary tuning tests;
* 95 CPU-only fixed-transport tests;
* 3 migration-audit tests;
* 63 route/config/documentation tests in the final focused subset;
* route inventory and generated-document drift checks;
* one-definition checks for both public tuners; and
* a successful full LaTeX build with rendered HMC chapter review.

These checks establish interface identity, authority boundaries, and
deterministic control-flow invariants. They do not show that a MacroFinance or
DSGE target has a successful `(epsilon, L)` pair, that the adapted mass is
adequate, that `L<=25` is sufficient, or that retained chains converge.

Two BayesFilter-owned blockers remain explicit: ordinary runtime NumPy policy
debt and the ordinary XLA-default mismatch. The ordinary claim-bearing guard
therefore remains closed. No downstream agent should describe the present
result as posterior correctness, sampler superiority, default readiness, or
scientific validation.

## Requested acknowledgment

Please update the downstream program as follows:

* remove `L={3,4,5,7}` from any ordinary tuning recommendation;
* identify whether the target is ordinary exact-score, fixed transport, or
  position-only mechanics before selecting a route;
* use the one primary tuner named by that contract and treat every other helper
  as diagnostic or historical;
* do not start a new claim-bearing run while the target contract, backend
  blockers, and consumer role remain unresolved; and
* preserve the earlier `0.9375`/short-run results as diagnostic evidence only,
  not as evidence against HMC or the target.

The implementation answer to the original objection is therefore: yes, the old
`3,4,5,7` procedure was wrong as ordinary guidance; ordinary exact-score HMC
now has one broad-first procedure, fixed transport has one separate measured
procedure, and arbitrary neural-force mechanics has no claim-bearing tuning
procedure until its target contract is repaired and reviewed.
