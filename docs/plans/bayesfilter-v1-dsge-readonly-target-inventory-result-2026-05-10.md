# Result: BayesFilter v1 DSGE Read-only Target Inventory

## Date

2026-05-10

## Purpose

This note executes the read-only DSGE inventory requested after another agent
claimed that the SGU work was done.  The goal is to decide whether any DSGE
target now justifies BayesFilter structural adapter work, and whether SGU can
be treated as production-ready.

## Repositories Checked

```text
/home/chakwong/BayesFilter   a86a51f
/home/chakwong/python        4129f05
```

The DSGE repository was inspected and tested read-only.  No DSGE source files
were edited in this BayesFilter lane.

## Verification Command

Deliberate CPU-only command:

```bash
cd /home/chakwong/python
DSGE_FORCE_CPU=1 CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_sgu_marginal_utility_timing.py \
  tests/contracts/test_sgu_causal_control_anchor_gate.py \
  tests/contracts/test_sgu_current_control_derivation_gate.py \
  tests/contracts/test_structural_dsge_partition.py::test_rotemberg_exposes_mixed_metadata_and_completion \
  tests/contracts/test_structural_dsge_partition.py::test_ez_exposes_all_stochastic_metadata_after_timing_audit \
  tests/contracts/test_structural_dsge_partition.py::test_ez_metadata_records_stability_policy_without_bk_claim \
  tests/contracts/test_dsge_strong_structural_residual_gates.py::test_rotemberg_second_order_dy_completion_closes_identity_residual \
  tests/contracts/test_dsge_strong_structural_residual_gates.py::test_rotemberg_second_order_dy_completion_fails_closed_when_singular \
  tests/contracts/test_dsge_strong_structural_residual_gates.py::test_sgu_second_order_filter_target_is_blocked_by_foc_residual_order \
  tests/contracts/test_dsge_strong_structural_residual_gates.py::test_sgu_state_identity_gate_does_not_make_quadratic_policy_better_than_linear \
  tests/contracts/test_dsge_strong_structural_residual_gates.py::test_sgu_quadratic_gate_failure_is_volatility_correction_driven \
  tests/contracts/test_dsge_strong_structural_residual_gates.py::test_sgu_joint_state_control_projection_is_new_target_not_gate_b \
  tests/contracts/test_dsge_strong_structural_residual_gates.py::test_ez_timing_audit_exposes_all_stochastic_metadata
```

Observed:

```text
16 passed, 3 warnings in 14.08s
```

Warnings:

- TensorFlow Probability `distutils` deprecation warnings;
- a pytest cache write warning because `/home/chakwong/python/.pytest_cache`
  is read-only in this sandbox.

These warnings do not change the target-status decision.

## SGU Claim Audit

The other agent's SGU claim is valid only in the narrow timing-contract sense.
The DSGE result note says the neural SGU residual helpers now use current
`mu_t`, not `mu_{t+1}`, in the normalized Euler and capital FOC residuals.

Allowed label:

```text
sgu_marginal_utility_timing_contract_passed
```

Still preserved blockers:

```text
blocked_sgu_causal_anchor_projection_nonlocal
blocked_sgu_causal_projection_residual_not_closed
blocked_nonlinear_equilibrium_manifold_residual
blocked_sgu_second_order_filter_foc_residual_order
blocked_sgu_quadratic_policy_not_better_than_linear_residual
blocked_sgu_projection_is_new_target_not_gate_b
```

No promotion is justified to:

```text
sgu_causal_filtering_target_passed
sgu_second_order_perturbation_filter_target_passed
```

Inventory classification:

```text
blocked_sgu_causal_locality
diagnostic_only
```

BayesFilter consequence:

- do not build BayesFilter SGU production filtering adapters;
- do not open derivative, JIT/static-shape, HMC, posterior, or GPU claims for
  SGU;
- treat SGU as a DSGE-owned diagnostic target until a causal local filtering
  target is derived and tested.

## Rotemberg NK Inventory

Evidence reviewed:

```text
docs/plans/rotemberg-second-order-dy-completion-derivation-2026-05-06.md
docs/plans/rotemberg-second-order-adapter-diagnostics-decision-2026-05-07.md
tests/contracts/test_structural_dsge_partition.py
tests/contracts/test_dsge_strong_structural_residual_gates.py
```

The Rotemberg second-order `dy` completion is model-owned and tested in the
DSGE repository:

- raw pruned `dy` remains a blocker diagnostic;
- the model-owned completion closes the `dy` identity;
- singular completion fails closed;
- the metadata exposes a mixed stochastic/deterministic structural partition.

Inventory classification:

```text
needs_test_only_bridge
ready_external_compatibility_fixture
```

BayesFilter consequence:

- Rotemberg is the best non-SGU candidate for a future optional live DSGE
  compatibility fixture;
- no generic BayesFilter production patch is justified now, because the
  economics and completion law are DSGE-owned;
- a later bridge should pass data through BayesFilter structural protocols
  without importing Rotemberg economics into BayesFilter production modules.

## EZ Inventory

Evidence reviewed:

```text
docs/plans/ez-state-timing-and-bayesfilter-metadata-note-2026-05-07.md
docs/plans/ez-analytical-solver-stability-policy-2026-05-07.md
tests/contracts/test_structural_dsge_partition.py
tests/contracts/test_dsge_strong_structural_residual_gates.py
```

EZ now has a narrow metadata/stability result:

- state names are `("a", "v")`;
- both state rows have innovation support;
- the BayesFilter metadata regime is all-stochastic;
- the default analytical transition has spectral radius below one.

The analytical EZ solver still does not compute a full QZ/BK determinacy
certificate:

```text
bk_ok = None
log_det_margin = None
```

Allowed label:

```text
ez_all_stochastic_metadata_passed
```

Blocked labels:

```text
ez_bk_passed
ez_qz_determinacy_passed
ez_hmc_ready
```

Inventory classification:

```text
needs_test_only_bridge
diagnostic_only
```

BayesFilter consequence:

- EZ may support a future all-stochastic metadata compatibility fixture;
- EZ should not be used for determinacy, HMC, posterior, or production
  readiness claims until the DSGE project derives and tests the stacked
  determinacy system.

## Decision

No DSGE target justifies new BayesFilter structural adapter implementation in
this pass.

The correct resumed BayesFilter work is:

1. keep SGU blocked as a production filtering target;
2. treat Rotemberg and EZ as optional live external compatibility candidates;
3. build BayesFilter-local v1 API, fixture, benchmark, GPU, and HMC gates;
4. defer client switch-over until v1 integration criteria pass.

## Next Hypotheses

H1. Rotemberg can become a useful optional live external fixture if a
test-only bridge validates BayesFilter structural metadata, deterministic
completion residuals, and value likelihood without importing DSGE economics
into BayesFilter production code.

H2. EZ can become a useful all-stochastic metadata fixture after a test-only
bridge records that determinacy remains uncertified; the bridge must fail any
attempt to label EZ as BK/HMC-ready.

H3. SGU cannot move beyond diagnostic status until DSGE derives either a
coefficient-level local perturbation target or a separate global/neural policy
target that satisfies causal locality and residual-order gates.

H4. BayesFilter v1 integration should remain blocked until API import tests,
local compatibility fixtures, CPU benchmarks, optional live external checks,
and rollback criteria are all present.
