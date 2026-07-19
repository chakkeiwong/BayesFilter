# Focused Native Review: SSL-LSTM A4 HMC Repair-02

Date: 2026-07-14 (Asia/Shanghai)

Review type: `FOCUSED_NATIVE_READ_ONLY_REVIEW`

Status: `AGREE_AFTER_REPAIR`

## Scope

Reviewed the prospective fixed-mass dual-averaging repair:

- `docs/plans/bayesfilter-ssl-lstm-completion-phase-a4-hmc-acquisition-repair-02-plan-2026-07-14.md`;
- `docs/benchmarks/run_ssl_lstm_a4_hmc_repair_02_2026_07_14.py`;
- `tests/test_ssl_lstm_a4_hmc_repair_02.py`;
- the reviewed `HMCTuningPolicy.fixed_mass_dual_averaging` and reusable
  full-chain HMC implementation; and
- all prior A4 HMC receipts used for budget and repair-trigger lineage.

The review checked target/geometry identity, original starts, fixed-mass
adaptation semantics, target acceptance, post-warmup step freezing, state
handoff, acceptance roles, GPU/XLA authority, archive separation, exact receipt
hashes, budget accounting, sequential stops, and nonclaims. Claude was not
used or required.

## Findings And Repairs

| Severity | Finding | Repair | Recheck |
| --- | --- | --- | --- |
| Material scientific description | The first plan draft said trajectory length remained fixed while adapting step size with a fixed leapfrog count. That is false: trajectory length becomes `L * epsilon`. | Corrected the plan and retained receipt to record initial trajectory `1.57` and frozen retained trajectory `4 * frozen_step`. | Plan and runner now describe the actual mechanism. |
| Material statistical gate | Applying the narrow `[0.55,0.85]` target neighborhood to each 64-draw chain would repeat repair-01's brittle per-chain threshold problem. | Made `[0.55,0.85]` the aggregate adaptation target band and retained `[0.20,0.95]` as the broad per-chain safety screen. | Added a focused test where one chain lies outside the aggregate target band but remains safe and aggregate acceptance is near `0.70`; the screen selects. |
| Material handoff lineage | The initial retained handoff checked hashes and shapes but did not require exact equality between the private final state and last screen draw or exact private adaptation config identity. | Added exact state equality, private config, initial-state policy, source, trusted-manifest, ancestry, and step-trace checks. | Focused and repository adaptation suites pass. |
| Moderate test environment | TensorFlow was initially imported before the test set its CPU-hiding variable. | Moved `CUDA_VISIBLE_DEVICES=-1` before TensorFlow import. | CPU-hidden tests report only CPU execution. |

## Verification

| Check | Result |
| --- | --- |
| Repair-02 plus repository fixed-mass tests | `21 passed` in `9.43s`; warnings are TensorFlow/Gast deprecations |
| Real TFP adaptation semantics | Tiny CPU-hidden test confirms fixed-mass dual averaging, scalar post-warmup step trace, and constant frozen step across returned draws |
| Compile | Runner and tests compile under `tfgpu` with GPU hidden |
| Whitespace | `git diff --check` passed |
| Prior GPU budget | `1556.734474526951s` charged; `27243.26552547305s` remains |
| Source hashes | Runner `5f38c2ddadda045cda64a85d85d16e0473563177ff23b8f88e2095e2a26b97c5`; tests `ca4d900f9d68a6e90d2d00e60379d146280b7ce7667681af152c5a5db53cb565`; plan `760dfc3c4c623200cf8b9a11308c4f13398af1ae1f33c75dacee1067b38af667` |
| Original harness | Byte-preserved SHA-256 `89e49435001613300c62e97f3227102919c298127984c9541ef9e2d50921564f` |
| Namespace | Fresh `repair-02` only; no existing artifact collision |

## Interpretation

This is a coherent next discriminating experiment. Repair-01 showed that a
manually halved step moved all chains but over-accepted. Dual averaging targets
`0.70` directly, using the reviewed fixed-mass policy and the same A0 affine
geometry. The adaptation screen is a tuning diagnostic only. It cannot admit
calibration draws, and its samples are not pooled into retained evidence.

The target acceptance is a tuning target, not a claim that realized acceptance
must equal `0.70`. The aggregate `[0.55,0.85]` band is a broad prospective
screen. Per-chain `[0.20,0.95]` catches pathologies without allowing a single
64-draw fluctuation near the target to dominate selection.

If selected, the final screen state and scalar frozen step may seed exactly one
fresh retained `250/250` run. The existing serious retained movement, R-hat,
ESS, MCSE, acceptance, finiteness, and native-divergence rules remain the
sampler-admission authority.

No possible outcome establishes posterior correctness, convergence proof,
sampler superiority, predictive equivalence, NeuTra readiness, model adequacy,
or default readiness.

## Verdict

`VERDICT: AGREE`

The single trusted GPU/XLA adaptation screen may run. Retained acquisition is
eligible only if the adaptation receipt is `SELECTED` and survives exact
private/public replay and hash checks.
