# Focused Native Review: SSL-LSTM A4 HMC Repair

Date: 2026-07-14 (Asia/Shanghai)

Review type: `FOCUSED_NATIVE_READ_ONLY_REVIEW`

Status: `AGREE_AFTER_REPAIR`

## Scope

Reviewed the separately authorized smaller-step repair contract and bounded
implementation:

- `docs/plans/bayesfilter-ssl-lstm-completion-phase-a4-hmc-acquisition-repair-plan-2026-07-14.md`;
- `docs/benchmarks/run_ssl_lstm_a4_hmc_repair_2026_07_14.py`;
- `tests/test_ssl_lstm_a4_hmc_repair.py`;
- the byte-preserved original acquisition harness and tests; and
- the four prior trusted-GPU receipts that establish the shared budget and
  unmoved-chain repair trigger.

The review checked target and geometry preservation, original-start semantics,
constant trajectory length, fresh labels and seeds, no-overwrite behavior,
prior receipt hashes, source binding, budget accounting, retained-extension
eligibility, hard-veto stopping, and claim boundaries. Claude was not used or
required.

## Findings And Repairs

| Severity | Finding | Repair | Recheck |
| --- | --- | --- | --- |
| Material lineage | The first repair draft checked the tuning candidate and source hashes before retained acquisition, but did not independently recheck the tuning receipt's trusted-GPU manifest, exact prior ancestry, initial-state policy, or private-manifest hash. | Added fail-closed validation for trusted GPU/XLA status, target/plan/result identity, finite wall time, exact receipt ancestry, original-start policy, and tuning private-manifest hash. | Focused repair suite passes `7/7`; compilation and whitespace checks pass. |
| Material extension safety | Prior repair segments were required to have schema/status/index and current source hashes, but their exact labels, seeds, ancestry, private-manifest hashes, and all-chain movement were not independently checked before extension. | Added exact checks for those fields and requires `PROMOTION_VETO_EXTEND_IF_BUDGET_ALLOWS`, no hard veto, and movement `[true,true,true,true]`. | Focused repair suite passes and the first rung remains independent of every failed original archive state. |

## Verification

| Check | Result |
| --- | --- |
| Repair-only focused tests | `7 passed` in `0.07s` after review repair |
| Combined original plus repair tests | `17 passed` in `196.34s` before the focused review repair; the repair-only suite was rerun after the repair |
| Compile | Repair harness and tests compile under the `tfgpu` interpreter with GPU hidden |
| Diff whitespace | Passed |
| Original source identity | Harness `89e49435001613300c62e97f3227102919c298127984c9541ef9e2d50921564f`; tests `c6ade5def9f7583ac28535120cb5e1bedca10af3439840ee289938ea382d9033`; original plan `bff39626439f4a43c5973910f53fa3bac1c3494a70404d0795a455a1fa25180d` |
| Prior receipt identity | All four current SHA-256 values match the blocker result and repair runner constants |
| Prior GPU budget | `1333.7487312000012s` consumed; `27466.2512688s` remains under the shared `28800s` cap |
| Kernel | `step_size=0.19625`; `num_leapfrog_steps=8`; trajectory length `1.57` |
| Starts | Tuning and retained segment 0 both construct the original four dispersed states directly; the failed `segment_0` state is never read |
| Namespace | Fresh `repair-01` public/private paths with `overwrite=False` and a pre-run collision check |
| Sequential stop | Non-selected tuning stops; retained hard veto stops; extension is allowed only for R-hat/ESS/MCSE/acceptance promotion vetoes after all chains move |

## Interpretation Review

The repair directly tests the predeclared mechanism: reducing integrator step
size while preserving nominal trajectory length. It does not change the A1
posterior, affine chart, starts, chain count, or admission thresholds. The
rejected balanced archive is comparator context and budget evidence only; no
sample or final state from it is eligible for repair tuning, retained sampling,
or calibration.

A passing tuning screen is not sampler admission. A retained archive is
admissible only if all four chains move, samples and telemetry are finite,
available native divergence is zero, acceptance is within the prospective
bounds, rank-normalized split R-hat is at most `1.05`, bulk/tail ESS are at
least `100`, and mean MCSE/SD is at most `0.10` in both coordinate systems.

Even admission would establish only fitness as an A4 calibration input. It
would not establish posterior correctness, convergence proof, sampler
superiority, predictive equivalence, NeuTra readiness, model adequacy, or
default readiness.

## Verdict

`VERDICT: AGREE`

The one authorized fresh tuning screen may run. If selected, exactly the next
eligible retained rung may run. Execution must stop on a hard veto or budget
failure and must stop HMC immediately on admission.
