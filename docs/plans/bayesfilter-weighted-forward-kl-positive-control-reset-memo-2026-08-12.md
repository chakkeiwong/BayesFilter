# Weighted forward-KL positive-control reset memo (2026-08-12)

Status: `PAPER_D100_COMPLETE_FUNNEL_FORWARD_VIABLE`

## Completed

- Plan: `docs/plans/bayesfilter-weighted-forward-kl-positive-control-regression-plan-2026-08-12.md`.
- Result: `docs/plans/bayesfilter-weighted-forward-kl-three-mode-result-2026-08-12.md`.
- Generic TensorFlow Gaussian-mixture diagnostics:
  `bayesfilter/testing/gaussian_mixture_diagnostics_tf.py`.
- Generic frozen weighted-mixture HMC authority:
  `bayesfilter/testing/weighted_neutra_gaussian_mixture_hmc_tf.py`.
- Three-mode corrected-HMC runner:
  `docs/benchmarks/run_weighted_neutra_three_mode_hmc_2026_08_12.py`.

The final three-mode weighted candidate passed. The small `(64,64)`,
three-stage candidate failed all fixed-kernel R-hat verification arms and is
preserved as candidate-level negative evidence. The predeclared `(128,128)`,
six-stage, 10,000-update capacity repair passed, then canonical sequential HMC
passed at `L=5`, epsilon `0.3433257029`.

## Exact artifact roots

- Small candidate HMC rejection:
  `docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/three-mode/hmc-weighted-budget1000-r1/`.
- Serious transport:
  `docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/three-mode/component-aware-width128-depth6-updates10000-r1/`.
- Serious HMC pass:
  `docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/three-mode/hmc-width128-depth6-r1/`.

Do not overwrite any of these roots. Checkpoint selection was update 8,750,
and the serious HMC runner verifies its checkpoint/state/tensor hashes before
use.

## Next lane

Port the source-anchored smooth varying-Hessian target from
`/home/ubuntu/python/dsge_hmc/src/dsge_hmc/benchmarks/nk_like_mild.py` into a
BayesFilter TensorFlow-only candidate path.

- `nk_like_mild_smooth`: `rot_alpha=0.35`, `weak_collapse=0.6`,
  `stiff_growth=0.25`.
- `nk_like_strong_smooth`: `rot_alpha=0.70`, `weak_collapse=0.9`,
  `stiff_growth=0.45`.
- Preserve the source's batched `log_prob_batch_tf` formula. Do not call the
  source NumPy path from a BayesFilter candidate runtime.
- Load source frozen affine-lift constants as read-only provenance and bind the
  source file hash plus constants file hash in the manifest. Do not modify
  `/home/ubuntu/python/dsge_hmc`.
- First run TensorFlow scalar/batched and source-probe parity. Then build a
  full-support replay proposal in the affine-lift coordinate; require importance
  ESS and shell diagnostics before any training/HMC claim.
- Train a matched reverse-KL comparator plus weighted candidate under a
  target-specific capacity/optimizer canary. Do not transfer three-mode
  capacity as a promoted default.
- Exact normalized posterior moments are unavailable for the varying-Hessian
  targets. Their initial downstream criterion is current corrected-HMC validity
  and match to an independently run, longer reference-chain diagnostic, not an
  analytic moment claim.

Strong-smooth continuation completed on 2026-08-13: reflected proposal r7 passed heldout ESS (`0.4039`) and max weight (`0.000902`); the disjoint replay and serious `(128,128)`, six-stage, 10,000-update arm selected update 8000; corrected fixed-length HMC selected `L=3`, epsilon `0.4158491`, and passed sequential R-hat/ESS screens (`max R-hat 1.00625`, minimum bulk/tail ESS `776.7/697.0`). This is sampler evidence only because no normalized posterior authority exists. The small `(64,64)` 200-update arm remains preserved as a valid HMC rejection; the serious arm had `7737/10000` clipped updates, which remains an optimization risk.

Mild-smooth continuation completed on 2026-08-13. Its reflected proposal r2
passed heldout ESS (`0.20337`) and max-weight (`0.006186`) screens. The
disjoint serious `(128,128)`, six-stage, 10,000-update arm selected update
9000. Fresh target-specific fixed-length tuning selected `L=3`, epsilon
`0.5610023`; canonical sequential HMC passed after 2,000 warm-up and 1,000
retained transitions per chain (`max R-hat 1.00396`, minimum bulk/tail ESS
`1845.6/1082.7`, no declared hard vetoes). The HMC loader now checks its
checkpoint hash and semantic state plus the training-manifest target name and
constants hash before tracing, preventing a valid checkpoint from being paired
with the wrong smooth target. This is sampler-route viability evidence only:
the source-bound target has no independent normalized posterior authority.
The serious arm clipped `9347/10000` updates, and the tuning record has a huge
but finite energy-proxy alert; both remain explanatory rather than promotion
metrics.

## Next lane

Proceed to the German-credit `gamma_scales2` preflight in the master plan. Do
not infer an objective ranking from the three-mode, strong-smooth, or
mild-smooth weighted passes. The German lane must first confirm the committed
data/reference loader, target-specific batched TensorFlow value/score route,
and a full-support replay proposal before any GPU training. Its Stan/PyStan
reference moments may support a posterior agreement screen that the
source-bound smooth targets cannot provide.

## German-credit terminal result

German target implementation and GPU/XLA batch-native preflight passed. The
1,000-update matched reverse transport selected its terminal checkpoint with
heldout/audit reverse-KL `553.10/553.56`, but `898/1000` updates clipped. The
initial pushed proposal had ESS `1.17/65536`; the one allowed
reference-marginal repair improved ESS to `7.10/65536`, but both failed the
global and median-batch ESS floor `0.0625`. Weighted replay/training was
correctly not launched.

The first reverse HMC artifact was launch-invalid because the shared weighted
IAF ELU pullback mixed float32 literals with float64 state. The dtype repair is
covered by focused tests and a finite one-arm XLA check. The valid `r2` retry
then rejected all `L=(3,5,10,15,20,25,32)` arms on current modern R-hat; the
best arm had max rank/folded R-hat `1.043/1.031` at `L=25`, while the historical
`L=32` arm had `1.125/1.114`. This validates the target-specific negative
candidate evidence while preserving the explicit nonclaim that no weighted
German candidate reached HMC.

Result note:
`docs/plans/bayesfilter-weighted-forward-kl-german-credit-result-2026-08-13.md`.

## Next lane

Do not add unplanned German optimizer/proposal repairs under this campaign.
Continue with the next reviewed positive-control or fresh-baseline target in
`bayesfilter-weighted-forward-kl-positive-control-regression-plan-2026-08-12.md`.

## Current worktree

The worktree remains dirty with unrelated concurrent changes, notably
`bayesfilter/inference/neutra_artifacts.py`, `bayesfilter/inference/neutra_training.py`,
`tests/test_neutra_reverse_kl_training.py`, chapter text, and prior terminal
campaign artifacts. Preserve them. The files in this memo are this lane's new
or modified files. Focused tests after the generic HMC additions passed.

## Paper d100 terminal result (2026-08-13)

Result note:
`docs/plans/bayesfilter-weighted-forward-kl-paper-d100-result-2026-08-13.md`.

- Exact paper Gaussian and funnel targets, CPU replay, GPU/XLA training, and
  canonical fixed-length HMC routes are implemented and covered by focused
  tests.
- Gaussian forward passed sampler gates at `L=25`; the hash-bound corrected
  analytic adjudication rejected projection-2 mean. The reverse smaller-step
  repair passed sampler gates at `L=32` but failed the same separate interval.
  Treat these as candidate analytic vetoes with single-run multiplicity
  uncertainty, not method or target invalidity.
- Funnel reverse passed sampler gates at `L=3` but compressed both `y` tails:
  `E[y^2]=0.83925`, tail probabilities `0.0105/0.0135`, and both extreme
  quantile-law intervals failed.
- Funnel weighted forward-KL passed at `L=10`, epsilon `0.4342689`: maximum
  retained R-hat `1.00497`, minimum bulk/tail ESS `1562.9/863.6`, all nine
  structural 99% intervals, and all five chain-aware quantile-law intervals.
  This is a viable target-specific positive control, not an objective ranking
  or default promotion.
- Recorded d100 execution was `11,931.964 s = 3.314435 h`, within the four-hour cap. No
  10,000-update repair was launched.

The next scientifically justified action, if continuing this exact lane, is
fresh-seed replication of the funnel forward pass and a matched replicated
reverse comparator. Do not infer superiority from the current one-seed
forward pass or spend more budget on unplanned Gaussian interval retries.
