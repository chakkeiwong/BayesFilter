# NeuTra Gap-Closure Plan (2026-08-17)

## Research Question

Why does the known three-component Gaussian-mixture control fail target-specific
HMC tuning at modern rank/folded R-hat while target values, scores, transport
outputs, and short-canary movement remain finite?

## Remaining Gaps

| Gap | Current evidence | Known closure path | Unknown hypothesis to test |
|---|---|---|---|
| Three-mode HMC tuning veto | All `L={3,5,10,15,20,25}` fail R-hat; first transformed coordinate has R-hat `4.77--4.89` | New three-mode-specific transport/tuning scope after diagnosis | Chains remain in different modes; the fixed transport is poorly connected; the verification starts are unbalanced; or the target scale is too difficult for the frozen map |
| Mode coverage | Component-aware starts use four chains for three modes, duplicating one mode | Use known component labels and balanced initialization diagnostics | One duplicated chain or local mode trapping dominates R-hat |
| Frozen transport support | Finite forward/inverse maps are proven only mechanically | Measure pushforward of each exact mixture component and local Jacobian/score behavior | One component maps to a distorted/remote latent region, causing bad mixing |
| Two-mode evidence breadth | One frozen checkpoint and one seed passed | Add independent control seeds and correlated Gaussian control | The pass is seed/checkpoint-specific or marginal diagnostics are underpowered |
| Divergence observability | TFP route reports native divergence unavailable | Treat unavailable as unknown; use finite/status/movement/energy proxy diagnostics only as stated | Hidden integration instability may coexist with finite values |
| Geometry/application coverage | Varying-Hessian, reverse funnel, KSC, German, and final learned model are not all promotion-complete | Run each under a fresh scope-specific plan after controls | Control success may not transfer to difficult targets |

## Evidence Contract

- **Primary question:** identify the mechanism behind the three-mode tuning veto.
- **Baseline:** frozen three-mode checkpoint SHA-256
  `57b21cc99778b0e24e6c5809ebbb6137709edf8177e7faeeac9d259deb2e7b12` and
  the existing fixed-mass tuning grid.
- **Hard vetoes:** nonfinite target/score/transport, hash mismatch, missing
  memory-growth/XLA provenance, invalid mode labels, or failed artifact writes.
- **Promotion veto:** any candidate with modern R-hat above `1.01` remains
  rejected; no threshold relaxation is permitted.
- **Explanatory diagnostics:** acceptance, log-accept proxy, mode counts,
  latent/physical moments, Jacobian values, and runtime.
- **Nonclaims:** no posterior correctness, no HMC validity, no training-method
  ranking, and no SSL-LSTM transfer from this diagnostic.
- **Artifact:** fresh root under
  `docs/plans/artifacts/neutra-gap-closure-2026-08-17/`.

## Hypotheses and Discriminating Tests

### H1: Initialization or mode-coverage artifact

Run the same frozen transport and fixed kernels (`L=20` and `L=25` at the
recorded tuned step sizes) with:

- the existing component-aware four-chain initialization;
- an alternate balanced four-chain assignment;
- a deliberately local same-mode initialization.

Record per-chain nearest-component labels, transitions, mode occupancy, first
coordinate means, R-hat, and finite/status/movement diagnostics. This is a
mechanics diagnostic; none of these starts can establish posterior validity.

### H2: Frozen transport support/geometry failure

For each exact mixture component, draw a small fixed cloud in physical space,
map it through the frozen inverse transport, and record latent means, latent
covariances, forward/inverse reconstruction error, log-determinants, and score
finiteness. Compare component separation in physical and latent coordinates.

If one component has extreme latent displacement, singular local scale, or
nonfinite score, the repair trigger is transport capacity/training, not HMC
tuning.

### H3: Tuning budget or verification-window artifact

Reuse the exact fixed kernel and run a short diagnostic ladder with independent
seeds and increasing verification lengths. Check whether the failing first
coordinate R-hat decreases with longer chains or remains mode-separated. This
cannot override the production R-hat gate; it only distinguishes transient
verification noise from persistent mode separation.

### H4: Target implementation or score mismatch

Run value/score finite-difference and autodiff parity on representative points
near every mode and along interpolation paths between modes. Compare the
transformed score from the adapter with a GradientTape total derivative. Any
mismatch is an implementation veto and blocks transport interpretation.

## Execution Order and Budget

1. Run focused three-mode contract tests and compile checks.
2. Execute H4 value/score parity and H2 support geometry on CPU/XLA or GPU/XLA
   small fixed clouds; cap 20 minutes.
3. Execute H1/H3 fixed-kernel initialization and verification diagnostics on
   GPU 1; cap 45 minutes.
4. Classify the result as implementation, transport/support, initialization,
   tuning-window, or unresolved. Do not launch a new full HMC claim run unless
   a reviewed repair trigger is met.
5. Write a result note and reset memo. A new training or retuning campaign is
   a separate plan after this diagnosis.

## Assumption Audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|
| Fixed `L=20/25` kernels | Existing tuning artifacts | Step sizes may be target-specific and stale | Verify exact step-size/hash provenance | Diagnostic reuse |
| Four chains | Existing serious controller | Duplicate mode assignment can inflate R-hat | Explicit per-chain mode labels | Baseline, not universal |
| Nearest-mean mode label | Known separated analytic target | Boundary points can be ambiguous | Report distance margins and do not use as sole gate | Explanatory label |
| Small physical clouds | Cheap local geometry probe | May miss global transport defects | Pair with cross-mode interpolation and H1/H3 | Diagnostic hypothesis |

## Skeptical Review

- Lowering the R-hat threshold or discarding the failing coordinate would make
  the result answer a different question and is forbidden.
- Acceptance and finite values cannot distinguish mode trapping from correct
  mixing, so mode labels and chain-specific diagnostics are required.
- A local transport Jacobian check cannot certify global support; it is only a
  hypothesis discriminator.
- A successful alternate initialization would identify an initialization
  sensitivity, not prove the learned transport correct.
- If all tests are inconclusive, the correct outcome is an unresolved gap and a
  new capacity/training experiment, not promotion.

Review verdict: fit for bounded execution. The plan separates known repairs
from hypotheses, preserves hard gates, and stops before an unreviewed retraining
or HMC claim run.
