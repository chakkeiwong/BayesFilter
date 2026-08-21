# Mild-smooth weighted NeuTra result (2026-08-13)

Plan: `docs/plans/bayesfilter-weighted-forward-kl-mild-smooth-proposal-plan-2026-08-13.md`

## Decision

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Frozen mild-smooth weighted candidate remains viable | Canonical sequential fixed-length HMC passed: retained maximum R-hat `1.00396 <= 1.01`, minimum bulk/tail ESS `1845.6 / 1082.7 >= 400`, and all retained numerical diagnostics were finite | No declared hard veto; native divergence is not exposed by this TFP kernel and is therefore not claimed to be zero | One frozen transport seed, one initial-state bank, identity z-mass, and no independent normalized posterior authority | Continue the reviewed regression to German credit; retain this arm for a later matched reverse-KL comparison rather than ranking it now | No posterior correctness, objective ranking, default promotion, or general NeuTra claim |

## Evidence contract executed

- Target: source-bound `nk_like_mild_smooth` with frozen affine lift and the
  local constants SHA-256 `0aa2eb0850b68d3afe5d67eea89ac7b066eab76c0436e015643ca0a7980894b5`.
- Transport: target-weighted forward-KL dense IAF, six stages and `(128,128)`
  hidden layers, float64 GPU/XLA training; update 9000 was frozen from the
  target-specific 10,000-update serious rung.
- Replay: 1,048,576 CPU-generated training rows, 65,536 disjoint selection
  rows, and 65,536 disjoint audit rows. The upstream training artifact bound
  its selected checkpoint and target constants before HMC.
- HMC: fixed-length TensorFlow Probability HMC only, target-specific retuning
  over `L=(3,5,10,15,20,25)`, identity mass in transformed coordinates, XLA,
  float64, TF32 disabled, four chains, and verified GPU-0 memory growth. `L=1`
  and NUTS were not used.
- Sequential policy: `bayesfilter_neutra_sequential_hmc_v1`, with 2,000
  archived warm-up transitions and 1,000 retained transitions per chain.
  Warm-up draws were excluded from the retained diagnostics.

## Results

| Quantity | Result |
|---|---:|
| Selected HMC `L` / epsilon | `3 / 0.5610023` |
| Tuning verification max rank-normalized/folded R-hat | `1.00245` |
| Tuning verification mean acceptance probability | `0.68925` |
| Warm-up / retained draws per chain | `2000 / 1000` |
| Retained maximum R-hat over z and model coordinates | `1.00396` |
| Retained minimum bulk / tail ESS | `1845.6 / 1082.7` |
| Warm-up maximum R-hat | `1.00209` |
| Warm-up minimum bulk / tail ESS | `1599.4 / 1291.2` |
| Sequential wall time | `24.9 s` |
| Total HMC runner wall time | `325.7 s` |
| Archive receipts | four warm-up and two retained chunks |

The tuner selected the finite `L=3` arm. `L=5` and `L=10` produced nonfinite
proposal/acceptance telemetry during fresh verification, `L=15`, `L=20`, and
`L=25` had no viable fixed step under the declared ladder. Those rejected grid
arms explain the local integration geometry; they do not invalidate the
selected `L=3` kernel or turn the selected candidate into a comparison winner.

The tuning record has an extreme but finite log-acceptance energy proxy maximum
(`1.77e62`). It is explicitly an explanatory alert in the current tuning
contract, not a convergence or acceptance veto. The selected tuning
verification and all sequential retained state/value/score diagnostics were
finite. Native divergence telemetry is unavailable, so the result does not
claim zero divergences.

The serious training run clipped gradients on `9347 / 10000` updates. This
weakens any interpretation of training NLL as an optimizer-convergence proof;
the downstream HMC gate is the basis for the limited viability decision.

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | Passed for the selected kernel: finite state, target, score, movement, archive, and checkpoint/target binding; divergence unavailable rather than zero |
| Viable candidate | Frozen weighted forward-KL mild-smooth transport with fixed `L=3`, epsilon `0.5610023` |
| Statistically supported ranking | None. No matched reverse-KL HMC comparison or uncertainty analysis was run |
| Descriptive-only diagnostics | Training NLL, clipping, acceptance, log-acceptance energy tail, tuning-arm rejection pattern, and runtime |
| Default readiness | Not assessed and not promoted |
| Next evidence needed | The planned German-credit target-specific replay/proposal preflight, weighted transport, and corrected-HMC/reference screens |

## Provenance

- Training root:
  `docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/varying-hessian/mild-smooth-serious-r1/`.
- HMC root:
  `docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/varying-hessian/mild-smooth-serious-hmc-r1/`.
- The HMC manifest verifies the training-state SHA-256, the training-manifest
  SHA-256, semantic state hash, target name, and target constants SHA-256
  before tracing. GPU memory growth was configured before logical-device
  initialization; XLA was confirmed at runtime.
- Top-level `result.json`, `run_manifest.json`, and `sequential_result.json`
  SHA-256 values are recorded in the HMC root's `artifact_hashes.json`.

## Post-run red team

The strongest alternative explanation is that this fixed initial bank and
identity z-mass are especially favorable, rather than that the transport has
globally whitened the target. A fresh transport seed or a disjoint
initialization bank failing the same sequential screens would weaken this
target-specific result. The most important missing authority is a normalized
posterior/reference distribution: the source-bound target supplies exact local
values and scores, but not an independent posterior correctness comparator.
