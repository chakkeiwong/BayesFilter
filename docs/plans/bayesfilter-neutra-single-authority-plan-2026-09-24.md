# NeuTra implementation consolidation and source-faithfulness

Status: reviewed for implementation, 2026-09-24. Owner request: implement the
literature mechanisms and make every active NeuTra transport a configuration or
compatibility facade of one numerical authority. This is engineering work;
the previous q20 fit canary is complete and its unused allowance is retired.

Completion: phases 1--5 executed. Results, terminal review and the unavailable
historical external-reference checks are recorded in
`bayesfilter-neutra-single-authority-results-2026-09-24.md`.

## Question and evidence contract

Can trainable, frozen, weighted, ordinary, legacy, and mechanism NeuTra callers
evaluate the same configured map, inverse, Jacobian, and gradients through one
implementation, including a complete conditional NAF and the applicable Vaitl
path gradient? Baselines are the existing serialized IAF conventions and the
inspected source equations, not the failed q20 map's validation loss.

Pass requires executable caller-to-core parity, preserved legacy forward maps
and checkpoint ordering, correct derivatives against independent derivatives,
conditional NAF invertibility/log determinant, finite compiled batched updates,
checkpoint/resume and frozen round trips, and a route guard against numerical
forks. Invalid inverses, missing gradients, disconnected callers, checkpoint
reinterpretation, pfor/scalar training, and nonfinite updates veto completion.
Runtime and tiny loss changes are explanatory only. No fit-quality, variance
superiority, posterior convergence, q20 readiness, or default hyperparameter
calibration follows from these checks. Failed engineering checks trigger repair.

## Sources and mathematical boundary

Local sources are in `.localresources/q20-flow-training-literature-20260923/`.

* Hoffman et al. (2019), section 4.1.1: three IAF layers, two hidden layers,
  dimension-wide ELU networks, reversal between layers. Author
  `code/neutra-utils.py:718-770` supplies the optional conditional clamp with
  free bias: `b + c*tanh(h/c)`, no final clamp and no sigmoid scale. This is an
  author-code option, not a claim that every published experiment used it.
* Huang et al. (2018), section 3.1 equation (8), positivity/monotonicity argument
  and appendix: conditional positive slopes, shifts, normalized mixture weights
  from a MADE conditioner; logit of a sigmoid mixture. Author
  `code/naf-flows.py:221-317` supplies the conditioner/pseudo-parameter layout,
  softplus slopes and small nonzero final weights. A full conditional DSF is a
  NAF; the earlier unconditional single-coordinate canary is not.
* Author sigmoid shrinkage changes the range. For unconstrained NeuTra use the
  paper's full-range map with algebraically equivalent log-domain evaluation:
  `logS=logsumexp(logw+logsigmoid(u))`,
  `logC=logsumexp(logw+logsigmoid(-u))`, `y=logS-logC`,
  `log y'=logsumexp(logw+loga+logsigmoid(u)+logsigmoid(-u))-logS-logC`.
  This explicit numerical adaptation preserves equation (8); do not claim
  bitwise reproduction of the author's shrinkage implementation.
* Vaitl et al. (2024), section 3.1 proposition 3.2 equation (16), appendix B.1:
  propagate `s_next=J^{-T}(s-grad(logdet))`, starting at `-z`, then parameter
  VJP with stopped `(s_q-s_target)`. Use each layer's forward pass once.
  Triangular autoregressive solves apply; the coupling-specific O(d) algorithm
  does not apply verbatim. No target Hessian is needed. Report actual RKL
  separately from the detached gradient carrier. Roeder (2017)'s dropped score
  term has zero expectation under its assumptions; no universal variance claim.

These papers define distinct mechanisms. Their configurable combination is a
local composition, not reproduction of one published end-to-end experiment.

## Implementation phases

1. Centralize masked networks, masks, affine IAF parameters, inverse, pullback,
   logdet score, scalar sigmoid-mixture math, and composition primitives in
   `bayesfilter/inference/neutra_transport_core.py`. Convert ordinary training,
   weighted training, legacy training and frozen components to facades. Retain
   old layouts/seeds/initialization and named scale conventions explicitly.
2. Add a public configurable transport using the same core: author-code free
   bias IAF and full conditional DSF/NAF with positive softplus slopes, nonzero
   near-identity initialization, permutations, checked inverse, and full-support
   log-domain evaluation. Provide explicit source profiles and configuration
   serialization. Inverses use bounded TensorFlow loops, not sample loops.
3. Implement layerwise forward proposal scores and wire the estimator into
   batched Adam training with stable signatures, finite/invalid-target rejection,
   optimizer checkpoint/resume, and actual loss telemetry. Move canary numerical
   mechanisms to core-backed facades; keep their historical test meaning.
4. Wire versioned frozen artifacts into the existing loader and fixed-transport
   interface, without changing old artifact hashes or interpreting NAF as IAF.
   Preserve latent identity mass. Generalize post-training 1,000-point probes
   to conditional NAF and distinguish IAF saturation from DSF derivatives.
5. Add source/analytic, round-trip, checkpoint, training/frozen, caller-wiring
   and drift tests; run affected existing regressions and a bounded trusted
   GPU/XLA engineering smoke. Publish the API/source mapping and result/reset
   note with exact remaining limitations.

## Defaults and numerical provenance

| Choice | Provenance/status | Risk and smallest check |
|---|---|---|
| Float64 | Existing q20 scientific configuration; retained baseline | CPU/GPU parity and nonfinite checks |
| Legacy masks, weights, caps, initialization | Saved-checkpoint compatibility, not endorsed training defaults | Nonidentity legacy fixtures and parameter ordering |
| 3 IAF layers, 2 dimension-wide ELU layers, reversal | NeuTra section 4.1.1 architectural profile | Mask/Jacobian checks; quality still requires target calibration |
| Conditional cap c | Explicit caller hypothesis; no universal recommended value | Free-bias gradient and saturation diagnostics |
| NAF mixture count/width/depth | Explicit caller hypotheses | Conditional-dependence, capacity and downstream validation later |
| NAF small final weights | Author source uniform ±0.001 near identity | Distinct component derivatives; not exact identity symmetry |
| NAF softplus positive slopes | Paper constraint/author parameterization | Strict positive finite slopes and tail tests |
| No output shrinkage | Derived from full-support target requirement | Extreme-input finite logdet/inverse checks |
| Inverse tolerances and iteration limit | Explicit numerical controls, tested against residual and bracket width | Unconverged solve produces invalid output, never silent acceptance |
| Adam LR, moments, epsilon, clipping | Explicit training configuration; no new q20 calibrated default | Resume parity, finite rejection, gradient/update telemetry |
| Test seeds/fixtures | Deterministic engineering examples, no statistical ranking | Multiple nontrivial maps, tails and nonzero derivatives |

Engineering checks may use small CPU-only analytic targets, explicitly hiding
GPUs before import. Serious learned quality is not evaluated on CPU. Use the
existing `tfgpu` environment. Start with focused tests, then affected regressions;
bound each test invocation to 10 minutes (operational cap, not a scientific
threshold). GPU smoke: at most 5 minutes, one device, memory growth, small
batched analytic target, XLA, no q20 campaign. Record actual commands/time and
source changes in `docs/plans/artifacts/neutra-single-authority-2026-09-24/`.
No new long research run or use of retired diagnostic allocation is authorized
by this plan. Implementation tests are within the requested repair work.

## Skeptical review before execution

Reviewed locally against source and consumer code; no independent agent used.
Material findings and revisions:

1. A new standalone class would not close drift. Require changes to all four
   existing IAF numerical implementations plus canary primitives and executable
   route coverage. Keep orchestration facades, remove duplicate equations.
2. Historical frozen `dsge_bounded_tanh` uses `c*tanh(h)` but its pullback omits
   the factor c. Correct the derivative in the common core; preserve its forward
   map. This is a correctness repair, not strict parity with the old wrong score.
3. Blindly porting author DSF shrinkage would destroy full real support. Preserve
   the paper equation using log-domain identities and disclose the difference.
4. A whole-map dense Jacobian labeled Vaitl would overstate implementation.
   Require layerwise score recursion and triangular solves for IAF/NAF, plus
   exact-transport zero-path-gradient tests. No coupling-complexity claim.
5. Bisection autodiff is not the inverse derivative. Implement implicit inverse
   differentiation for inverse-density consumers, or reject unsupported use;
   test both input and parameter derivatives before admitting weighted training.
6. An old artifact must not acquire new free-bias semantics. Use explicit new
   configuration/schema and preserve legacy decode; cover parameter hashes.
7. Smoke loss is not readiness. No target retuning, mass adaptation, quality
   ranking or campaign relaunch is included. HMC capability registry and the
   public tuning interface remain the consumer authority.

Wrong baselines, proxy promotion, stale checkpoint context, hidden numerical
defaults, environment mismatch, unsupported rankings and missing stop conditions
were examined. With these revisions the plan answers the engineering question;
execution may proceed. Any discovered additional caller is added to the route
inventory and migrated before declaring consolidation complete.

## Implementation review additions

The second source pass inspected the author's `naf-iaf_modules.py` (retrieved
from `https://raw.githubusercontent.com/CW-Huang/torchkit/master/torchkit/iaf_modules.py`)
and `naf-nn.py`, including cMADE/CWNlinear. The source profile retains the
constant-context specialization, context parameters, final-layer direction
normalization, and shared 1x1 pseudo-parameter projection. It does not substitute
a plain MLP for that author implementation. Appendix E's plain small-weight
MADE is a separately named configuration. This also prevents identical
sigmoid components in the unconditional first coordinate. Stateless RNG and a
fixed balanced hidden-degree assignment are explicit reproducibility choices.

NeuTra `neutra.py:102-125` selects `L2HMCInitializer(factor=.01)`;
`neutra-utils.py:612-614` translates this to fan-in variance scale .02.
The source profile now preserves that initializer on every layer and preserves
TFP's block-mask convention, checked against its independent mask generator.
The normalized truncated-normal correction is inherited from TensorFlow's
variance-scaling definition. The optional Glorot/small-final setting is labeled
a local variant. Caps, learning rates and target-specific widths remain
uncalibrated hypotheses, not scientifically promoted defaults.

The NAF inverse is exercised through the actual weighted trainer; tempered
checkpoint construction/restore recognizes the new map and its affine wrapper.
The actual HMC geometry builder is tested, and its source closure now includes
the new authority. Compatibility weighted/RKL trainers use the existing bounded
fixed-signature graph cache and reject singleton updates.

The first CPU derivative check required a tighter *reference-test* inverse
tolerance: inverse error divided by finite-difference step exceeded its
derivative error allowance. The reference fixture uses 1e-14 tolerance; runtime
inverse controls were not relaxed. GPU-01 checked the earlier plain-MADE
engineering revision; after the author-conditioner/initializer refinements,
GPU-02 must check the final source. Both attempts together remain within the
five-minute engineering GPU allocation.

Three external parity tests require dsge_hmc commit
`d94566c9f70b3143e599a56eba7cb461ff2bda88`. The sibling checkout has advanced,
the pinned commit's tree is missing locally, and an isolated remote fetch
reports `not our ref`. Preserve that checkout and record this unavailable
external-reference check. Independent local equation/derivative tests and
existing artifact fixtures remain required; a missing reference is not a
passing parity result.
