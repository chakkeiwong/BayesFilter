# NeuTra implementation authority

`bayesfilter.inference.neutra_transport_core` owns the numerical transport
operations. `neutra_transport` supplies the public configured map, Adam trainer,
and versioned checkpoint format. Ordinary, weighted, legacy, frozen, tempered,
and mechanism APIs retain compatibility facades that call this core. New
transport mathematics belongs in the core; a new target or campaign supplies
configuration and a batch-native target callback.

## Canonical architecture and historical boundary

**Owner directive, 2026-09-25: `bayesfilter_neutra_iaf_author_v1` is the canonical
NeuTra architecture. All superseded local implementations are HISTORICAL —
UNFAITHFUL TO THE AUTHOR'S CODE.** The
[migration notice](../plans/bayesfilter-neutra-canonical-architecture-policy-2026-09-25.md)
covers old constructors, configurations, checkpoints, archived copies and
results, including later compatibility facades that preserve those old maps.
The repaired IAF introduced in `8e132893e` and evaluated in the September 24
study is the current architecture; a blanket date cutoff is not its definition.

The canonical IAF uses these explicit choices:

| Property | Required mechanism | Current q20 instance / provenance |
|---|---|---|
| Numerical implementation | `NeuTraTransport` through `neutra_transport_core` | One authority for trainable and frozen evaluation |
| Scalar transformation | `y_j=x_j*exp(s_j(x_<j))+mu_j(x_<j)` | Invertible autoregressive affine stage |
| Conditional log scale | `neutra_conditional_tanh`: `b_j+c*tanh(h_j/c)` | Free bias outside the cap; c=2 is the recorded local choice |
| Conditioner | ELU; `hoffman_block_masks_v1` | Two hidden layers per stage; exclusive first mask, inclusive later masks |
| Initializer | `hoffman_variance_scaling`, `iaf_variance_scale=0.02` | Nonzero truncated-normal kernels including output kernels, zero biases; author initializer |
| Composition | Full coordinate reversal between stages | Three stages, matching the source architectural profile |
| Conditioner width | Explicit, source-compatible width, justified for the target | `(16,16)` for q20 dimension 4; this is a local width choice, not the paper's dimension-wide profile |
| Fixed outer affine | Explicit center and positive scales with provenance | Moments of 8,192 saved-baseline draws in the completed study; an initialization hypothesis, not posterior moments |

`NeuTraTransportConfig.hoffman_author_iaf` selects the defining source options;
its hidden widths default to `(dimension, dimension)`. The q20 study explicitly
used `(16,16)`. Direct construction of `NeuTraTransportConfig(kind="iaf", ...)`
does **not** by itself select the author masks: its compatibility default is
`legacy_degree_masks_v1`. Use the source profile or specify every defining
choice explicitly. Generic constructor defaults are not architectural policy.

The accompanying q20 training recipe used path gradients, batch 32, learning
rate 0.01, Adam `(0.9,0.999,1e-8)`, calibrated emergency clipping, 4,096 updates
per seed, FP32 transport/Adam with TF32 enabled and an FP64 target. These are
recorded target-specific settings. They are not universal author defaults or
proof of sufficient training on another target. Path gradients are the separately
cited Vaitl/Roeder mechanism, and the TensorFlow port uses stateless seeds and
grouped output storage. Thus canonical means the owner-designated, documented
source-based architecture, not bitwise reproduction of one paper experiment.

The configured full conditional DSF/NAF remains a separately identified research
alternative with the source correspondence below. The preliminary scalar canary
is historical and must not be called that full NAF. Old affine/dense artifact
schemas and legacy scale conventions remain readable for historical purposes;
conversion or successful checks cannot relabel them as canonical.

Any departure requires a written, substantive explanation before implementation
or execution: exact source/configuration difference, demonstrated need,
mathematical consequences, bounded validation against the canonical setup,
criteria, stop conditions and skeptical review. The migration notice specifies
the requirements. Preserve every necessary local adaptation explicitly, and do
not replace the canonical architecture without owner direction.

This policy selects an implementation. The completed study supports local
training improvement for its IAF fits; posterior coverage and HMC convergence
remain unproved. Old-map continuation also improved, so the study does not
attribute all past failures or present gains to architecture alone.

## Source correspondence

The inspected sources are preserved in
`.localresources/q20-flow-training-literature-20260923/`. The implementation
combines distinct published mechanisms; it is not a reproduction of a single
paper's experiment or evidence of a well-trained q20 posterior map.

| Mechanism | Source | Implementation and explicit differences |
|---|---|---|
| NeuTra IAF architecture | Hoffman et al. 2019 §4.1.1; `code/neutra-utils.py:695-770` | `hoffman_author_iaf`: three layers, two dimension-wide ELU layers, coordinate reversal; author TFP block masks, exclusive first layer and inclusive later layers. Output heads are grouped in one matrix. |
| Conditional cap with free scale | Author `neutra-utils.py:739-763` | `iaf_parameters`, `neutra_conditional_tanh`: scale is `b+c*tanh(h/c)`; final scale-output bias is outside the cap. No final clamp or sigmoid. The cap is caller-supplied. This is an author-code option, not a formula asserted to describe all paper experiments. |
| NeuTra initialization | `code/neutra.py:102-125`, `code/neutra-utils.py:612-614` | All IAF kernels use fan-in variance scaling with scale `2*0.01=0.02`, truncated-normal correction, and zero biases. Stateless seeds replace TF1 global RNG; array layout groups scale/shift heads. |
| Full conditional sigmoid-mixture NAF | Huang et al. 2018 §3.1 Eq.(8), monotonicity/universality theory, appendices C and E; `code/naf-flows.py:221-317` | Every coordinate has conditional slopes, shifts, and weights. `naf_dsf` implements DSF, one scalar hidden layer per autoregressive stage, and arbitrary composition of stages; it is not DDSF. |
| Author NAF conditioner | `code/naf-iaf_modules.py:46-80,135-177`, `code/naf-nn.py:96-132`, `code/naf-flows.py:238-258` | `author_cmade`: cMADE, hidden unnormalized CWNlinear directions, normalized final CWNlinear directions, context scales/biases and a shared 1x1 output projection. Context is the constant vector `[1]` for an unconditional target. Balanced hidden ranks are fixed to a valid source assignment; parameter arrays use input-by-output and grouped-coordinate storage. Masks are applied after normalization. The first coordinate receives constant-context features, avoiding identical sigmoid components at initialization. |
| NAF initialization | Huang appendix E; author `naf-flows.py:251-258`, `naf-nn.py:108-113` | Source conditioner directions and context weights start at Normal(0,0.001); context-linear biases retain their source uniform initialization; projection weights are uniform ±0.001 and its bias sets unit slopes. Stateless seeds replace source global RNG. An explicit `paper_made` configuration implements appendix E's small-weight plain MADE instead; it is not the author cMADE experiment. |
| NAF positivity | Huang Eq.(8), appendix C.2.1; author `naf-nn.py` softplus helper | `a=softplus(raw)+1e-6`, `w=softmax(logits)`. The source floor is recorded in config and can be changed only as an explicit numerical hypothesis. |
| Full real support | Huang Eq.(8), appendix C log-domain calculus | Log-domain mixture evaluation omits author `SigmoidFlow`'s output shrinkage. Shrinkage would bound the transform's range and exclude posterior support. This is an explicit algebraic implementation of the paper equation, not bitwise reproduction of the clipped author code. |
| Path gradient | Vaitl et al. 2024 §3.1 Prop.3.2 Eq.(16), appendix B.1; Roeder et al. 2017 | Layerwise forward score propagation and stopped-score parameter VJP, actually used by `NeuTraTransportTrainer(estimator="path")`. One forward per layer, triangular solves for IAF/NAF. The coupling-specific linear-cost algorithm is not claimed. No official Vaitl implementation was found in the bounded source survey. |

The optional `glorot_small_final` IAF initializer is explicitly a local variant;
the source profile uses the author's variance scaling on all kernels. No
initializer is a calibrated q20 training protocol. Legacy initializers remain
unchanged in their compatibility constructors.

## Equations and derivatives

For a fixed coordinate and its earlier inputs, write
`u_k=a_k*x+b_k`, `S=sum(w_k*sigmoid(u_k))`, and `C=1-S`.
The map is `y=log(S)-log(C)`. Compute `logS` and `logC` by log-sum-exp
of `logw+logsigmoid(u)` and `logw+logsigmoid(-u)` respectively.
With `N=sum(w*a*sigmoid(u)*sigmoid(-u))`,

\[
\log\frac{\partial y_j}{\partial x_j}
=\log N_j-\log S_j-\log C_j.
\]

The conditioner depends only on earlier coordinates, so these are the diagonal
entries of a triangular Jacobian. Their sum is the full log determinant even
though the off-diagonal derivatives also include conditioner derivatives.
Positive slopes and weights give a positive diagonal. For finite parameters
and strictly positive weights, each scalar map tends to ±infinity in its two
tails; no output clamp is used. Floating-point nonfinite or underflow-invalid
parameters propagate invalid values instead of silently changing the map.

To invert one coordinate, the extrema of `(y-b_k)/a_k` bracket its root:
at the lower endpoint every sigmoid is at most `sigmoid(y)`, and at the upper
endpoint every sigmoid is at least that value. TensorFlow bisection stops only
after both output residual and input bracket tests pass. The outer coordinate
solve is another TensorFlow loop. An exhausted inverse yields nonfinite output.
The default tolerances are `atol=rtol=1e-11` for float64 and sixteen FP32
machine epsilons (`1.9073486328125e-6`) for float32, with at most 100 bisections.
These are checked engineering settings, not guarantees for every tail or scale;
controls and acceptance tests are serialized. A tighter caller-supplied
tolerance still rejects if the selected precision cannot satisfy it.

## Training and evaluation precision

`NeuTraTransportConfig(..., dtype="float32")` places the configured IAF/NAF
weights, activations, derivatives and Adam moment arrays in FP32. Both source
profile constructors accept the same explicit `dtype` argument. The default
remains float64 for compatibility with existing callers and saved maps; it is
not a mathematical requirement of either method. Historical configurations
without the new field retain their original float64 meaning.

`NeuTraOptimizerConfig(target_dtype="float64", ...)` selects target-call
precision independently. The trainer explicitly casts stopped physical points
before calling the batch-native target. Target values, scores and reported
loss retain target precision; the parameter pullback casts back to the flow
dtype. Standard and path estimators use the same boundary. The target callback
supplies first scores and is never differentiated for a Hessian.

TF32 is a process-level execution choice for eligible FP32 matrix products,
not a dtype. Set and record TensorFlow's TF32 policy before tracing the
numerical graphs. FP64 matrices do not use TF32. The library does not mutate
this global setting while constructing an individual map. The weighted
forward-KL consumer also accepts configured FP32 maps; historical training
facades preserve their saved FP64 configurations. The optional serialized
`affine_center` and positive `affine_scale` compose a fixed outer affine map
through the same shared primitives, including its log determinant and score.

`transport.as_dtype("float64")` makes an explicit frozen evaluation copy with
the represented trained weights; it does not resume or convert Adam state.
Precision conversion is included in the frozen artifact provenance. Changed
rounding requires fresh evaluation. `trainer.finalize(...)` runs the standard
1,000-point diagnostic on this FP64 copy and exports that same mathematical
configuration for the existing FP64 target/HMC consumer. Its report records
both training and evaluation precision. Finite diagnostics still do not grant
training-quality or posterior status. Same-dtype training resume preserves
the complete optimizer; cross-dtype checkpoint resume is rejected.

Inverse differentiation uses the implicit equation `T_phi(x)=y`:

\[
\frac{\partial x}{\partial y}=J^{-1},\qquad
\frac{\partial x}{\partial\phi}=-J^{-1}\frac{\partial T_\phi}{\partial\phi}.
\]

The custom gradient solves the transposed triangular system and applies a
parameter VJP. It does not differentiate the bisection decisions. The log
determinant is then differentiated at that inverse. This matters for weighted
forward-KL training and cross-component densities in a tempered ensemble.

For path gradients, at each layer `x_next=T_l(x)`, the change-of-variables
identity gives

\[
s_{l+1}=J_l^{-T}\{s_l-\nabla_{x_l}\log|\det J_l|\},\qquad s_0=-z.
\]

The parameter gradient of reverse KL after dropping the zero-mean explicit
parameter score is the VJP of `T_phi(z)` with stopped cotangent
`s_q(T_phi(z))-s_target(T_phi(z))`. The implementation builds each layer's
Jacobian with a TensorFlow loop over coordinate VJPs, never pfor or a sample
loop, and reuses that layer's forward evaluation. Autoregressive Jacobian
storage and solves are quadratic in dimension; this is not Vaitl's specialized
coupling O(d) algorithm. General fixed mixing layers use a dense solve.

The reported loss is `mean(-log_target(T(z))-logdet)`, reverse KL up to the
parameter-independent Gaussian base term and unknown target normalizer. The
gradient carrier is reported only internally as a differentiated expression;
it is not substituted for the loss. Target values and scores are opaque first
order inputs, so no target Hessian is evaluated. Path gradients have no
universal variance advantage away from a fitted map.

## Public use and handoff

```python
from dataclasses import replace
from bayesfilter.inference.neutra_transport import (
    NeuTraTransportConfig, NeuTraTransport,
    NeuTraOptimizerConfig, NeuTraTransportTrainer,
)

# Canonical IAF source profile; numerical choices belong in the target plan.
map_config = NeuTraTransportConfig.hoffman_author_iaf(
    dimension, conditional_scale_cap=conditional_scale_cap,
    seed=initialization_seed, dtype="float32",
)
# q20 uses (16, 16); other targets must justify their selected widths/moments.
map_config = replace(
    map_config, hidden_layers=hidden_layers,
    affine_center=affine_center, affine_scale=affine_scale,
)
transport = NeuTraTransport(map_config)
optimizer_config = NeuTraOptimizerConfig(
    batch_size=batch_size, estimator="path", learning_rate=learning_rate,
    beta1=beta1, beta2=beta2, epsilon=adam_epsilon,
    gradient_clip_norm=calibrated_clip_or_none,
)
trainer = NeuTraTransportTrainer(
    transport, target_value_score, optimizer_config,
    target_signature=target_signature,
)
result = trainer.train_step(latent_batch)
# result["valid"] must be checked; rejected steps leave parameters/Adam intact.
checkpoint = trainer.checkpoint()  # Includes Adam moments and iteration.
handoff = trainer.finalize(bridge, beta=1., diagnostic_seed=fresh_seed)
```

The final handoff runs the standard 1,000-point Gaussian-score residual probe.
It requires complete finite evaluation of all rows. Finite large residuals are
repair evidence, not convergence evidence or an automatic reason to abandon
training. DSF diagonal log derivatives are distinguished from IAF conditional
scale saturation; a DSF has no conditional tanh cap.

`load_frozen_neutra_artifact` reads the new configured schema and checks its
target and full configuration/parameter hash. It constructs constant tensors
and disallows parameter restoration once bound. Its actual consumer path is
`_rebuild_geometry` → frozen loader → configured transport → common core →
`FixedTransportValueScoreAdapter`. The HMC execution source closure includes
both new modules. The public fixed-transport tuner continues to use identity
mass in latent coordinates; this implementation adds no mass adaptation.

The configured map can also be supplied to
`WeightedForwardKLNeuTraTrainer(config, transport=transport)` and to the
independent/joint tempered trainers. Tempered checkpoint structures recognize
both the configured map and a reference-affine wrapper. Their existing
objective and campaign procedures remain distinct configuration choices.

All serious training still requires trusted GPU access, memory growth configured
before TensorFlow import/initialization, native batched target evaluation, XLA
checks, a target-specific calibration/training plan and downstream validation.
The trainer's fixed input signature enforces its batch size. Python loops over
static layer lists are tracing-time architecture construction, not Python
sample or leapfrog execution.

## Compatibility and drift prevention

**Compatibility preserves old semantics; it does not confer canonical status.**
The old map constructors and artifacts below are historical and unfaithful to
the author-code architecture. Shared consumers may use the explicitly configured
canonical IAF, while old default transports remain historical. Do not copy their
settings into a new campaign or treat earlier reported readiness as current.

| Existing caller | Shared numerical route |
|---|---|
| `neutra_training._TrainableDenseIAF` | `iaf_forward`, `iaf_network_diagnostics` |
| `neutra_training_legacy.TrainableDenseAutoregressiveIAF` | `iaf_forward` |
| `neutra_weighted_training._DenseAutoregressiveStage` | IAF parameters, forward/inverse, pullbacks and logdet score |
| `neutra_artifacts._DenseAutoregressiveIAFComponent` | The same IAF kernels with constant tensors |
| Composed and affine facades | Common composition, affine and permutation primitives |
| `neutra_training_mechanisms` canaries | Common sigmoid mixture, inverse, affine and RKL/path-gradient kernels |
| Scale-aware training / q20 sessions | Their ordinary or weighted transport facade and frozen reader call the core |
| Configured IAF/NAF / frozen configured map | `AutoregressiveStage`, common masks, composition, score propagation |
| Reference-affine / tempered ensemble | Common core map under the wrapper; common inverse density |

Historical `bounded_tanh` remains `c*tanh(h/c)`, `dsge_bounded_tanh` remains
`c*tanh(h)`, and `identity` remains unbounded `h`. They are explicit saved
conventions, not aliases for the new free-bias option. Old arrays, variable
ordering and artifact hashes retain their interpretation. The frozen
`dsge_bounded_tanh` derivative previously omitted `c`; that derivative is
corrected while its forward map is preserved. Prior score-based evidence for
that affected variant needs rechecking; forward-map parity cannot validate the
old score. Source-bound execution artifacts must be regenerated after source
changes; do not restamp old results as new evidence.

`tests/test_neutra_single_authority.py` checks caller wiring and discovers new
IAF/autoregressive classes across runtime modules to catch copied network
equations. It also checks independent equations, nonidentity derivatives,
source masks/conditioner operations, tails, inverse derivatives, estimator
behavior, optimizer continuation, invalid-input rejection, the actual HMC
adapter path, weighted/tempered consumption and the 1,000-point procedure.
This guard complements code review; renaming a new numerical fork cannot make
it an authorized implementation.

## q20 posterior event monitoring

The q20 posterior policy declares `positive_theta_2` in
`HMCPosteriorAssessmentPolicy.binary_quantity_names`. Event probability uses
rank split R-hat, split indicator ESS, and event MCSE. Rank normalization is
affine on a binary variable, so its bulk ESS is the same indicator ESS.
Continuous parameter coordinates retain max(rank, folded rank) R-hat and
continuous quantile-tail ESS. Binary quantile-tail ESS can be undefined, and
folding an exactly balanced binary sample gives a constant; the raw diagnostics
remain in the report rather than being relabeled as event information.
Both outcomes must be observed, and nonbinary or stuck event data cannot pass.
