# M37 learned-map protocol and engineering boundary

The question is whether target-specific learned whitening improves usable
model-coordinate posterior sampling on the banana and separated mixture. The
current phase funds protocol construction and tiny composition tests. It does
not fund or establish a serious training-quality result. The 5,400 GPU-second
phase envelope is a ceiling, not a measured training price.

The selected call chain is `NeuTraReverseKLTrainer.train_step` through
`FixedShapeTrainingProgram` and the batch-native `ValidationTarget.log_prob_and_grad`,
then `frozen_transport_payload`, `load_frozen_neutra_artifact`, and
`tune_fixed_transport_hmc_kernel` via the validation pipeline. Its executable
check is `tests/inference_validation/test_training_pipeline_composition.py`.
The test uses one update of eight rows and a four-unit, two-layer IAF on CPU;
these are convenience sizes for shape, graph, inverse, serialization and public
retuning checks, not a recommended optimizer or capacity. CPU tests hide GPU
before import and cannot support GPU readiness. A separate bounded 600-second
GPU/XLA canary runs this exact call chain for both targets with verified memory
growth now that GPU 1 is available. Its one-update cost includes tracing and
retuning; it cannot price an adequately trained map or establish transport
quality or posterior accuracy. No external dataset is generated; the tiny
stateless base batch is a mechanics input. This path does not use the
concurrently modified Q20 training protocol. Skeptical review: preserve these
nonclaims and stop on graph, batch, inverse, reload or numerical invalidity.

For banana, use the declared law `x~N(0,1)`,
`y|x~N(b(x²-1),1)`. Analyze central and tail x separately, and bend values
zero, .5 and 1 as explicitly provisional regime probes. Exact unbending is an
analytic reference. Construct identity, fitted diagonal affine, fitted dense
affine and exact unbending comparators; each tests whether learning adds useful
geometry beyond a simpler transformation. Affine fit data are calibration data.

For mixture, preserve the declared two-normal law and exact mode probability
`w Phi(a)+(1-w)Phi(-a)`. Analyze near-overlap and separated modes, balanced and
imbalanced weights, and single-mode versus mode-dispersed starts separately.
Provisional regime probes are separations 2 and 5 and weights .3 and .5, drawn
from existing diagnostic cases. Construct identity, moment-matched diagonal
and dense affine maps, and exact independent mixture draws as the reference.
The exact sampler is a diagnostic authority, not an HMC transition. A smooth
invertible map may leave a bottleneck between modes; small reverse-KL loss
does not remove that failure. Mode-mass, tail and between-mode checks are
mandatory downstream criteria, alongside the M35 local-diagnostic adversary.

Both protocols require independent calibration, selection and untouched
posterior streams. Before serious execution, price the actual target-specific
batch-native GPU/XLA route with memory growth and retained posterior assessment.
Audit loss scaling, analytic value/score agreement and gradients first. Compare
affine/untrained, plain reverse-KL IAF and any enhanced objective as distinct
arms. A forward-KL/replay enhancement needs its own exact sampling/weight law;
it cannot inherit a Q20 objective. Architecture, activation, scale bound,
initialization, optimizer clipping and learning rate must all be explicit.

A candidate development grid is widths 8/16, depths 1/2, learning rates
1e-4/1e-3 and batch sizes 64/256 with three independent seeds. These are
**unreviewed search hypotheses**, not transferable defaults. They imply 48
training fits per target before downstream validation. Update ladders and
selection tolerances are unresolved until the first target-specific gradient,
cost and holdout-variance measurements; no arbitrary update cap is promoted
here. If an adequately replicated search and untouched downstream inventory
cannot fit the remaining allowance, stop training as underfunded and preserve
that uncertainty. No serious training launch is authorized by this draft grid.

Promotion requires numerical validity, frozen map inverse/Jacobian and score
agreement, fresh per-L tuning, model-coordinate reference agreement and the
declared posterior precision/delivery criteria with uncertainty. Conditional
underperformance against a simple comparator vetoes promotion. Training loss
and descriptive runtime only explain failures. A candidate failure can trigger
the next funded capacity/objective repair; it does not reject learned maps as
a research direction. Invalid density, unbounded cost, missing diagnostics or
exhausted allocation stop the affected experiment.

Skeptical review: the selected callable is batch-native and has an executable
composition check. The proposed regimes and hyperparameters remain hypotheses;
no external validity, comparative advantage or adequate training price has been
established. Tiny CPU tests are admissible engineering evidence. R7 stays open
until the pricing, target-specific protocol review and downstream studies exist.
