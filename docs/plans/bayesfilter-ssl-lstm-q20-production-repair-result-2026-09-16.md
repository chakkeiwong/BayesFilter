# q20 production repair result

Date: 2026-09-16

The private repair checkout now has an explicit q20 production entrypoint at
`docs/benchmarks/run_ssl_lstm_q20_production_2026_09_15.py`. Its `validate`,
`price`, and `train` modes consume one versioned protocol. Development runs
require the declared q20 target, batch-native TensorFlow transport training,
GPU/XLA policy, complete cohort checkpoints, heldout validation, and fresh
output directories. A shortened route is accepted only as a smoke role and
emits `promotion_eligible: false`.

The former two-updates-per-beta path is no longer the production protocol.
Each update records loss, target status, raw and clipped gradient norms,
clipping, optimizer iteration, beta, seed position, and wall time. Checkpoints
include map tensors, every Adam variable, iteration, root seed, RNG index,
beta, source/data scope and a checksum. Resume restores these values and the
next stateless seed; scope, source, checksum, and nonfinite state changes fail
closed. Beta-zero is analytic initialization and positive-temperature levels
carry Adam state.

The exporter writes the actual weighted dense IAF, permutations, scale
transform, and prior affine into the repository frozen artifact schema. Forward,
 inverse, log-Jacobian, and analytic score pullbacks are checked on nonzero
trained maps. Heldout paired loss and map reliability are separate from
posterior whitening; neither can certify coverage by itself.

The public candidate-set tuner is the only tuning authority in the new HMC
adapter. Verified members are retained and selected explicitly. The posterior
route uses the shared sequential controller, four chains, modern split/folded
R-hat, declared ESS and coordinate/quantile/event MCSE targets. The ensemble
route binds every temperature and chart, preserves replica identities, and
exposes only the beta-one stream. A missing reference remains incomplete.

Validation evidence from this repair checkout:

| Check | Result | Role |
| --- | ---: | --- |
| Shared HMC/NeuTra prerequisites | 68 passed | engineering regression |
| New q20 checkpoint/export/assessment tests | 11 passed | engineering acceptance |
| NeuTra and tempered transport regressions | 35 passed | numerical regression |
| Route-policy/ensemble/replica regressions | 28 passed | integration regression |
| q20 CPU smoke | bounded compile/resource timeout | unqualified diagnostic |
| trusted TensorFlow GPU probe | no idle permitted GPU | environment availability only |

The software is ready for a bounded development retry after GPU memory-growth
and XLA readiness are checked. It is not a validated q20 production campaign:
no q20 map has yet passed the declared learning ladder, public tuning, four
chain posterior precision, independent reference comparison, or confirmation
budget. The saved CPU smoke is therefore not evidence that NeuTra whitens the
q20 posterior.
