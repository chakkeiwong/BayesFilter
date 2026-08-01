# P2 Result: Scalar-Force Training And Campaign Harness

Date: 2026-07-17

Decision: `PASS_P2_TRAINING_HARNESS_ADMITTED`

## Outcome

P2 adds a target-specific TensorFlow scalar residual-potential trainer, frozen
force artifact/loader, complete transformed-target binding, disjoint-seed
checks, health-aware tuning nomination, and five-cell Tier A identity dry run.

The learned proposal potential is `0.5 * sum(z**2) + residual_network(z)`.
Its force is obtained by differentiating that scalar, not by an unconstrained
vector-output network. Training uses standardized force MSE plus centered
potential MSE. One XLA `tf.while_loop` executes all training steps; active
paths use neither NumPy nor Python sample/step loops.

Frozen artifacts bind target signature, transport signature, config,
normalization, topology, weights, and biases. Cross-target, cross-transport,
and tensor substitution fail closed.

Serious target binding requires a separate deterministic value-only endpoint
function and parity against the complete transformed value/score adapter. This
repair prevents the proposed sampler from computing an unused true gradient at
each endpoint. The complete scalar includes the transport log-Jacobian.

## Checks

- CPU/reference: `13 passed in 7.65s` after the value-only boundary repair.
- Trusted GPU/XLA: `3 passed in 8.24s` including P1 kernel canaries.
- GPU scalar-force training canary: 12 batched XLA steps in 2.54 seconds.
- Gaussian, banana, and funnel fixtures improved heldout standardized force
  RMSE relative to the zero-residual baseline.
- Conservative-force, offset invariance, frozen reload, signature substitution,
  transformed value/force/log-Jacobian, and health-aware tuning checks passed.
- All five Tier A rows resolved their registered target and frozen transport
  semantic identities.

## Repair History

| Attempt | Classification | Finding | Repair | Result |
| --- | --- | --- | --- | --- |
| CPU 1 | implementation | autodiff reduction target was created outside tape | explicit output gradients | target/force tests progressed |
| CPU 2 | XLA harness | conditional TensorArray writes had incompatible branch shapes | fixed per-step arrays plus graph-native heartbeat mask | all CPU tests passed |
| terminal audit | research-question boundary | endpoint binder used value/score and paid unused true-gradient cost | required distinct value-only endpoint plus parity gate | focused tests passed |

No experiment was launched before these repairs.

## Decision Table

| Field | Status |
| --- | --- |
| P2 primary criterion | passed |
| Scalar conservative force | passed |
| Batched GPU/XLA training | passed |
| Frozen reload and identity | passed |
| Complete transformed target | passed |
| Value-only endpoint boundary | passed on fixture; model-specific parity required in P3 |
| Disjoint evidence seeds | passed |
| Five Tier A dry runs | passed |
| Main uncertainty | target-specific recipe quality and downstream mixing |
| Next justified action | exact-likelihood LGSSM pilot |
| Not concluded | no learned-force model validity, convergence, speedup, superiority, or default readiness |

## Inference Status

| Field | Status |
| --- | --- |
| Hard veto screen | clear for P2 mechanics/identity fixtures |
| Statistically supported ranking | none |
| Descriptive-only differences | analytic heldout losses and canary runtimes |
| Default readiness | false |
| Next evidence | fresh LGSSM training/tuning/corrected chains and posterior/truth diagnostics |

## Post-Run Red Team

Small smooth two-dimensional fixtures understate the capacity and tail-coverage
problem of an 18-dimensional filter posterior. P3 therefore uses preserved
posterior-region NeuTra coordinates, disjoint training/heldout slices, a fresh
selected run, and corrected downstream HMC.

The terminal audit caught the most important possible false pass: a
mathematically correct endpoint wrapper could have paid for the full true
gradient and erased the intended cost advantage. Model phases must provide and
time a separate value-only endpoint and prove value parity first.

## P3 Handoff Review

The exact LGSSM likelihood and score target replays to target signature
`f47619320ded5f70259c6932eb2436642a02834c7a0249c7c52c20a5a2302f30`.
The registered NeuTra chart replays to semantic hash
`bcbe925f2ca77996bfe05cd5b951d1a66f540327789093d0ade8fecdf0773363`.
Preserved source evidence includes 2,000 warm-up and 4,000 retained draws per
chain in this chart. P3 may reuse disjoint coordinate slices for supervision,
but tuning, warm-up, and retained HNN seeds remain fresh.

The prior NeuTra step size `0.8` and 10 leapfrog steps are a warm-start
hypothesis, not a default. The likelihood is exact Kalman; the posterior is not
closed-form analytic, so posterior agreement uses the preserved tuned
plain-HMC reference plus generating-truth tail checks.

Status: `CONTINUE_TO_P3`.

