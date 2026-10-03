# q20 three-mechanism implementation and test result

The problem and three proposed solutions are now documented in the NeuTra
LaTeX chapter. Optional TensorFlow implementations pass **62 tests**, including
16 new mechanism tests and 46 existing IAF/tempered-ensemble regressions.
This establishes the tested arithmetic and rejection behavior. It does not
establish successful training on q20 or a production-ready new transport family.

The [plan](bayesfilter-q20-three-mechanisms-plan-2026-09-23.md) contains the
pre-implementation skeptical review, evidence contract, numerical provenance,
test budget, and next integration tasks. The
[new LaTeX section](../chapters/ch26b_neutra_training_repair.tex) is included
from the existing NeuTra chapter. The
[compiled chapter](artifacts/q20-three-mechanisms-2026-09-23/implementation-01/build/chapter-review.pdf)
includes the added section on pages 5--11.

## What is implemented

`bayesfilter/inference/neutra_training_mechanisms.py` provides:

- `FreeDiagonalAffine`: explicit free global location/log scale, identity
  initialization when all supplied parameters are zero, and exact inverse and
  log determinant. Exponent underflow/overflow is invalid.
- `ScalarSigmoidMixture`: a smooth full-range scalar map in one explicitly
  selected coordinate, with stable forward/logdet and analytic logdet-score
  formulas. Bisection starts from a derived bracket and succeeds only when
  both output residual and input bracket width meet explicit tolerances.
  Failure returns a false status and nonfinite rows. The inverse deliberately
  supplies no gradients and is not a forward-KL training implementation.
- `ComposedMechanism`: pre- or post-composition with an existing map, with
  explicit selection of trainable components. Freezing the parent preserves
  its input derivatives.
- `MechanismReverseKLTrainer`: standard and path-gradient evaluators with the
  same reported true RKL loss, stable batch signatures and XLA enabled. The
  target supplies its value, first score and row validity. Invalid rows veto
  the whole update; they are never filtered out. A small coordinate-wise VJP
  engine computes the proposal score without pfor or sample-wise target calls.
- Mechanism parameter serialization/reconstruction that explicitly does not
  issue an admitted frozen HMC artifact.

The tiny optimizer is SGD, used only to check that the supplied gradient can
drive a batched update and that rejection preserves state. The existing
production Adam trainer, checkpoint format, master program and HMC codec were
not changed. The plan identifies their required integration work explicitly.
The whole-Jacobian path-score implementation is appropriate to this small
correctness exercise; its cost has not been compared with a layerwise GPU
implementation.

## Tests and implementation review

The first run passed 13 new tests. Post-implementation review identified two
extra numerical risks: a small output residual can mask large inverse-coordinate
error in a flat region, and a finite optimizer proposal can make an exponential
scale overflow or underflow. The implementation now checks the inverse bracket
width and the proposed scale/weight representability. Three further tests cover
these issues and an explicit finite-value target-status veto.

The final 62-test run passed with no failures or skips. The new tests cover
exact affine identities, scale derivatives beyond the inherited cap, scalar
value/logdet derivatives against finite differences, analytic logdet-score
agreement, XLA inverse roundtrips through inputs of magnitude 1,000, inverse-cap
failure, one-component affine behavior, non-affine response, identical-component
initialization degeneracy, and parameter-state reconstruction. These tail points
are deliberately extreme analytic fixtures, not posterior probability estimates.

For the estimator, an exactly matching Gaussian gives zero path gradient on
each tested point while the standard finite-batch gradient remains nonzero.
A nonlinear scalar map with a quartic target passes 32/64-node Gauss--Hermite
convergence checks: expected standard and path gradients agree within the
declared numerical tolerance and with finite differences of the same integrated
objective. A composition with the existing dense IAF also matches the proposal
score obtained independently by differentiating its inverse density. Tiny
standard/path optimizer smokes preserve batching and do not call the scalar
inverse. Invalid inputs/status and invalid proposed scales leave parameters and
the optimizer iteration unchanged.

Tests ran in the installed `tfgpu` conda environment with GPUs deliberately
hidden, memory-growth environment set before imports, and two CPU framework
threads. New numerical kernels use XLA; independent finite differences and
quadrature are diagnostic references. The two test commands took approximately
8.54 and 21.46 seconds of CPU wall time. Third-party TensorFlow/TFP Python
deprecation warnings were emitted and are preserved in the logs; there were no
test errors. No target evaluations, GPU checks, q20 optimizer updates or
scientific-campaign budget charges occurred.

The LaTeX build succeeded after fixing a long equation, long path text and an
overlong running heading. Pages 5--11 were inspected in the rendered PDF.
The final build has no overfull boxes and resolves every new citation. The
standalone chapter still has two references to sections in other chapters;
those are outside the isolated build, not missing new q20 equations. The
existing chapter's equations and results and all existing bibliography entries
were preserved; the only replaced chapter prose adds the September investigation
to its introduction. This is an author review of the rendered text, not a
claim of independent or human final acceptance.

## Decision

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | What is not established |
| --- | --- | --- | --- | --- | --- |
| Retain optional mechanisms for integration | All 62 analytic/regression tests pass | No fixture validity or numerical rejection defect remains in tested cases | GPU cost and behavior at the actual saved q20 endpoint | Integrate with Adam/checkpoint handling, then price the trusted GPU canaries | q20 learning, posterior correctness, performance improvement or HMC readiness |
| Keep inverse-based use restricted | Root accuracy and failure behavior checked; inverse gradients absent by design | Forward-KL/inverse-gradient use remains unsupported | Cost of repeated ensemble cross densities and an implicit inverse derivative | Measure inverse demand before adopting a scalar family for the full ensemble | Cheap bidirectional flow or supported frozen HMC payload |

No stochastic candidate comparison was performed. There is no statistically
supported ranking, no descriptive claim that one mechanism improves q20, and
no change of production default. The strongest remaining alternative is that
ordinary IAF continuation with better calibrated optimizer settings would
suffice. Fresh target-specific evidence is needed to distinguish that from
parameterization and gradient-noise explanations.

Commands, logs, baseline copies, source hashes and build records are under
`docs/plans/artifacts/q20-three-mechanisms-2026-09-23/implementation-01/`.
