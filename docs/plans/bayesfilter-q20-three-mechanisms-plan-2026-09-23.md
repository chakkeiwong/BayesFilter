# q20 nonlinear transport: three-mechanism implementation and test plan

Owner request: document the problem and proposed solutions in LaTeX, create
and review the implementation plan, and test the three mechanisms. This plan
implements optional numerical components and runs bounded analytic correctness
tests first. It does not interpret those fixtures as a trained q20 posterior.
The target remains the four-parameter q20/T30 UKF approximate posterior.

## Research intent and evidence contract

Main question: can free global scale, exact reverse-KL path gradients, and a
direct smooth scalar deformation remove identifiable obstacles to nonlinear
learning without changing the target or corrupting its Jacobian/score?

The baseline is the standard reparameterized reverse-KL derivative and the
frozen depth-four map audited on September 23, not an untrained identity map
or an earlier main-program stopping bug. An optional affine correction is a
control for the scalar deformation. No method-comparison campaign or ordinary
latent mass adaptation is introduced.

| Role | Criterion |
| --- | --- |
| Engineering pass | Analytic value, log determinant, inverse, input-score and parameter-gradient identities hold; TensorFlow/XLA graph signatures are stable; batches stay native; rejected invalid inputs do not update parameters |
| Promotion veto | Wrong derivative, missing Jacobian, nonfinite numerical state, inverse failure, missing scope/serialization support, or inconsistent checkpoint reconstruction |
| Continuation veto | Invalid target/fixture, corrupted checkpoint, unauthorized resource boundary, or exhausted assigned budget; a failed candidate alone is not a research-direction veto |
| Repair trigger | Saturated scale, noisy gradient, or weak nonlinear parameter directions with otherwise correct arithmetic |
| Explanatory only | Affine-fit fraction, gradient variation, local residual magnitude, training loss, and fixture timing |
| Later scientific success | Frozen-map qualification and stable posterior estimation under the existing q20 convergence, precision, and reference requirements |

The current artifacts are this plan, an updated LaTeX chapter section, source
and test files, a focused-test result note, and a versioned directory
`docs/plans/artifacts/q20-three-mechanisms-2026-09-23/implementation-01/`.
No scientific or GPU speed claim follows from CPU analytic tests.

## Implementation phases

1. Preserve the relevant chapter/bibliography baseline. Add a self-contained
   section to the NeuTra chapter with the measured q20 failure, RKL-versus-score
   distinction, source-backed mechanisms, efficient DSF forward/logdet and score
   formulas, inverse costs, numerical choices, and the proposed experimental
   sequence. Keep historical evidence and equations intact.
2. Implement optional TensorFlow components in a dedicated module, preserving
   existing production defaults and unrelated ongoing repairs:
   - a free diagonal affine layer with exact forward/inverse/logdet and score
     operations, initialized as identity around the existing map;
   - a reverse-KL path-gradient calculation with detached proposal/target scores
     and a parameter VJP, reporting the real RKL integrand separately from the
     gradient carrier;
   - a smooth full-range scalar sigmoid mixture, initially in one declared
     coordinate, with stable log-domain derivative formulas and bounded inverse.
     Nonidentical component locations expose nonlinear parameter directions.
3. Add tests against mathematical authorities: affine and Gaussian identities,
   a nonlinear transformed density, finite differences of values and logdet,
   parameter VJP checks, deterministic quadrature for equality of expected
   standard/path gradients, tail roundtrips, and a tiny batched optimizer smoke.
   Use a non-pfor derivative engine; ordinary reverse-mode VJPs suffice.
4. Review the final code against target/support, batching, detachment, tail,
   inverse and checkpoint risks. Compile the affected LaTeX section/chapter and
   inspect rendered pages. Record failures, fixes, tests, and exact remaining
   integration work.
5. Subsequent q20 campaign phase: load the actual frozen endpoint, price the
   optional mechanisms on trusted GPU/XLA, then run isolated scale, estimator,
   and scalar arms before combining them. Bind fresh streams, preserve optimizer
   state or explicitly reset it, retain checkpoints and the standard 1,000-point
   post-training report. Any new scalar map needs supported frozen payloads and
   public-tuner reconstruction before HMC. This phase must have an actual measured
   price and current campaign ledger before target runs begin; this plan does not
   invent a target-evaluation cost or reclaim spent diagnostic time.

## Numeric choices and assumptions

| Choice | Provenance, rationale, failure mode, smallest check, status |
| --- | --- |
| float64 | Inherited q20 dtype; isolates derivative error; roundoff still affects extreme sigmoid tails; finite differences and tail roundtrips; baseline |
| Identity affine correction | Derived map-preserving initialization, free log scale has derivative one; changes optimizer state only for new parameters; exact value/logdet identity; reviewed mechanics choice |
| Scalar coordinate 2 | Saved depth-four residual concentration, one-based convention; dependence on other coordinates may remain; scalar/affine controls and fresh residuals later; hypothesis |
| Mixture component count | Must be explicit, not a production default; one is exactly affine, at least two needed for nonlinear shape; test K=1 and distinct small K; convenience fixtures only |
| Positive slopes and normalized positive weights | Monotonic full-range DSF derivation; underflow/nonfinite logits can invalidate it; log-domain arithmetic plus finite guards; mathematical requirement |
| Inverse bracket | Derived from min/max of `(y-b_k)/a_k`; each component crosses the desired logit within it; verify bracket and roundtrip; mathematical construction |
| Inverse tolerance/iteration cap | Caller-explicit float64 absolute/relative tolerances and finite bisection cap; no silent approximate success; tolerance failure test and iteration count; not a q20 default |
| Fixture tolerances | Derived from float64 arithmetic or finite-difference truncation, recorded in tests; tighter numerical identities than noisy research metrics; mechanics-only |
| Fixture quadrature | Deterministic Gaussian quadrature with convergence cross-check; expectation equality, not per-batch equality, is required for path gradients; reference-only |
| Test seeds, dimensions, and tiny steps | Convenience reproducible fixtures; cannot establish posterior learning; declared at use; mechanics-only |

## Skeptical review before implementation

The following flaws were identified and addressed in this plan:

- An efficient log determinant is not the full cost of scores or inverse
  evaluation. Test input gradients and inverse separately; do not assign a
  GPU runtime from operation counts. Plain latent HMC needs forward/score each
  step, but ensemble cross densities can make inverse cost recur.
- Standard and path gradients need not match on a finite draw bank. Compare
  their expectations against analytic derivatives or converged quadrature;
  independently check the path VJP with detached scores.
- A zero-output network or identical sigmoid components may have no useful
  first-order nonlinear directions. Preserve near-identity but distinct
  components and explicitly inspect the nonlinear response.
- A scalar mixture's clipped sigmoid output would restrict support. Use exact
  log-domain sums and an inverse with an explicit convergence result.
- Free scale can cause overflow. Treat nonfinite values as rejection, not an
  excuse to clip the scientific target or drop rows.
- Analytic CPU/XLA tests cannot establish GPU cost, q20 improvement, production
  readiness, or coverage. Those claims remain with subsequent target phases.
- Current checkout contains unrelated changes. Add optional components rather
  than alter production defaults or reinterpret historical checkpoints.

Review verdict: proceed with documentation, optional components and bounded
mechanics tests. No material unexamined default is promoted. A new architecture
is a proposed training repair, not automatic authority for an HMC route.

Implementation review clarification: the first path-score engine forms only the
small parameter-space Jacobian through coordinate-wise VJPs, retaining all batch
rows. It implements the general change-of-variables equation, not the optimized
coupling-flow recursion. The tiny optimizer smoke uses explicitly documented
SGD to isolate the derivative; production Adam continuation, its slots and RNG,
and the frozen HMC codec are later integration tasks. Inverse bisection is
evaluation-only with stopped derivatives; forward-KL use is not supported.
After the first 13 tests passed, review strengthened the inverse criterion to
require both output residual and input bracket width, and added a pre-update
scale/weight representability guard. These repairs require the second focused
run, with the existing IAF and tempered-ensemble mechanics as regressions.

## Commands, limits, and stop conditions

Use `/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest` for the focused new
test file and directly affected regression files. Set `CUDA_VISIBLE_DEVICES=-1`,
`TF_FORCE_GPU_ALLOW_GROWTH=true` and `BAYESFILTER_TEST_DEVICE_SCOPE=cpu` before
framework import. GPU devices are deliberately hidden: this is the explicit
tiny analytic/reference exception, not NeuTra quality training. Numerical
kernels default to XLA and have stable batch/dimension signatures.

Convenience execution limit for this implementation pass: 20 CPU wall minutes
across focused test commands, at most three substantive repair/rerun cycles,
plus the LaTeX build. These are engineering limits, not statistical thresholds
or an extension of the scientific campaign. Stop and record if the fixture or
environment is invalid; narrow a failing test before rerunning a suite. No
q20 target evaluations or GPU training are included in this pass. The inherited
scientific ledger is unchanged until an explicitly metered campaign action.

## Completion and subsequent integration

Phases 1--4 are complete; see the
[implementation/test result](bayesfilter-q20-three-mechanisms-results-2026-09-23.md).
The final run passed 62 tests, including 16 new mechanism tests. The standalone
NeuTra chapter builds, its added section has been visually inspected, and the
final build has no overfull boxes. This is engineering correctness evidence.

Phase 5 is a subsequent target campaign, with these concrete integration tasks:

| File or interface | Required work before the respective use |
| --- | --- |
| `tempered_transport_ensemble_tf.py::IndependentTemperedReverseKLTrainer` | Wire an explicit standard/path choice into the existing Adam trainer while preserving its true loss, target-status veto, step, RNG and optimizer state; test segmented versus uninterrupted continuation |
| `capture_trainable_transport_checkpoint` / `restore_trainable_transport_checkpoint` in the same module | Represent the frozen base, optional pre/post composition, trainable parameter selection and scalar inverse controls; reject unsupported old/new mixtures rather than change the meaning of an old checkpoint |
| `neutra_training_protocol.py` and `q20_training_repair.py` | Add isolated scale/estimator/scalar arms with explicit optimizer migration, common fresh diagnostic banks and worker-time metering; retain the standard post-training 1,000-point report |
| `neutra_artifacts.py::load_frozen_neutra_artifact` | Add a supported reconstruction for a new scalar family, with roundtrip, Jacobian, score and target/source checks; existing artifact authority does not extend automatically |
| `tune_fixed_transport_hmc_kernel` | Use only the supported frozen payload, identity latent mass, and a fresh map-specific tuning/verification scope; no arbitrary-force substitution |

The baseline benchmark is the audited frozen depth-four checkpoint, whose exact
path and checksum are recorded in the earlier saved-analysis artifact. Before
the target phase, bind that actual saved state and the live budget ledger in the
run manifest rather than assuming that today's dirty checkout is its source.
Measure setup/compile, forward/score, and any recurrent inverse cost separately.
Allocate the affordable target batches and continuation rungs from that price;
the 32/64-node quadrature and three-step SGD smoke here are not such allocations.
