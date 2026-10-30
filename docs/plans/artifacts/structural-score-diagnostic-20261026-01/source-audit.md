# Model B score call-chain audit — 2026-10-26

Base commit: `a54add9e78678c5bd5a36ba89164586bd97dab34`. Branch:
`structural-score-diagnostic-20261026`. This audit covers the structural
Model B consumer and its shared batched implementation, not every other
consumer of LEDH. No production implementation was changed.

## Claimed and computed quantities

The tested model has two state coordinates, one stochastic innovation and
one observation. It estimates q=(atanh(rho),log(sigma),log(beta)); the
deterministic persistence 0.55 and observation standard deviation 0.30 are
fixed. The initial law is m0~N(0,0.25), k0~N(0,0.20), independently and
independently of q. All reported scores are derivatives in q coordinates.

For fixed draw design u, the structural endpoint returns ell_N(q;u), the
log likelihood produced by its particle program, and an analytical Jet
score s_N. The checked claim is s_N=d ell_N/dq along a valid smooth
execution path, interpreted in real arithmetic with floating-point error.
It is not a claim that ell_N equals the exact nonlinear-model log likelihood
ell, nor that s_N equals d ell/dq. Floating-point rounding itself is not
literally differentiated. Finite differences test numerical consistency,
not a derivative of the discontinuous rounding map.

## Executed call chain

1. `bayesfilter/testing/structural_ledh_target.py:31` constructs
   `make_batched_score_filter` with fixed design, controls, and dtype. Its
   `likelihood` at line 56 calls that program and checks both validity and
   `steps_completed == horizon`. `value_score` adds the declared q-prior.
2. `bayesfilter/highdim/covariance_proposal_score_batch_tf.py:236` builds
   the fixed-design wrapper and shared numerical program. Both use explicit
   TensorSpecs and `tf.function`. The observation recurrence at line 350
   uses `tf.while_loop`. The diagnostic calls this production consumer,
   rather than a copied derivative implementation.
3. `bayesfilter/highdim/structural_completion_proposal_tf.py:43` proposes
   the scalar innovation conditional on a fixed selected parent. All three
   mixture densities use that same parent. `complete` then constructs k
   deterministically; no nonsingular two-dimensional transition density or
   artificial process noise for k is introduced.
4. The shared recurrence forms the exact innovation importance ratio for
   the implemented proposal, normalizes weights, and carries value/tangent
   pairs through densities, moments, transport and reset.
   `reset_cloud` at line 197 performs transport and covariance recolouring;
   `repair_moments` is called at line 315 when the configured correction
   is enabled. The diagnostic includes that correction.
5. `bayesfilter/highdim/analytical_jet_tf.py` implements explicit
   first-order dual arithmetic. The checked numerical path does not call
   GradientTape, vectorized_map or pfor. Its horizon, transport blocks and
   correction iteration loops use TensorFlow loops. Python loops in the
   diagnostic orchestrate parameter perturbations and independent runs;
   they are not Python loops over the production filtering recurrence.
6. `StructuralTarget.hmc_components` at line 76 binds this consumer. The
   Model B endpoint uses its finite LEDH target; Model A has a separate
   exact Kalman endpoint. The FP64 HMC position is cast to the consumer
   dtype, then returned value/score are cast back to FP64. The executed
   tests exercise `likelihood` and the shared program; HMC endpoint wiring
   was read, but a new HMC wiring/parity test was not run in this task.

The GPU T50 diagnostic log records an XLA-compiled cluster and the executed
consumer requests XLA. This does not mean the whole Python campaign is XLA:
data preparation, reference integration, serialization and the explanatory
CPU reset wrapper are outside XLA. The FP64 diagnostic uses a CPU graph
without XLA.

## Why the bootstrap reference has different score formulas

`structural_reference_diagnostics.py:70` uses Fisher's identity in latent
coordinates (m0:T,k0). Conditional on this m path,

    k_t = .55*k_(t-1) + beta*tanh(m_t),
    e_t = (m_t-rho*m_(t-1))/sigma,
    D_t = d k_t/d log(beta) = .55*D_(t-1)+beta*tanh(m_t).

Differentiating the complete-data log density with that path held fixed
gives the three additive scores implemented in lines 86–88:

    S_rho   = sum e_t*m_(t-1)*(1-rho^2)/sigma,
    S_sigma = sum (e_t^2-1),
    S_beta  = sum (y_t-m_t-k_t)*D_t/.09.

The prior contributes no score because it is parameter independent.
Differentiating the marginalization integral gives
`d log p_q(y)/dq = E_q[S_complete | y]`, under differentiation/integration
interchange. The Gaussian densities, positive noise scales, finite horizon
and bounded tanh allow this locally on interior compact parameter sets.
The bootstrap smoother approximates this conditional expectation by
weighted resampled paths. In these coordinates it is correct that the
observation contribution is explicit only for log(beta); rho and sigma
enter through transition density scores. Adding pathwise rho/sigma
derivatives of k to these formulas would mix differentiation conventions.

The formulas therefore pass the dependency audit. Their finite-particle
approximation is not an exact oracle. Normalized weighting and repeated
ancestor selection leave finite-particle bias and potentially poor
historical path diversity. The code records filtering minimum ESS but
does not record unique surviving ancestors or smoothing-score uncertainty
over time. These are reference limitations, not measured causes of the
observed T50 gap. A larger bootstrap reference alone is not a proof.

## Additional issues and their scope

- **Reset changes nonlinear predictions.** Exact mean/covariance restoration
  does not fix expectations of the nonlinear next-observation density.
  The measured weighted-before/uniform-after difference is substantive.
  This does not violate deterministic completion: that identity is enforced
  conditional on each parent at the next transition.
- **Unneeded terminal reset can veto a likelihood.** The program always
  resets and checks its cloud after calculating an increment, including
  the final time. A final reset failure can invalidate an otherwise finite
  accumulated likelihood even when only value/score are requested. This
  possibility is established by source inspection, not reproduced in these
  runs: all retained T50 checks pass. Skipping the terminal reset would
  require a separately tested likelihood-only mode because some consumers
  need terminal particles. It was not silently changed here.
- **Mixed precision requires careful checking.** FP64 HMC positions are
  rounded to FP32 at the target boundary. The short-horizon FP64 comparison
  finds tiny value/score changes, but it does not certify T50 rounding or
  the full posterior domain. Small FP32 finite differences are unreliable.
- **Replication recompiles targets.** Each StructuralTarget instance
  closes over a different fixed design and creates fresh graphs. Replicate
  orchestration produced retracing warnings. This affects campaign cost;
  a single frozen HMC target is reused and does not imply per-step retracing.
- **Diagnostics were discarded by the new runner.** The first runner saved
  only one of six computed T50 finite-difference rows and hardcoded FP32 in
  its FP64 manifest. Both reporting defects are repaired in the script.
  Original evidence is preserved; metadata corrections are explicit.
- **Environment/diagnostic invocation issues.** The isolated checkout
  needed its custom Sylvester op built. This TensorFlow build reports no
  native compute-capability-12.0 kernels and uses PTX JIT. Eager diagnostic
  random/Jet operations hit CUDA invalid-handle errors; invoking transport
  eagerly also exposed TensorArray read semantics. Fixed design generation
  and a traced reset diagnostic were moved to CPU. The production GPU
  recurrence passed; none of these failures establishes a mathematical bug.
- **Paper-file integrity.** Both locally stored `poyiadjis2011*.pdf` files
  begin with HTML and cannot be parsed as PDF. They were not used as
  evidence for this result; the reference-score explanation above is a
  direct local derivation. Repairing that literature archive is separate.

## Evidence boundary

Executed finite-program checks support the derivative at the tested truth
point and fixed designs. They do not establish derivative agreement at
every posterior point, convergence in particle count, correctness of the
exact-model posterior, or global generality across unrelated LEDH lanes.
