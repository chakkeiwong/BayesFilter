# FAB training of the canonical IAF

`bayesfilter.inference.neutra_fab` ports the author's FAB AIS, loss and
prioritized replay components into TensorFlow. It consumes `NeuTraTransport`;
it introduces no transport architecture. This implements an optional trainer,
not a new default or posterior admission.

Source: [Midgley et al., ICLR 2023](https://arxiv.org/abs/2208.01893), Section 3,
Equations 4–8, Algorithm 1 and Appendices A–C; [author fab-jax revision
c9f9913](https://github.com/lollcat/fab-jax/tree/c9f991366ca94b2678a7ed620bc9e12655cfef1d).
The local reference checkout is `.localresources/fab-jax-c9f9913` in the shared
repository. The complete MIT notice is retained in the derived Python module.

| Mechanism | Author source | TensorFlow behavior |
|---|---|---|
| Annealing target | `sampling/base.py::get_intermediate_log_prob` | `(1-2*beta)*log_q + 2*beta*log_p`; alpha fixed at 2 for this trainer |
| AIS order | `sampling/smc.py::step`, `ais_inner_transition` | Initial beta0→beta1 weight; mutation at each interior beta; weight to the next beta; no final-target mutation |
| HMC transition | `sampling/mcmc/hmc.py`, `blackjax_hmc_rewrite.py` | Batched identity-mass Metropolis-corrected leapfrog, native TF loops; optional source-style multiplicative step adaptation after each mutation |
| Metropolis transition | `sampling/mcmc/metropolis.py::build_metropolis` | Optional batched isotropic Gaussian random walk with symmetric MH ratio; step fixed across all mutations at a temperature, then adapted once from mean acceptance |
| Fresh weighted loss | `train/fab_without_buffer.py::fab_loss_smc_samples` | Mean after normalized weights, matching the pinned JAX implementation; the paper's displayed expression differs only by overall batch scaling |
| Replay loss | `train/fab_with_buffer.py::fab_loss_buffer_samples_fn` | Mean of detached `exp(log_q_old-log_q)` times negative log-density, with explicit optional correction cap |
| Replay selection | `buffer/prioritised_buffer.py::sample_without_replacement` | Gumbel top-k and shuffled distinct indices; without-replacement bias remains explicit |
| Replay initialization | `train/fab_with_buffer.py::init` | Fill `floor(replay_min_size/batch_size)+1` batches before the first update; the 40-batch configured minimum therefore requires 41 initialization passes |
| Priority refresh | `buffer/prioritised_buffer.py::adjust` | Add the uncapped log correction; save the pre-update density |
| Outer order | `train/fab_with_buffer.py::step` | Fresh AIS uses pre-update map, old replay drives optimizer updates, then fresh samples are inserted. Numerical scheduling is serial here; probability operations follow the same order |
| Initial invalid states | `sampling/smc.py::replace_invalid_samples_with_valid_ones` | Copies valid rows using the author's replacement operation; all-invalid batches fail closed because no replacement law exists |
| Target evaluation | JAX per-row `vmap(create_point)` | Existing batch-native TensorFlow value/score/status callback; no scalar row-mapped target |
| Optimizer | `train/fab_with_buffer.py` Optax Adam | `FABAdam` uses the same bias-corrected moment equations and post-bias-correction epsilon placement; direct FP64 state/update parity is recorded |
| Derivatives | JAX autodiff log-q and target | TF gradient of configured-map inverse density plus supplied exact first target score; no target Hessian required |

All numerical configuration is explicit in `FABConfig`. Betas are linear and
there are `intermediate_distributions + 2` endpoints including 0 and 1. No
resampling, arbitrary bounds, defensive mixture, extra reverse-KL loss, mass
adaptation or replacement flow is silently added. These are optional directions
for future reviewed changes, not implemented claims.

`transition_operator` selects `hmc` or `metropolis`; the original `hmc_steps`
field counts mutations per interior temperature for either operator.
Metropolis omits the spatial derivative of the flow density. The current
target callback still returns its existing score and validity status; no
replacement target evaluator is introduced. Random streams are reproducible
within this port, but are not bitwise identical to JAX streams. Common-draw
component comparisons and a bounded native-RNG law check are recorded separately.
The replay permutation sorts independent FP64 random keys because this
TensorFlow build cannot compile `StatelessShuffle` on GPU. Distinct iid keys
induce a uniform permutation; finite-precision ties are a negligible RNG
limitation at the declared minibatch counts. The complete replay path has its
own compiled GPU smoke in addition to CPU reference checks.

The q20 harness binds frozen maps to the beta-1 adapter signature used by the
existing configured NeuTra consumer. It records the underlying target and
bridge signatures separately. Initial HMC calibrations 01–03 used the older
underlying-target checkpoint binding; they contain zero training updates and
are not map handoffs.

The training proposal density and HMC arithmetic have the transport dtype;
the target callback receives FP64 coordinates. FP32/TF32 training therefore
retains the existing high-precision target arithmetic while using the finite
FP32 proposal states. Final map diagnostics evaluate the represented weights in
FP64, as in the existing canonical training procedure.

The exact AIS weighted-expectation identity assumes fixed invariant transition
kernels. Batch-dependent step adaptation is the author's practical training
heuristic; finite adaptive runs do not establish unbiased estimates. Use frozen
step sizes for reference/normalizer checks. Self-normalization, replay without
replacement and capped corrections also prevent an exact unbiased-gradient
claim. Corrections must not be normalized again as ordinary forward-KL weights.

`step(train=False)` performs AIS and can fill replay, but performs no optimizer
update. Each optimizer update is a compiled batch; host loops schedule outer
iterations and replay minibatches. Checkpoints include the map, Adam slots,
per-temperature steps, random counter and entire replay state. Restoring into a
different target, numerical configuration or seed is rejected. The executable
JAX comparison plan and result are
`docs/plans/bayesfilter-fab-equivalence-plan-2026-09-25.md` and
`docs/plans/bayesfilter-fab-equivalence-results-2026-09-26.md`. FP64 and FP32
without TF32 passed the finite common-draw and complete-iteration checks; TF32
exceeded the predeclared rounding screen on seven four-dimensional entries.
This qualifies TF32 numerically rather than showing a source-algorithm mismatch.

For handoff, freeze the represented map, run `PostTrainingProbe` on 1,000 fresh
points and preserve independent coverage evidence. No training result changes
the downstream posterior: NeuTra still targets
`log_p(T(z)) + log_abs_det_J_T(z)` with a fixed transport and identity latent
mass. Training AIS targeting `p^2/q` is not posterior sampling or a public HMC
tuning artifact.
