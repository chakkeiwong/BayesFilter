# Saved q20 NeuTra post-training geometry

Owner request: run the diagnostic and establish what can be said about the
trained maps. This phase extends the existing authorized training-repair
campaign; it does not allocate new compute or change its target.

## Question, evidence and decisions

Do the four saved repaired maps execute valid transformed scores, and how large
is their departure from standard Gaussian geometry on a common small base
bank? The comparator is their original shared starting map, not a separate
sampler. The target remains the saved q20/T30 float64 UKF posterior
approximation at beta one. Restore each map without an optimizer update.

For theta=T(z), the transformed score is
`s_z = J_T(z)^T s_theta(T(z)) + grad_z log|det J_T(z)|`.
Compute `r(z)=s_z+z` using the already tested shared batch diagnostic. An exact
standard-normal pullback has r=0. Report per-coordinate RMS, maximum row norm,
centered log-density residual and per-layer scale diagnostics. Evaluate all
five maps on the same 32 standard-normal latent points. These points map to
different physical locations after training; the comparison describes local
Gaussianization, not posterior coverage, map ranking or causal attribution.

Finite scores, valid target status and matching immutable checkpoint/target
identities are hard screens. Nonfinite values reject that map's readiness and
trigger repair. Identity mismatch, incorrect source or missing device growth
invalidates the diagnostic. A large finite residual and the existing tanh-slope
alert are explanatory repair signals, never an invented HMC admission cutoff.
Any completed subset remains recorded if time expires; missing maps are not
passed. No map can be called adequately trained for stable estimation without
fresh HMC tuning and posterior checks. The existing binary-tail-ESS integration
issue remains separate.

## Numerical choices and scope

- Rows=32: the existing `reliability_rows` setting, used here as a bounded
  reconnaissance bank. It is too small for tail guarantees or statistical
  ranking. No optimization or repeated seed search.
- Seed: the existing `scoped_seed(config, "post-training-geometry", candidate,
  1.0)` derivation, shared across arms. Record the actual two integers.
- Scale slope alert=0.1: existing tenfold-attenuation explanation, not a tuned
  threshold. Exact-zero residual is a mathematical Gaussian identity; no
  finite numerical residual threshold is promoted.
- Source: `/tmp/BayesFilter-q20-training-repair-20260923-r1` for all target,
  bridge, restore and score operations. Load an immutable copy of the tested
  `neutra_post_training.py` as the new diagnostic wrapper. Do not modify the
  frozen training source or any saved checkpoint. Record hashes of both.
- TensorFlow/TFP float64, batch native, XLA enabled, stable [32,4] signatures.
  Trusted GPU execution, memory growth set before import and verified by the
  repository helper. GPU1 was available at readiness check; recheck at launch.
- One worker, 450-second aggregate cap including setup and compilation. The
  pre-run ledger has 144452.4036210811 campaign seconds and 484.8290616521481
  diagnostic seconds left. Charge measured worker time to both balances using
  the existing Campaign supervisor. At most one localized retry, using only
  the unspent portion of the same 450-second stage cap. A timeout is not a
  failed NeuTra method. Preserve each completed probe before starting the next.

Output root: `docs/plans/artifacts/q20-post-training-geometry-2026-09-23/run-01`.
The adjacent immutable request, probe-module copy and Python runner preserve
exact inputs and command. Execution uses the tfgpu interpreter, with
`TF_FORCE_GPU_ALLOW_GROWTH=true` in the supervised worker environment. The
existing campaign ledger receives a distinct `post-training-geometry` attempt;
the completed training result remains historical and is not overwritten.

## Skeptical review before execution

Wrong-baseline risk is addressed by restoring the common saved parent and
using common latent points. Source drift is addressed by retaining the actual
training source; workspace numerical changes are not substituted. The shared
wrapper passed exact-Gaussian, wrong-scale, nonlinear log-Jacobian, invalid-score
and CPU/XLA tests. GPU compatibility/cost is still unmeasured and belongs to
this bounded attempt. Optimizer state is not loaded for updates. Small-bank
geometry cannot certify convergence, and a saturated layer cannot reject NeuTra
as a method. Compilation may consume the allowance before all maps finish;
partial reports and the external timeout prevent an unbounded attempt. These
checks answer the stated question; the plan passes the skeptical audit.
