# Amortized OT Resampler — Architecture Note

This document describes the neural operator that amortizes the entropy-
regularized optimal-transport resampling step from Corenflos et al. (2021)
for use inside differentiable particle filters. The operator takes a
weighted particle cloud `{(x_i, w_i)}` and returns a uniformly-weighted
cloud `{y_j}` whose distribution approximates Corenflos's barycentric
projection at fixed regularization strength ε.

## Motivation

Corenflos et al. (2021), *Differentiable Particle Filtering via Entropy-
Regularized Optimal Transport*, replaces stochastic categorical resampling
with a deterministic transport step computed by Sinkhorn iterations. At
each resampling step they solve

```
min_T  Σ_ij T_ij ||x_i − x_j||²  +  ε · Σ_ij T_ij log T_ij
s.t.   Σ_j T_ij = w_i,   Σ_i T_ij = 1/N
```

then take the barycentric projection `y_j = N · Σ_i T_ij x_i`. Sinkhorn
iterations solve this in O(N² · n_iters), typically 50-200 iterations for
convergence. Inside an HMC chain that traverses the DPF many times, the
inner Sinkhorn loop dominates wall-clock.

The amortized operator replaces the inner Sinkhorn solve with a single
neural-network forward pass: ~3 ms at N=1000 versus ~300 ms for batched
Sinkhorn on the same hardware.

## Architecture

```
{(x_i, w_i)}
    │
    ▼ Normalize (weighted mean and weighted std)
{x̃_i, w_i}
    │
    ▼ Per-particle embedding: Dense(d+1 → d_model) → GELU → Dense(d_model)
{e_i ∈ R^{d_model}}
    │
    ▼ Set Transformer encoder (equivariant, no pooling):
    │   ISAB layer 1 → ISAB layer 2
    │   each ISAB: m=16 inducing points, 4 attention heads
{f_i ∈ R^{d_model}}
    │
    ▼ Concatenate log ε
{[f_i ; log ε] ∈ R^{d_model + 1}}
    │
    ▼ Coupling head:
    │   q_i = W_q · [f_i; log ε]
    │   k_i = W_k · [f_i; log ε]
    │   L_ij = q_i · k_j / sqrt(d_head)
    │   π_ij = (1/N) · softmax_i(L_ij)        (column-softmax)
    │
    ▼ Aggregate (barycentric projection)
ỹ_j = N · Σ_i π_ij · x̃_i
    │
    ▼ Denormalize
{y_j}
```

**Why column-softmax**: makes the target marginal `Σ_i π_ij = 1/N` exact
by construction. Source marginal `Σ_j π_ij ≈ w_i` is satisfied
approximately (~3×10⁻³ residual on test clouds).

**ISAB blocks** (Lee et al., 2019, *Set Transformer*) reduce attention
cost from O(N²) to O(N · m) where m=16 is the number of inducing points.
This is necessary to avoid making the operator's encoder itself
quadratic in N, which would defeat the purpose.

**Architecture config**: d_model=64, num_heads=4, d_head=64, num_inducing=16,
num_isab_layers=2. Total trainable parameters: 115,648.

## Training procedure

Targets are precomputed offline. For each cloud, run Sinkhorn iterations
to convergence, save the barycentric projection in normalized space.

- **Cloud distribution (Source B)**: mixture of Gaussians with 1–3
  components, Dirichlet weights with α drawn over log10 ∈ [-1, 1].
  N=1000 particles per cloud, state dim d=2.
- **ε grid**: log10(ε) ∈ {-0.5, -0.3, -0.12} → ε ∈ {0.32, 0.50, 0.76}.
- **Splits**: 20K train, 1K val, 1K test.
- **Optimizer**: AdamW, base lr 3×10⁻⁴, batch size 256, 50 epochs,
  cosine-with-warmup schedule.
- **Loss**: MSE in normalized space against precomputed targets.

Best checkpoint by validation MSE; early stopping with patience 10
epochs. Training takes ~1 hour on a single A100.

## Test-set evaluation

On 1,000 held-out test clouds, all three ε values:

| Metric                          | Value          |
|---------------------------------|---------------:|
| MSE (normalized space)          | 2.13×10⁻³      |
| Per-particle L₂                 | 0.041          |
| Predicted transport cost        | 0.0601         |
| Sinkhorn target transport cost  | 0.0611         |
| Sinkhorn divergence (set-level) | 6.6×10⁻³       |

Predicted transport cost matches the Sinkhorn target within 2%, so the
operator captures both *where* mass goes and *how concentrated* the
output cloud becomes.

## Speedup vs direct Sinkhorn

Wall-clock per resampling call, N=1000, d=2, A100:

| Batch | Operator (ms) | Sinkhorn (ms) | Speedup    |
|------:|--------------:|--------------:|-----------:|
| 1     | 2.93          | 330           | 113×       |
| 8     | 2.82          | 870           | 309×       |
| 32    | 3.78          | 2,101         | 555×       |
| 128   | 14.13         | 6,960         | 493×       |

These are forward-only timings. Inside HMC, backward-pass costs also
matter and are comparable between the two methods after both adopt
implicit differentiation (which our `tf_utils/sinkhorn.py` does, following
Corenflos et al., 2021).

## Pseudo-marginal HMC integration

For HMC to mix on a stochastic log-evidence target, the random draws
inside the DPF must be deterministic for a given θ within one MCMC
iteration. This is implemented via a `seed` argument threaded through
the DPF instead of a stateful `tf.random.Generator`. The custom HMC
runner in `hmc/run_hmc_corenflos_lg.py` advances a master seed once per
HMC iteration and freezes it across leapfrog evaluations.

The amortized operator itself contains no randomness — it is a
deterministic function of (particles, weights, log ε) — so seeding only
needs to cover particle-initial and dynamics noise.

## Boundaries and known limitations

- **State dimension fixed at d=2**. Larger state dims require retraining;
  the architecture is not d-agnostic.
- **ε range**: trained on ε ∈ [0.32, 0.76]. Inputs outside this range
  (e.g., ε=0.25) produce extrapolation that degrades sharply when used
  inside HMC.
- **Approximation bias**: the ~2×10⁻³ MSE residual translates to a small
  posterior-mean shift when the operator replaces Sinkhorn inside HMC.
  Visible at the level of posterior comparisons against Kalman ground truth.

## Code locations

- Operator architecture: `dpf_pretrained/mgn_ot_operator/models/operator_b.py`
  (and other models in the same directory)
- Training pipeline (config, dataset, train loops, evaluation):
  `dpf_pretrained/mgn_ot_operator/{config.py, data/, training/, evaluation/}`
- Pretrained weights:
  `dpf_pretrained/mgn_ot_operator/checkpoints_option_b/best/ckpt-82.*`
- DPF adapter: `tf_utils/amortized_resampler.py`
- DPF integration: `tf_filters/differentiable_particle.py` (resampler='amortized')
- HMC integration: `hmc/run_hmc_corenflos_lg.py`
- Training notebook:
  `notebooks/03_neural_amortized_ot/02_train_coupling_operator.ipynb`
- Brenier-gap diagnostic notebook:
  `notebooks/03_neural_amortized_ot/01_brenier_gap_evidence.ipynb`
- HMC comparison notebook:
  `notebooks/03_neural_amortized_ot/03_hmc_amortized_comparison.ipynb`
