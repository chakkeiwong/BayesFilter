# German-credit weighted NeuTra regression plan (2026-08-13)

Status: `GERMAN_WEIGHTED_STOPPED_REVERSE_HMC_NEGATIVE_EVIDENCE`

## Preflight implementation status

The immutable source copies are now stored under
`docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/
german-credit-source/` with the source SHA-256 values recorded below. The
TensorFlow target/score adapter is
`bayesfilter/inference/neutra_german_credit_target.py`; its focused source,
preprocessing, closed-form, autodiff, batch-shape, and XLA-capability tests
pass. The next action is the GPU/XLA batch-native target canary
`docs/benchmarks/probe_weighted_neutra_german_credit_target_2026_08_13.py`.

The first target-canary attempt reached a confirmed XLA-compiled finite target
evaluation but failed before writing its artifact because the generic JSON
normalizer did not handle `tf.TensorShape`. This is an infrastructure/harness
failure, not scientific target evidence. Shape normalization was added and the
single unchanged-contract infrastructure retry authorized below is used.

The unchanged-contract retry passed in `2.48 s`: batch size 64, parameter
dimension 51, float64, XLA compiled, GPU 0 with verified memory growth, all
value/score/constrained outputs finite. This is an engineering target-route
pass only; it is not HMC or posterior evidence.

## Research intent ledger

| Item | Contract |
|---|---|
| Main question | Does target-weighted forward-KL NeuTra remain viable on the 51-dimensional German-credit `gamma_scales2` posterior where the local plain reverse-KL route previously passed three seeds? |
| Mechanism under test | Train a fresh matched reverse-KL dense IAF on the exact German target, freeze it, use a defensive mixture of base-normal scales pushed through that map as a full-support importance proposal, train a separate weighted forward-KL IAF on disjoint replay partitions, then freeze and retune fixed-length HMC for each candidate. |
| Expected failure mode | The reverse proposal may have inadequate tail coverage; weighted rows may have too little effective sample size; the inherited 51-wide three-stage capacity may underfit; or fixed-length HMC may expose residual geometry despite finite training loss. |
| Promotion criterion | The weighted candidate passes canonical sequential HMC numerical/status/movement, modern R-hat `<=1.01`, and bulk/tail ESS `>=400`; constrained first/second moments and their chain-aware MCSE relative to the committed Stan reference are reported as required posterior diagnostics. |
| Promotion veto | Nonfinite target/proposal/replay/checkpoint/HMC tensors, broken source parity, invalid target/reference binding, failed movement/status/archive checks, or failed canonical sequential HMC R-hat/ESS. A gross reference mismatch that cannot be explained by retained-draw MCSE is also a veto; no arbitrary 102-coordinate joint test is introduced. |
| Continuation veto | Corrupt or mismatched data/reference, wrong target math, loss of batch-native TensorFlow execution, GPU/XLA/memory-growth launch invalidity, or exhausted 120-minute German campaign cap. Candidate/proposal failure alone is a repair trigger. |
| Repair trigger | Proposal ESS failure triggers one defensive-scale/probability repair. HMC rejection after the historical-capacity rung triggers one target-specific capacity/budget repair if time remains. Repeated failure stops German and preserves negative evidence. |
| Explanatory diagnostics | Reverse/weighted validation loss, clipping, proposal ESS/max weight, acceptance, finite energy tails, individual moment errors/z-scores, runtime, and rejected tuning arms. |
| Must not be concluded | No weighted-versus-reverse superiority without paired uncertainty, no default promotion, no original-paper replication claim, and no broad NeuTra guarantee. |

## Exact baseline and source provenance

- Data source:
  `/home/ubuntu/python/dsge_hmc/tests/data/neutra/german.data-numeric`,
  SHA-256 `2752b044394958ab6dd193a0b56ca0f0b3a2d8bc7cb8c008e35a5e84bbec02f8`.
- Reference source:
  `/home/ubuntu/python/dsge_hmc/tests/data/neutra/logistic_gamma_gate1_reference.json`,
  SHA-256 `605fbca76b076bb23cf865f7210ef8e6da2b29c1c87964d13463126e71faeb09`.
- Local target source anchor:
  `/home/ubuntu/python/dsge_hmc/src/dsge_hmc/benchmarks/neutra_german.py`.
  It defines `u=[z, log_local_scale, log_global_scale]`,
  `beta=z*local_scale*global_scale`, Bernoulli-logit likelihood, standard-normal
  `z`, Gamma(0.5,0.5) positive scales, and the log-coordinate Jacobians.
- Preprocessing source anchor:
  `/home/ubuntu/python/dsge_hmc/src/dsge_hmc/stan_reference/neutra_logistic.py`.
  It divides each raw predictor by its range without subtracting the minimum,
  maps by `2*x-1`, adds an intercept, and maps labels `1/2` to `0/1`.
- Historical comparator:
  three-stage `(51,51)` dense IAF, 1,000 reverse-KL updates, batch 1,024,
  learning rate `1e-3`; fixed `epsilon=0.05`, `L=32` produced seed-level
  constrained second-moment R-hat `1.00840--1.01047` and minimum ESS
  `794--877`. These settings are warm-start hypotheses, not current defaults.

The BayesFilter implementation must be TensorFlow/TFP and batch-native. Python
standard-library parsing may load immutable text/JSON at configuration time;
all target values, scores, training, proposal density calculations, and HMC
must use TensorFlow. NumPy is not permitted in the candidate path.

## Mathematical target

For `d=25`, unconstrained state

```text
u = (z, a, b),
lambda = exp(a), tau = exp(b), beta = z * lambda * tau.
```

With design `X` and binary response `y`, the unnormalized log density is

```text
log pi(u) = sum_n [y_n log sigmoid(X_n beta)
                   + (1-y_n) log sigmoid(-X_n beta)]
            - 0.5 sum_j z_j^2
            + sum_j [0.5 a_j - 0.5 exp(a_j)]
            + 0.5 b - 0.5 exp(b).
```

Let `g_beta = (y-sigmoid(X beta))^T X`. The exact score is

```text
grad_z = g_beta * lambda * tau - z,
grad_a = g_beta * beta + 0.5 - 0.5 lambda,
grad_b = sum_j g_beta_j beta_j + 0.5 - 0.5 tau.
```

The implementation will be checked against TensorFlow autodiff and fixed
source-probe values before proposal or training evidence is used.

## Default and assumption audit

| Choice | Provenance/status | Justification | Failure mode | Early diagnostic |
|---|---|---|---|---|
| Reverse architecture `(51,51)`, three stages | Historical warm start | Exact architecture of the prior successful local route | Historical reverse capacity may not generate a useful weighted proposal | 200-update canary followed by the fixed 1,000-update rung and heldout reverse loss |
| Reverse batch 1,024, LR `1e-3` | Historical warm start | Previously finite and successful on this exact target | New transport implementation may optimize differently | clipping/loss trace and proposal ESS screen |
| Proposal base scales `(1.0,1.25,1.5)` with probabilities `(0.90,0.08,0.02)` | Convenience hypothesis | Keeps most mass near the fitted reverse map while preserving heavier full-support tails | High-dimensional tail arms can destroy importance ESS or still miss relevant target tails | 65,536-row CPU proposal diagnostic; one repair allowed |
| Proposal screen ESS fraction `>=0.0625` | Derived operational threshold | At weighted batch 4,096 this corresponds to at least 256 effective rows under representative weighting | Global ESS may overstate per-batch stability | report per-batch ESS distribution and max weight; failure only triggers repair |
| Weighted architecture/budget initially `(51,51)`, three stages, 1,000 updates | Target-specific historical-capacity hypothesis | Smallest matched capacity/budget that tests whether weighting preserves the prior success | Weighted objective may require greater capacity or optimization time | 200-update canary, disjoint selection/audit, downstream HMC |
| Weighted batch 4,096, LR `1e-3` | Cross-target warm start only | Existing weighted trainer is stable at this batch on prior controls | 51-dimensional inverse autoregression may clip or underfit | canary wall time, clipping, NLL curve; one reviewed repair only |
| HMC grid `(3,5,10,15,20,25,32)` | Existing fixed-grid policy plus historical `L=32` | Retunes modern shorter trajectories while including the exact historical length | Grid may miss a stable trajectory | tuning diagnostics; `L=1` remains forbidden |
| Four chains and canonical sequential gates | Repository policy `bayesfilter_neutra_sequential_hmc_v1` | Current common downstream viability screen | One bank/seed can be favorable | explicit one-seed nonclaim and later replication need |
| Stan point moments | Committed reference authority with unknown MCSE | Strongest available posterior comparator | Treating point moments as exact or inventing a joint threshold would over-reject | report chain-aware candidate MCSE, per-coordinate deltas/z-scores, maxima/quantiles, and historical error context |

## Staged execution

1. Copy the immutable data/reference into a versioned BayesFilter source root
   and record hashes plus source paths.
2. Implement the TensorFlow batch-native German target and exact score; test
   preprocessing, shapes, finiteness, autodiff parity, fixed source-probe
   parity, and constrained-coordinate conversion.
3. Run a GPU/XLA 200-update matched reverse canary. If finite, run the fixed
   1,000-update historical-capacity rung with disjoint validation seeds and
   freeze the minimum heldout reverse-KL checkpoint.
4. In a separate CPU-only process, diagnose the frozen defensive pushed-forward
   proposal on 65,536 rows. If the derived ESS screen fails, use the one allowed
   scale/probability repair; do not alter the target or HMC criteria.
5. Generate disjoint CPU replay partitions with unique stateless seeds and
   hash-bound receipts. Start with 524,288 training, 65,536 selection, and
   65,536 audit rows; these counts are bounded convenience choices and may be
   reduced only for an explicitly non-claiming canary.
6. Run a GPU/XLA 200-update weighted canary, then the 1,000-update matched
   historical-capacity rung if finite. Freeze by minimum disjoint selection NLL
   and inspect untouched audit NLL/ESS/clipping.
7. Retune and run canonical sequential fixed-length HMC separately for the
   reverse and weighted frozen transports. Apply the same initial bank,
   precision, HMC grid, controller, and reference diagnostics.
8. If weighted HMC fails but the harness remains valid, use at most one
   target-specific capacity/budget repair within the remaining German cap.
9. Write a decision and inference-status result separating hard vetoes,
   viable candidates, descriptive differences, ranking status, and nonclaims.

## Compute and attempt budget

- German wall cap after preflight: 120 minutes, inherited from the master plan.
- At most one proposal repair, one infrastructure retry with unchanged
  scientific contract, and one weighted capacity/budget repair.
- Serious NeuTra training and HMC use one trusted GPU process with
  `TF_FORCE_GPU_ALLOW_GROWTH=true`, verified memory growth before
  initialization, float64, TF32 disabled, and XLA enabled.
- External proposal/replay generation is CPU-only with
  `CUDA_VISIBLE_DEVICES=-1`; target calls remain batch-native tensor programs.
- Every output root is fresh and versioned; failed artifacts are preserved.

## Skeptical plan audit

| Risk | Audit finding |
|---|---|
| Wrong baseline | The fresh reverse arm uses the identical transport class and is both comparator and proposal generator. The historical result is context, not pass authority. |
| Proxy promoted | Training loss, clipping, proposal ESS, acceptance, and moment point errors do not replace corrected HMC. |
| Missing stop conditions | Target/data/reference corruption and execution invalidity stop the campaign; candidate failures trigger only the declared bounded repairs. |
| Unfair comparison | Reverse and weighted arms share architecture, initialization family, precision, HMC controller, initial bank, and diagnostics. Objective-specific row laws are explicit. |
| Hidden assumptions | Architecture, budgets, proposal mixture, ESS floor, grid, reference limitations, and seeds are classified above. |
| Stale context | The historical local route used classical second-moment diagnostics and a fixed `L=32`; the new route must satisfy current rank/folded R-hat and ESS gates. |
| Environment mismatch | Candidate paths are TensorFlow/TFP GPU/XLA; CPU is restricted to external batched proposal/replay generation and diagnostics. |
| Misleading successful command | A run cannot pass merely by completing: hashes, target parity, proposal support, frozen transport identity, HMC hard vetoes, and posterior diagnostics are all preserved. |

Audit verdict: `PASS_FOR_STAGED_GERMAN_EXECUTION`.

### Proposal-screen clarification before execution

For the 65,536-row proposal diagnostic, both the global ESS fraction and the
median of the sixteen consecutive 4,096-row ESS fractions must be at least
`0.0625`. The median is the plan's operational meaning of a representative
training batch; the minimum and lower batch quantiles are reported as
explanatory tail-stability diagnostics rather than added vetoes. Maximum
normalized weight is also explanatory because no independently calibrated
threshold exists for this target. This clarification is recorded before the
proposal run and does not use its outcomes.

### Single proposal repair after `proposal-r1`

The initial pushed radial mixture failed severely: global ESS was
`1.17 / 65536`, median 4,096-row ESS fraction was `0.000839`, and one row held
`0.923` of normalized mass. The dominant row came from the scale-1.5 component
at latent radius `10.29`, which is ordinary for a 51-dimensional scale-1.5
normal; therefore the observed failure is directional/joint mismatch, not
simple radial tail truncation.

The one allowed proposal repair replaces blind radial inflation with a
reference-augmented full-support mixture. It fits independent normal marginals
for constrained `z` and lognormal marginals for positive scales from the
committed Stan first/second moments, maps those scale marginals into log
coordinates exactly, and uses mixture weights `0.85` at marginal scale 1,
`0.10` at marginal scale 1.5, and `0.05` from the frozen reverse-NeuTra
standard-base pushforward. The weights are a convenience choice: most mass is
placed on reference-marginal geometry, one overdispersed component protects its
tails, and a small reverse component preserves learned joint structure. This
repair tests representation when posterior marginal information is available;
it does not test discovery and cannot by itself promote weighted forward-KL.

The repair also failed: global ESS increased to `7.10 / 65536` and maximum
global weight fell to `0.243`, but median 4,096-row ESS fraction was only
`0.000476` versus `0.0625`. All target/proposal values and hashes remained
valid. The proposal repair budget is therefore exhausted and weighted German
training is stopped. The already planned reverse-comparator corrected-HMC run
remains justified as the smallest discriminating check of whether the current
target/harness/transport are valid; it cannot reopen weighted training or turn
reverse success into weighted evidence.

## Terminal decision

The target and TensorFlow/XLA harness passed implementation preflight. The
reverse transport failed the current modern HMC screen, and both full-support
proposal attempts failed the predeclared ESS screen. The German weighted lane
is stopped under the bounded plan. This is target-specific candidate/proposal
negative evidence with one shared dtype defect repaired during execution; it is
not evidence against the weighted forward-KL research direction.

Detailed result: `docs/plans/bayesfilter-weighted-forward-kl-german-credit-result-2026-08-13.md`.
