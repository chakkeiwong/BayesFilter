# GenUT Austria-SIR AD Root-Cause Result and Reset Memo

Date: 2026-08-17

Plan:
`docs/plans/bayesfilter-genut-sir-ad-root-cause-localization-plan-2026-08-17.md`

Terminal result:
`docs/benchmarks/artifacts/genut-sir-ad-root-cause-20260817/attempt07_terminal_cpu_reference/result.json`

Supporting GPU checkpoints:

- `docs/benchmarks/artifacts/genut-sir-ad-root-cause-20260817/attempt05/checkpoint.json`
- `docs/benchmarks/artifacts/genut-sir-ad-root-cause-20260817/attempt05/full_program_checkpoint.json`

## Direct verdict

The manual GenUT JVP is correct relative to the checked finite graph program.
The earlier callback and Contract-E reset failures were false alarms caused by
unscaled FP32 finite-difference thresholds and cancellation. Do not repair the
callback, Sinkhorn, reset, or score algebra based on those old failures.

The Austria-SIR `T=20` `j0` concern is not explained by standard deviation
alone. Two distinct effects are supported:

1. between-seed particle/cubature variability grows sharply with horizon; and
2. the higher-moment maps amplify FP32 backend and validity-branch sensitivity.

The exact contribution of each effect at `N=1008,T=20` is not identified. No
exact observed-data SIR score has been established, so finite-program bias also
remains possible.

## Code and mathematical trace

For each observation time the scalar route:

1. propagates particles and their parameter tangents through the SIR RK map;
2. forms `log_weights = log(weights) + log_likelihood`;
3. adds `logsumexp(log_weights)` to the finite likelihood value;
4. computes its tangent as the normalized-weight expectation of the log-weight
   tangent;
5. applies the fixed-iteration Sinkhorn row quotient;
6. applies the Contract-E total mean/covariance reset;
7. applies the diagonal, pairwise, and cap shape maps;
8. restores uniform weights and recurses; and
9. sums the per-time value and score increments.

Thus the reported score is mathematically the total derivative of that finite
program if each map JVP is correct and the validity branch is stable. It is not
automatically the derivative of the exact observed-data likelihood.

The equal-weight `d=18` cubature design consists of replicated points
`+/-sqrt(18)e_j`. Its marginal fourth moment is `18`, whereas a standard
Gaussian's fourth moment is `3`. Large nonlinear higher-moment corrections are
therefore acting on a highly structured base cloud. This is a plausible
variance/backend-sensitivity amplifier, not a proved exact-likelihood bias
formula.

## AD evidence

All errors below are symmetric relative L2 errors between the existing manual
JVP and TensorFlow AD of the same map.

| Checked map | Precision/scope | Maximum relative error | Verdict |
|---|---|---:|---|
| SIR transition callback | FP32, total tangent | `2.38e-8` | correct in checked scope |
| SIR observation callback | FP32, total tangent | `4.57e-8` | correct in checked scope |
| Sinkhorn coupling | FP32, 32 fixed updates | `3.83e-8` | correct in checked scope |
| Sinkhorn particles | FP32, 32 fixed updates | `6.06e-8` | correct in checked scope |
| Contract-E intermediates | FP32, condition proxy through `459` | `3.15e-7` | correct within FP32 roundoff |
| Contract-E intermediates | FP64, same ladder | `3.90e-16` | correct within FP64 roundoff |
| Composed Sinkhorn plus reset | FP32 | `2.46e-7` | correct in checked scope |
| No-shape map | FP32 | `0` | exact agreement |
| Diagonal shape map | FP32 | `2.72e-4` | small graph-order/roundoff difference |
| Pairwise shape map | FP32 | `2.68e-4` | small graph-order/roundoff difference |
| Dual-cap shape map | FP32 | `2.69e-4` | small graph-order/roundoff difference |

The direct shape-map differences are not material in the checked full graph:
at `N=36,T=2`, diagonal manual versus AD `j0` differs by `2.71e-6`
symmetrically.

## Graph prefix ladder

CUDA was deliberately hidden for this reference-only terminal ladder after the
GPU permission reviewer timed out four times before process creation. These
rows check mathematical derivative parity only; they are not GPU/XLA evidence.

| N | T | Arm | Manual `j0` | AD `j0` | Relative error | Valid |
|---:|---:|---|---:|---:|---:|---|
| 36 | 1 | none | `-5.399624` | `-5.399626` | `1.77e-7` | yes |
| 36 | 2 | none | `-7.383509` | `-7.383514` | `3.87e-7` | yes |
| 36 | 3 | none | `44.075676` | `44.075790` | `1.30e-6` | yes |
| 36 | 4 | none | `100.817825` | `100.813950` | `1.92e-5` | yes |
| 36 | 5 | none | `-56.990128` | `-56.990742` | `5.39e-6` | yes |
| 36 | 1 | diagonal | `-5.399624` | `-5.399626` | `1.77e-7` | yes |
| 36 | 2 | diagonal | `-7.575515` | `-7.575556` | `2.71e-6` | yes |

These rows falsify a material manual-JVP algebra defect at the checked onset
scales.

## Backend sensitivity

The supporting RTX 4080 SUPER run compared the XLA scalar/manual route with the
graph route on identical inputs.

| N | T | Arm | XLA value vs graph value | XLA manual `j0` | Graph AD `j0` | Interpretation |
|---:|---:|---|---:|---:|---:|---|
| 36 | 2 | none | `5.90e-7` relative | `-7.380935` | `-7.383562` | small backend drift |
| 36 | 2 | diagonal | `8.91e-4` relative | `-8.304531` | `-7.570648` | shape correction amplifies drift |
| 36 | 2 | pairwise | `1.05e-3` relative | `-8.275208` | `-7.475715` | shape correction amplifies drift |
| 36 | 2 | dual cap | `1.06e-3` relative | `-7.992507` | `-7.061172` | cap does not remove backend drift |
| 36 | 5 | none | `1.84e-4` relative | `-59.713776` | `-56.990875` | drift accumulates with recursion |
| 36 | 5 | diagonal | XLA nonfinite | nonfinite | `33.140575` | backend changed the validity branch |

The last row had an XLA minimum covariance-gap eigenvalue of `-0.00849`, while
the graph route remained finite. This is a hard small-rung numerical validity
failure for that XLA configuration. It is not evidence that all `N=1008,T=20`
rows are invalid; those rows reported finite validity diagnostics.

## Variance evidence

The three-seed `N=1008,T=20` ablation remains descriptive:

| Arm | Mean `j0` | Sample SD |
|---|---:|---:|
| No higher moments | `-168.82` | `75.75` |
| Diagonal correction | `38.16` | `104.92` |
| Pairwise correction | `12.96` | `60.41` |
| Dual cap | `89.03` | `58.97` |

For the diagonal arm, sample SD grows from `0.31` at `T=2`, to `17.26` at
`T=5`, to `104.92` at `T=20`. No single time step dominated; the maximum
absolute-increment share was approximately `0.14` to `0.22`. This supports
accumulated recursive sensitivity rather than one isolated observation.

Three seeds do not support a ranking or a stable variance estimate. The SD
evidence is real but descriptive.

## Decision table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Manual JVP correctness | graph manual/AD errors at most `1.96e-5` in the prefix ladder | no local-map or graph-rung veto | full `N=1008,T=20` AD is not available | do not repair JVP algebra | exact SIR score |
| Old FD failures | contradicted by AD and FP64 reset ladder | old absolute thresholds rejected | none for the checked maps | retire those failures as false alarms | all possible derivative defects |
| FP32 XLA stability | shape-amplified value/score drift and one small-rung branch failure | hard veto for `N=36,T=5` diagonal XLA row | magnitude at `N=1008,T=20` | run a reviewed FP64/no-TF32/conditioning ladder | production invalidity for every scope |
| T20 seed variance | large descriptive SD and horizon growth | no statistical ranking | only three seeds | many-seed paired variance decomposition | repaired permutation or dual cap is best |
| Exact observed-data score | no admitted oracle | classifier and LGSSM oracle gates remain failed | finite-program bias | develop/validate an independent oracle | correctness of the SIR score value |

## Inference status

| Item | Status |
|---|---|
| Hard veto screen | local maps and graph prefix pass; `N=36,T=5` diagonal XLA validity branch fails |
| Statistically supported ranking | none |
| Descriptive-only differences | T20 arm means/SD, horizon SD growth, and backend drift magnitudes |
| Default/HMC readiness | not supported |
| Next evidence needed | FP64/no-TF32 XLA sensitivity ladder; paired many-seed variance decomposition; then an admitted independent observed-data score reference |

## Attempt ledger

- `attempt01`: XLA forward AD required explicit loop bounds; no result.
- `attempt02`: loop bounds exposed an XLA dynamic-transpose limitation; no result.
- `attempt03`: reporting-key typo after callback/transport/reset stages; no terminal result.
- `attempt04`: value-only forward AD hit a TensorFlow `tf.cond` assertion.
- `attempt05`: checkpointed all local GPU results and six small full-program
  rows; stopped when the progress formatter encountered the deliberately
  preserved nonfinite diagonal row.
- `attempt06_terminal`: redundant parallel Jacobian vectorization failed before
  the first prefix row.
- `attempt07_terminal`: GPU process was not created after four permission-review
  timeouts across two approved launch forms.
- `attempt07_terminal_cpu_reference`: terminal CPU-hidden graph derivative
  artifact used above; wall time `97.6 s`.

## Post-run red team

The strongest alternative explanation is that the graph program is internally
differentiated correctly while the XLA FP32 program at `N=1008,T=20` is both a
numerically different finite program and a biased approximation to the true
likelihood. This result does not separate backend drift, cubature bias, and
particle variance at that full scope.

Evidence that would overturn the manual-JVP verdict is a reproducible,
scale-material graph manual/AD disagreement on a valid rung, or a local-map
disagreement that persists in FP64. Evidence that would overturn the backend
sensitivity concern is a controlled FP64/no-TF32/conditioning ladder showing
the same branch and score within declared numerical error across backends.

## Clean restart state

Resume from this memo, the terminal CPU-reference result, and the two
`attempt05` GPU checkpoints. Do not rerun the old finite-difference thresholds.
Do not interpret `attempt01` through `attempt04` as scientific evidence.

The next narrow task is a reviewed backend/precision ladder on the smallest
scope that reproduces the drift: `N=36`, `T=2` and `T=5`, no-shape and diagonal,
comparing FP32 XLA/TF32, FP32 XLA without TF32, FP32 graph, and FP64 graph/XLA
where supported. Only after that should a paired many-seed `N=1008,T=20`
variance study be launched.

## Nonclaims

No exact SIR score, unbiasedness, algorithm superiority, repaired-permutation
promotion, default readiness, posterior correctness, NeuTra readiness, or HMC
readiness is established.
