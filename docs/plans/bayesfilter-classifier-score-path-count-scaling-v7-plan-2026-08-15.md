# V7 Classifier-Score Path-Count Scaling Campaign

Date: 2026-08-15  
Status: `REVIEWED_READY_FOR_IMPLEMENTATION`

## Research Intent Ledger

| Field | Frozen definition |
|---|---|
| Main question | Does doubling independent training paths per class and perturbation from 8,192 to 16,384 reduce fitted classifier-score bundle variance, and is the reduction compatible with `Var(hat{s}_N) proportional to 1/N`? If 16,384 improves the primary variance endpoint without a validity/accuracy veto, does 32,768 continue that scaling? |
| Candidate mechanism | reduced empirical-training-sample noise from exact nested path prefixes |
| Expected failure | variance plateaus because optimization, validation/calibration noise, model misspecification, or irreducible learned-estimator instability dominates simulation noise |
| Primary endpoint | joint bundle variance averaged over the same 128 audit paths and nine `(T,j)` cells, scaled by the frozen 8,192 baseline path variance |
| Secondary endpoint | variance at the actual fixed observation path; important but lower-powered with ten bundle replicates |
| Accuracy veto | on Gaussian, the 95% lower bound of either audit or fixed exact-MSE ratio exceeds 1 |
| Continuation veto | invalid nesting, mismatched old baseline, data leakage, non-finite fit, optimizer incomplete, invalid artifact, GPU/XLA/memory failure, or exhausted compute budget |
| Repair trigger | localized harness, serialization, import, or resource failure under the unchanged scientific design |
| Nonclaims | no exact SIR score, no filter validation, no claim that optimization work is held fixed, no HMC/default/readiness claim |

## Frozen Ladder And Pairing

The path count `N` means observation paths **per class and per perturbation**.
With six perturbations and two classes, total training rows per parameter
coordinate are `12N`:

| Stage | N per class/delta | Total paths per coordinate |
|---|---:|---:|
| Existing paired baseline | 8,192 | 98,304 |
| Stage 1 | 16,384 | 196,608 |
| Conditional stage 2 | 32,768 | 393,216 |

Only the independent-noise arm is used. V6 showed that SIR CRN effects are
path-heterogeneous; mixing CRN into this ladder would confound the path-count
question. CRN remains a separate completed V6 result.

For each of ten bundle IDs, the 8,192 training noise is the exact bitwise
prefix of 16,384, and 16,384 is the exact bitwise prefix of 32,768. Minus and
plus noises remain independent. The same bundle-specific validation,
calibration, test, fixed, and 128 audit paths are used at every count. The old
V6 `independent_n8192` outputs are reused only after source hashes, seeds,
evaluation hashes, and prefix hashes pass.

## Scaling Estimands

For each count `N`, cell `c`, audit path `m`, and bundle `b`, let
`hat{s}_{N,b,c}(Y_m)` be the fitted score. Define

`V_N,c = mean_m Var_b(hat{s}_{N,b,c}(Y_m))`.

The primary joint statistic uses the frozen V6 8,192 between-path scale:

`J_N = mean_c V_N,c / max(path_scale_c^2, numerical_floor)`.

For adjacent counts `(N,2N)`, report

- variance ratio `R_N = J_2N/J_N`;
- normalized efficiency `Q_N = R_N / 0.5`; and
- scaling exponent `alpha_N = -log(R_N)/log(2)`.

Under exact `1/N` variance scaling, `R_N=0.5`, `Q_N=1`, and `alpha_N=1`.
Use one paired 5,000-replicate cluster bootstrap that resamples bundle IDs and
audit-path IDs while preserving count pairing and all cells. The fixed-path
version resamples bundle IDs only.

Classification is direct:

- `faster_than_1_over_n` if the ratio interval lies entirely below 0.5;
- `compatible_with_1_over_n` if the ratio interval contains 0.5;
- `slower_than_1_over_n` if the ratio interval lies entirely above 0.5; and
- `no_supported_variance_reduction` if the ratio interval also includes or
  exceeds 1 in the direction needed to establish a reduction.

Compatibility means failure to reject the `1/N` value, not proof of it. If
32,768 runs, a single log-linear slope across all three counts is the final
primary scaling summary; adjacent ratios remain explanatory diagnostics.

## Sequential Continuation Rule

Stage 2 at 32,768 is authorized only if the completed 16,384 stage satisfies
all of the following predeclared gates:

1. all Gaussian and SIR bundles are hard-valid and exactly paired to 8,192;
2. the SIR audit variance-ratio 95% upper bound is below 1;
3. neither Gaussian audit nor fixed exact-MSE ratio has a 95% lower bound above
   1; and
4. neither model has statistically supported fixed-path variance worsening,
   defined as a fixed variance-ratio 95% lower bound above 1.

The fixed-path ratio need not have an upper bound below 1 for continuation;
the prior ten-bundle experiment showed that this single-path endpoint is much
less precise. It remains authoritative for any fixed-path claim and cannot be
silently replaced by the audit endpoint.

## Frozen Training Protocol And Default Audit

| Choice | Provenance | Justification | Failure mode / early diagnostic | Status |
|---|---|---|---|---|
| independent noise | V6 SIR result | isolates sample count from heterogeneous CRN effects | variance differs because pairing broke; prefix/hash audit | reviewed choice |
| ten paired bundles | V6 design | supports paired bundle variance intervals at bounded compute | fixed-path interval may remain wide | baseline |
| counts 16,384 and 32,768 | user request | exact doubling tests adjacent `1/N` prediction | insufficient device memory; maximum-count capacity probe | hypothesis |
| V5 architecture/L2 per cell | completed selection artifacts | preserves comparability to 8,192 | controls may be under-tuned at larger N | frozen causal-ablation control, not optimal default |
| batch 2,048; max 80 epochs; minimum 15; patience 10 | V6 protocol | aims to fit each empirical-risk problem to early-stopped completion | larger N receives more updates per epoch; record epochs/steps and nonclaim | inherited protocol |
| shared 512/512/1024 validation/calibration/test paths | V6 | exact paired comparison | held-out finite-sample noise may set a variance floor | frozen baseline |
| 128 audit paths plus one fixed path | V6 | separates typical-path and actual-path behavior | fixed endpoint underpowered | reviewed endpoint split |
| TensorFlow/XLA FP32, TF32 off | V6 artifact | exact execution continuity and GPU default | capacity or compilation failure; trusted GPU probe | reviewed execution default |

This is a fitted-estimator scaling test. It is **not** a fixed-number-of-update
experiment: total minibatch updates rise with N. Optimizer completion is a hard
gate, and no result may attribute all scaling exclusively to raw Monte Carlo
sampling if optimization behavior also changes.

## Evidence Contract

| Evidence | Role |
|---|---|
| source/seed/evaluation-hash and nested-prefix audits | hard validity veto |
| finite values, positive temperature, optimizer completion | hard execution veto |
| Gaussian exact audit/fixed MSE ratios | accuracy veto |
| joint audit variance ratio | primary promotion/continuation criterion |
| fixed-path variance ratio | secondary criterion and worsening veto |
| per-cell ratios, AUC/ECE, epochs, score means | explanatory diagnostics |
| global/adjacent scaling exponent | `1/N` assessment |

Even a successful ladder will not establish an exact SIR score, filter
correctness, universal architecture optimality, or default readiness.

## Skeptical Plan Audit And Pre-Mortem

| Risk checked | Disposition |
|---|---|
| Wrong baseline | reuse only V6 `independent_n8192`, not the CRN combined arm |
| Stale source/program | current estimator and V6 runner hashes exactly match completed bundle manifests; new runner binds its own dependency closure |
| Prefix assumption | TensorFlow stateless SIR noise at 8,192 was verified bitwise equal to the first 8,192 rows of a 32,768 draw for all three noise tensors |
| Proxy promoted to claim | audit variance is the declared typical-path target; fixed-path results remain separate |
| More data changes optimizer work | explicit nonclaim and optimizer-completion hard gate |
| `1/N` judged by an arbitrary margin | compare paired bootstrap interval directly with the theoretical ratio 0.5 and report exponent intervals |
| Conditional 32,768 causes selection bias | continuation rule is frozen before 16,384 results; final claims state the sequential design |
| Gaussian variance falls through bias | exact MSE veto |
| SIR variance falls around a shifted answer | score-mean shifts reported; no correctness claim without an oracle |
| Capacity failure | 16,384 capacity probe before stage 1; 32,768 probe only after continuation gate |
| Long-run disconnect | persistent session, per-bundle result/manifest, fresh versioned roots, resumable campaign |

Audit verdict: the plan answers the user's path-count and `1/N` questions
without mixing CRN or natural path variation into the estimand. Execution may
begin after focused implementation tests and the 16,384 capacity probe pass.

## Compute Budget And Stop Conditions

Total authorized budget: **7.5 GPU hours**, including capacity probes and at
most one localized retry per stage. Planned allocation:

- implementation/CPU checks: no claim-bearing GPU budget;
- 16,384 capacity plus Gaussian/SIR: at most 3 GPU hours;
- conditional 32,768 capacity plus Gaussian/SIR: at most 4.5 GPU hours.

Use fresh roots under
`docs/benchmarks/artifacts/classifier_score_path_count_scaling_20260815/`.
Stop before a full stage if the measured capacity projection exceeds the
remaining budget. Stop on a hard validity veto, corrupted baseline, invalid
artifact, or exhausted budget. A candidate variance result that simply fails
to improve is a scientific result and triggers the frozen sequential stop; it
is not an infrastructure retry.

