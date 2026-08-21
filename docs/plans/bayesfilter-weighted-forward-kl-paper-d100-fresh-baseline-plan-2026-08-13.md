# Weighted forward-KL paper d100 fresh-baseline plan (2026-08-13)

Status: `TERMINAL_EXECUTED_2026-08-13`

This is the target-specific continuation of
`bayesfilter-weighted-forward-kl-positive-control-regression-plan-2026-08-12.md`.
It covers only the dimension-100 Gamma-spectrum Gaussian and paper funnel. The
May `dsge_hmc` runs are historical context, not current pass evidence: they used
CPU/non-XLA sampling, an older fixed burn-in/draw protocol, and a grid containing
the now-forbidden `L=1` arm.

## Research intent ledger

| Item | Contract |
|---|---|
| Main question | On two exactly sampleable dimension-100 paper targets, can a fresh matched reverse-KL NeuTra baseline and the new exact-replay forward-KL training route each produce a frozen transport that passes current fixed-length sequential HMC and analytic posterior diagnostics? |
| Mechanism under test | Train the identical dense-IAF family by (a) reverse KL using fresh standard-normal latent rows or (b) forward KL using independent exact target draws with uniform weights; freeze by disjoint selection loss; retune fixed-length HMC separately; assess retained physical draws against the exact target law. |
| Exact baseline | For each target, fresh reverse KL with the same target, target-specific selected architecture/optimizer budget, transport implementation, precision, GPU/XLA class, HMC controller, initial-state law, and analytic diagnostics. Historical results are context only. |
| Expected failure modes | The historical architecture or optimizer may not transfer to the current trainer; the funnel may retain neck/tail curvature after training; a finite heldout loss may hide missing funnel tails; the HMC grid may miss a stable trajectory; or high-dimensional marginal multiplicity may cause a misleading after-the-fact verdict. |
| Primary promotion criterion | An arm independently passes canonical sequential NeuTra HMC numerical/status/movement gates, maximum modern retained R-hat `<=1.01`, bulk/tail ESS `>=400`, and its predeclared target-specific analytic diagnostics. Passing means viable on this target, not better than the other arm. |
| Promotion veto | Nonfinite target/score/replay/transport/HMC tensor; source-identity or derivative-parity failure; invalid artifact/hash; failed movement/status/archive; failed retained R-hat/ESS; or a predeclared analytic diagnostic whose uncertainty interval excludes the exact target value. Native divergence is a veto only if the runtime exposes it; unavailable is recorded as unavailable, never zero. |
| Continuation veto | Wrong target formula or frozen Gaussian constants, broken exact sampler, batch-native/XLA/memory-growth invalidity, artifact corruption, or exhaustion of the four-hour d100 execution cap. Failure of one trained candidate or one target is a repair trigger and does not cancel the other target. |
| Repair trigger | Failed training canary triggers the bounded learning-rate screen; improving heldout loss or failed HMC at the historical budget triggers at most one 10,000-update/capacity repair; a valid HMC grid failure triggers one smaller-step retuning repair. |
| Explanatory only | Training/selection/audit NLL, clipping, acceptance, finite energy-error tails, runtime, per-coordinate moment errors, extreme sample values, and observed reverse-versus-forward differences without paired uncertainty. |
| Must not be concluded | No objective superiority, equal-cost efficiency ranking, original-paper replication, universal funnel solution, default promotion, or validity beyond these exact targets and tested seeds. |

Before stopping this plan after a failed arm, the result must say whether the
target, exact sampler, implementation, artifact, or execution contract was
invalid, or whether only the current candidate failed. A valid candidate failure
does not stop the planned repair or the other target.

## Source-bound mathematical targets

### Paper funnel

For `d=100`, write `theta=(y,x_1,...,x_99)`. The exact generative law is

```text
y ~ Normal(0, 1),
x_i | y ~ Normal(0, exp(2y)), independently for i=1,...,99.
```

Here `exp(2y)` is the conditional variance and `exp(y)` the conditional standard
deviation. Up to an additive constant,

```text
log p(theta) = -0.5 y^2 - 0.5 exp(-2y) sum_i x_i^2 - 99 y.
```

The exact score is

```text
d/dy log p = -y + exp(-2y) sum_i x_i^2 - 99,
d/dx_i log p = -exp(-2y) x_i.
```

The standardized residuals `r_i=x_i exp(-y)` are independent standard normal
and independent of `y`. Exact marginal moments include `E[y]=0`, `E[y^2]=1`,
`E[x_i]=0`, and `E[x_i^2]=exp(2)`.

### Gamma-spectrum ill-conditioned Gaussian

The source target is the exact `new_ill_cond_gaussian` construction used by
`/home/ubuntu/python/dsge_hmc/src/dsge_hmc/benchmarks/neutra_paper_targets.py`:

```text
rng = numpy.random.RandomState(10)
lambda = sort(rng.gamma(shape=0.8, scale=1, size=100))
Q, _ = qr(rng.randn(100,100))
Sigma = symmetrize((Q * lambda**-1) @ Q.T)
theta ~ Normal(0, Sigma).
```

The naming is potentially confusing: the sampled `lambda` values are the
precision eigenvalues, while `lambda**-1` are covariance eigenvalues. The
BayesFilter candidate path will not recreate this NumPy RNG. A diagnostic-only
source exporter will freeze the source-generated `Sigma`, precision, and
Cholesky factor into a versioned JSON artifact. Candidate runtime then uses only
stdlib parsing and TensorFlow constants. The artifact binds source path/hash,
dimension, seed, Gamma shape/scale, matrix hashes, symmetry residual, Cholesky
reconstruction residual, eigenvalue range, and realized condition number.

## Evidence contract

| Role | Evidence |
|---|---|
| Engineering prerequisite | Source-bound target spec loads; values and analytic scores match TensorFlow autodiff; exact sampler shape/moments are finite; batch size exceeds one; GPU canary records one visible GPU, float64, TF32 off, XLA on, verified memory growth, and no row-mapped/scalar target fallback. |
| Training selection | Each objective uses a disjoint exact selection cloud. Checkpoint selection minimizes its own target-appropriate heldout loss. Selection is allowed to choose training settings but is not posterior or HMC evidence. Untouched exact audit data are opened only after freezing. |
| HMC promotion | `bayesfilter_neutra_sequential_hmc_v1`, fixed-length TFP HMC, `L>=2`, four batched chains, archived discarded warm-up, maximum recent-window warm-up R-hat `<=1.05`, retained modern R-hat `<=1.01`, bulk/tail ESS `>=400`, and target analytic diagnostics below. |
| Gaussian analytic authority | Exact `Sigma`; predeclared scalar summaries are whitened grand mean `0`, whitened grand second moment `1`, Mahalanobis-radius mean `100`, and four fixed orthonormal projection means/second moments `0/1`. Each is reported with its own chain-aware 99% interval; there is no omnibus p-value. Per-coordinate mean/covariance errors are descriptive. |
| Funnel analytic authority | Exact standard-normal `y` law and `r_i=x_i exp(-y)` law. Required separate 99% diagnostics are `E[y]=0`, `E[y^2]=1`, probabilities `P(y<-2)=P(y>2)=Phi(-2)`, `E[r]=0`, `E[r^2]=1`, `Cov(y,r^2)=0`, and `E[r^2 | y<-2]=E[r^2 | y>2]=1`. At exact `y` quantiles for probabilities `(0.01,0.10,0.50,0.90,0.99)`, the candidate empirical CDF is tested against the exact probability with a separate chain-aware 99% interval; empirical quantiles are also reported descriptively. There is no joint test. |
| Analytic uncertainty | Means, probabilities, covariance, tail-conditional ratios, and CDF-at-exact-quantile values use chain-aware batch-means MCSE from retained draws. Tail ratios use their ratio-estimator influence series. A diagnostic fails only when its exact value lies outside its declared interval; no post-run tolerance is invented. The independent exact replay calibration partition remains a sampler/reference check, not an interval authority that would incorrectly assume retained HMC draws are independent. |
| Artifact | Fresh roots under `docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/paper-d100/`, containing target source, replay receipts, training state/manifest/result, tuning, every warm-up/retained chunk, analytic diagnostics, result, and artifact hashes. |

The analytic diagnostics are deliberately a small structural set. Testing every
one of 100 means and 10,000 covariance entries as a single pass rule would make
the multiplicity decision dominate the scientific question. Individual
coordinate errors remain visible but are not silently upgraded into a joint
test.

## Default and assumption audit

| Choice | Provenance/status | Justification | Failure mode | Earliest diagnostic |
|---|---|---|---|---|
| Dimension 100 and exact formulas | Direct source-bound target definition; reviewed target choice | These are the paper-scale fresh-baseline targets requested by the master plan | Accidental Neal-funnel variant with conditional variance `exp(y)`, or different Gaussian spectrum | Fixed formula tests, source target probes, score/autodiff checks, immutable Gaussian constants hashes |
| Exact target replay | Derived from exact generative laws; target-specific reviewed choice | Removes the German importance-proposal confound and makes forward-KL rows unweighted posterior draws | Sampler formula or partition reuse could invalidate the objective | Exact sampler formula tests, stateless disjoint seeds, receipt hashes, empirical smoke moments |
| Replay sizes `1,048,576 / 65,536 / 65,536` | Convenience counts scaled from prior weighted controls | 256 training batches at batch 4096 and precise independent selection/audit while using about 0.9 GB per target | Dataset cycling may overfit; storage may be wasted | available disk/RAM check; disjoint selection curve; audit opened after freeze |
| Batch 4096 | Inherited paper-scale and weighted-control warm start | Historical paper suite and current weighted controls both used 4096 | GPU OOM or poor update noise | one compiled update and allocator telemetry; reduce only as an explicit target-specific repair |
| Dense IAF `(100,100)`, three stages, ELU, `s_max=1`, init scale `0.02` | Exact historical paper-suite architecture and local May baseline; warm-start hypothesis | Establishes a fresh comparable baseline before capacity invention | Current transport parameterization may underfit funnel tails or overparameterize Gaussian | 200-update canary, selection/audit, downstream HMC; one `(128,128)`, six-stage repair only if triggered |
| LR candidates `1e-3` constant and `1e-2` with drops after 1,000 and 4,000 completed updates | `1e-3` from current weighted controls; `1e-2` schedule from historical paper suite source steps 1000 and 4000; both warm starts | Tests the two directly relevant optimizer regimes without an arbitrary broad grid | 200 updates cannot reveal later schedule behavior; high LR can clip or destabilize | finite 200-update canary for each objective/target; paired selection-NLL difference and clipping are nomination diagnostics |
| Historical budget 5000 updates | Direct historical paper-suite budget; baseline rung | Answers whether the current implementation reproduces a credible target-specific transport at the prior budget | Loss can still improve at the boundary | paired selection-loss change over the last 1000 updates with MCSE; HMC result |
| Optional 10,000-update repair | Inherited successful weighted-control budget, repair only | Gives one bounded continuation when 5000 is undertrained or HMC exposes residual geometry | Local optimization may continue without solving sampler geometry | require statistically resolved late selection improvement or failed valid HMC before launch |
| HMC grid `L=(3,5,10,15,20,25,32)` | Current fixed-grid policy plus historical `L=32`; reviewed target-specific screen | Retains current grid and historical long trajectory while excluding meaningless `L=1` | Stable region may need smaller epsilon | target-specific tuning; one bounded smaller-step repair |
| Four chains and policy thresholds | Repository policy `bayesfilter_neutra_sequential_hmc_v1` | Current common sampler-readiness screen | One bank/seed can be favorable | record one-seed viability nonclaim; replicate only after an arm passes |
| GPU 1, sequential objectives | Fresh 2026-08-13 capacity probe plus a valid GPU1/XLA compilation canary after the other agent finished; execution choice | One process avoids unmeasured co-residency and keeps the validated device binding | GPU 1 may become busy or unhealthy | trusted device/capacity probe immediately before launch; wait instead of silently moving lanes |
| Float64, TF32 off, XLA on | Repository serious-NeuTra policy | Matches current target and HMC evidence class | compilation or memory failure | focused GPU/XLA target and one-update canaries |

The LR and capacity choices are not promoted defaults. Each target and each
objective receives its own canary, selection, audit, budget decision, and HMC
retuning. If the four-hour cap cannot support that target-specific protocol, the
remaining arm is reported under-budgeted rather than promoted from transferred
settings.

## Staged execution and stop logic

1. Create a diagnostic-only exact source exporter for the Gaussian constants;
   freeze a fresh versioned JSON artifact and record source/exporter hashes.
2. Implement one TensorFlow module for both target specs, batch-native value and
   analytic score, exact CPU replay sampling, and graph-native HMC adapters. No
   NumPy import is allowed in this candidate/runtime module.
3. Add focused tests for source constants, formula values, analytic-score versus
   autodiff, exact-sampler deterministic seeds/shapes/moments, float64, batch
   behavior, XLA compilation, funnel conditional standardization, and rejection
   of wrong dimensions/nonfinite inputs.
4. Implement a CPU-only replay generator with stateless disjoint training,
   selection, audit, calibration, and HMC-initialization seeds. Serialize tensors
   with hashes and record `CUDA_VISIBLE_DEVICES=-1`; no optimizer update occurs.
5. Implement a shared GPU/XLA training runner for matched reverse and exact-replay
   forward KL. Run finite 200-update LR canaries, then the selected 5000-update
   historical-capacity rung. Freeze minimum disjoint selection loss and open the
   untouched audit only after freezing.
6. If the selected 5000-update candidate is finite, retune fixed-length HMC on
   disjoint tuning seeds and run the canonical sequential controller. Training
   loss, clipping, and acceptance cannot substitute for this stage.
7. If the candidate fails HMC while target/harness evidence remains valid, or if
   paired late-checkpoint selection improvement is statistically resolved, use
   at most one 10,000-update/capacity repair and rerun HMC. Preserve all failed
   artifacts.
8. Run Gaussian first as the linear exact-sampler/target sanity target. A valid
   Gaussian candidate failure does not skip funnel; an invalid shared harness
   does pause funnel until repaired. Run funnel with its independent protocol.
9. Write a terminal result and update the campaign reset memo with decision and
   inference-status tables, run manifests, strongest alternative explanations,
   and the next justified model.

## Compute and attempt budget

- Four-hour serious execution cap for the two d100 targets. This is a bounded
  allocation within the user's earlier eight-hour campaign allowance, not an
  estimate that four hours are required.
- At most one unchanged-contract infrastructure retry per rung, one
  optimizer/capacity repair per objective/target, and one HMC smaller-step
  repair per frozen candidate.
- Execute one TensorFlow GPU process at a time on GPU 1 after a trusted capacity
  check. Every launch sets `TF_FORCE_GPU_ALLOW_GROWTH=true` before TensorFlow
  import and verifies growth before any logical GPU or tensor initialization.
- External exact replay generation is CPU-only and batch-native. Serious
  training and HMC are GPU/XLA. Every output root is fresh and versioned.
- If both objectives cannot receive their target-specific protocol within the
  cap, stop as under-budgeted; do not infer equivalence or failure from a missing
  arm.

## Pre-mortem

- The Gaussian can pass even with an unnecessary nonlinear transport because
  its density is easy. This validates mechanics and analytic agreement, not
  funnel capability or objective superiority.
- Funnel heldout NLL can improve while the learned map undersamples extreme
  negative `y`. Tail probabilities, tail-conditional standardized residuals,
  modern R-hat/ESS, and corrected HMC are therefore claim-bearing.
- R-hat can look good when all chains explore only the central funnel. Exact
  `y` tail/quantile diagnostics and conditional residual checks distinguish this.
- A giant per-coordinate test family could reject a correct sampler by
  multiplicity. The small structural diagnostic set is frozen before the run;
  the full coordinate table is descriptive.
- An exact replay file could accidentally be reused across selection/audit or
  between targets. Stateless seed partitions, target identity, shapes, and file
  hashes are bound in every receipt.
- A command can complete while using CPU, eager mode, TF32, scalar target loops,
  or invalid memory policy. Those are launch-invalid and cannot become
  scientific evidence.
- The historical schedule may look favorable only because the first 200 updates
  precede its decay. Final selection and HMC, not canary loss, decide viability.

## Skeptical pre-execution audit

| Audit question | Finding and repair |
|---|---|
| Wrong baseline? | Repaired. Fresh reverse KL uses the same current transport and downstream controller. May CPU/non-XLA results are context only. Exact target laws, not historical metrics, are the reference authority. |
| Proxy promoted? | Repaired. Loss, clipping, canaries, audit NLL, acceptance, and runtime may select or explain training but cannot pass an arm. Corrected sequential HMC plus analytic agreement is primary. |
| Missing stop conditions? | Repaired. Shared target/sampler/device/artifact invalidity pauses execution; candidate failure triggers the bounded repair and does not reject the method or cancel the other valid target. |
| Unfair comparison? | Repaired. Transport family, architecture candidates, precision, hardware class, initialization family, controller, and analytic diagnostics match. Each objective may select its own LR because this is independent viability, not an equal-cost superiority test. |
| Hidden assumptions? | Repaired. Target RNG, architecture, replay counts, batch, optimizer regimes, budget ladder, grid, diagnostics, thresholds, seeds, device, and attempt caps have provenance and early checks. |
| Stale context? | Repaired. Historical `L=1`, fixed 1536/1536 chains, CPU/non-XLA execution, and second-moment-only gates are not reused as current authority. Native divergence unavailability is explicit. |
| Environment mismatch? | Repaired. Candidate target/training/HMC paths are TensorFlow/TFP float64 GPU/XLA with TF32 off and verified growth. The source exporter and exact replay generation are explicit CPU diagnostics/preparation only. |
| Would artifacts answer the question? | Yes. Frozen target identity, exact disjoint replay, objective-specific state/selection/audit, current tuning, all sequential chunks, analytic diagnostics, command/environment/seeds/wall time, and hashes are preserved. A successful training command alone cannot answer the question. |
| Could the plan pass while misleading us? | Remaining limitation is one HMC seed bank per target/arm. A pass is classified only as viability; statistically supported objective ranking and default readiness remain false pending replication. |

Audit verdict: `PASS_FOR_IMPLEMENTATION_AND_STAGED_EXECUTION`. No serious run is
authorized until focused target/export/replay tests and a GPU/XLA one-update
canary pass. A valid Gaussian candidate failure is evidence about that candidate,
not a continuation veto for the planned funnel target.

## Pre-funnel diagnostic correction (2026-08-13)

The pre-funnel implementation audit found two claim-invalidating defects before
any funnel candidate was trained or sampled. The first implementation computed
the declared `Cov(y,r^2)` as the raw cross-moment `E[y r^2]`, and its
tail-conditional MCSE series divided each observation by the global tail count,
giving the wrong ratio-estimator uncertainty. Both are corrected to the sample
covariance and the ratio influence series.

The same audit rejected the planned independent-exact-draw quantile interval as
a promotion gate because it would calibrate iid sampling error while the
candidate archive is serial HMC output. The revised, mathematically equivalent
quantile-law diagnostic evaluates `F_y(q_p)=p` at each exact Normal quantile and
uses chain-aware batch-means MCSE on the five indicator series. Empirical
quantiles remain reported. This correction was frozen before observing funnel
training or HMC results, preserves the 99% individual-interval rule, and does
not alter the target, objective, sampler gates, or campaign budget.

## Terminal execution outcome (2026-08-13)

The staged protocol completed for both fresh d100 targets within the four-hour
campaign cap. The terminal decision is preserved in
`docs/plans/bayesfilter-weighted-forward-kl-paper-d100-result-2026-08-13.md`.

| Target/arm | Training | HMC sampler | Exact-law screen | Terminal status |
|---|---|---|---|---|
| Gaussian reverse | 5000-update finite checkpoint | Passed after smaller-step repair, `L=32` | Projection-2 mean interval failed | Candidate rejected |
| Gaussian forward | 5000-update finite checkpoint | Passed, `L=25` | Projection-2 mean interval failed | Candidate rejected |
| Funnel reverse | 5000-update finite checkpoint | Passed, `L=3` | Tail and quantile-law intervals failed | Candidate rejected |
| Funnel forward | 5000-update finite checkpoint | Passed, `L=10` | All 9 structural and 5 quantile-law intervals passed | Viable target-specific candidate |

Recorded execution was `11,931.964 s = 3.314435 h`. Every serious root has a
run manifest and hash ledger; all 24 d100 ledgers were reverified with zero
mismatches. The focused d100 suite completed with `20 passed`; syntax and
`git diff --check` passed. No 10,000-update repair was launched because the
remaining cap was reserved for the required funnel HMC screens.

This result is terminal for the reviewed two-target rung. It does not establish
objective superiority, default readiness, universal funnel performance, or a
general Gaussian failure. A continuation requires a new reviewed plan with
fresh training/HMC seeds, especially replication of the funnel forward pass.
