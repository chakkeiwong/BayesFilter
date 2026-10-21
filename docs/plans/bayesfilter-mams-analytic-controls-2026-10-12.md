# D4R3 analytic quantity and oscillation controls

Date: 12 October 2026. This is the bounded remaining analytic-control work in
the [generic repair](bayesfilter-generic-trajectory-adaptation-repair-2026-10-11.md),
executed alongside the immutable D4R4b cost pilot. It changes diagnostic
fixtures and the analytic validation consumer, not tuning membership. NUTS
remains excluded.

## Question and pre-execution audit

Can the diagnostics expose slow squared quantities, missed tails and wrong
model-coordinate maps even when coordinate means or local checks look good?
The comparators are exact Gaussian, Student-t(5) and centered-funnel laws;
the oscillation control has a known stationary Gaussian AR(1) law. These
checks assess diagnostic mechanisms, not the comparative efficiency of MAMS.

The audit found two material baseline defects before execution. The partial
funnel's density implements `v=3*z0`, `x=exp(v/2+delta)*zchild`, but its
inherited `to_model` implements `v=z0`, `x=exp(z0/2)*zchild`. That map is
wrong for its declared target. Also, the old exact-map fixture used a
unit-scale funnel while the partial-map fixture represented scale three.
Repair the partial map and version its identity; use a diagnostic exact-map
adapter with `v=3*z0` so both fixtures represent the same model. Preserve old
results as the earlier, explicitly limited feasibility experiment.

Use the existing public tuning authorities and retained-member replay. New
quantities are passed only to posterior assessment and post-run reference
checks. No mean, tail, R-hat, ESS, MCSE or endpoint-return statistic may remove
a verified tuning member. All member outcomes remain recorded.

## Mathematical and evidence contract

For the common funnel, `V=3Z`, `X_j=exp(V/2)U_j`, with independent standard
normals `Z,U_1,U_2`. The exact chart samples `(Z,U)`; the partial chart samples
`(Z,exp(-delta(Z))*U)`, where `delta(Z)=.5*tanh(Z/2)`. The latter chart's
log Jacobian is `log(3)+V+2*delta`, and its target density, up to constants, is
`-.5*(Z^2+sum(zchild^2)*exp(2*delta))+2*delta`. Therefore

* `E[V]=0`, `E[V^2]=9`, `E[X_j]=0`, `E[X_j^2]=exp(9/2)`;
* `P(V<-3)=P(V>3)=Phi(-1)`;
* `P(|X_j|>2*exp(V/2))=2*Phi(-2)`;
* the joint neck/tail probability is `Phi(-1)*2*Phi(-2)`.

Model-coordinate means and squares are reported. For the scale-three funnel,
the child square has fourth-moment dependence `E[X_j^4]=3*exp(18)`, so a
small run is particularly unreliable for estimating its uncertainty. Keep
child moments descriptive in this bounded phase; use the bounded model neck,
flare and conditional-tail events for reference screens. This limitation does
not excuse missing model-coordinate quantities in the eventual confirmation.

For Student-t(5), `E[q]=0`, `E[q^2]=5/3`, and the exact two-sided tail at two
is `I_(5/9)(5/2,1/2)`. Its fourth moment exists; its eighth does not. Never
make the fourth moment's MCSE a precision target. The tail threshold two is
a convenience-selected moderate-tail stress point, not a default policy.

| Evidence | Role |
| --- | --- |
| Independent map, Jacobian, density and exact expectation identities | Engineering pass criteria; mismatch vetoes use of that fixture |
| Negative AR(1) with fast means and slow squares; exact sign-flip endpoint return | Mechanism controls; endpoint return alone has no numerical-invalidity authority |
| Existing lugsail MCSE and Bonferroni normal intervals against exact quantity expectations | Approximate fixed-horizon reference diagnostic; discrepancy vetoes a reference-agreement claim, not tuning membership |
| Missing batches, constant quantities, nonfinite values or nonpositive long-run variance | Inconclusive/invalid reference report; never a fabricated zero MCSE or a pass |
| Exact-law repeated calibration and known tail-removal/map mutations | Check false alarms and sensitivity in the declared synthetic cases; no MCMC-wide coverage claim |
| Complete source, seed, target and result inventories | Execution validity; missing/corrupt evidence stops dependent interpretation |
| Aggregate allocation exhaustion | Resource continuation stop; preserve partial counts |

Per-member multiplicity counts all declared screened quantities, including
unavailable ones. The working family level is .05 (inherited research-screen
convention); approximate normal/lugsail intervals are neither exact intervals
nor anytime-valid intervals for the sequential posterior controller. Record
that distinction in every reference report. Do not claim familywise coverage
across candidates, targets or repeated looks, nor infer accuracy from
non-rejection. Unsupported state-space references remain explicitly absent.

## Implementation and tests

1. Correct the partial-funnel model map and version its signature. Add the
   exact diagnostic whitening adapter for the same scale-three model. Check
   both against the independently evaluated centered density and full map
   Jacobian, without pfor.
2. Add one diagnostic quantity authority: versioned names, coordinate roles,
   exact expectations and explicit precision/screen roles. Wire it into the
   existing validation consumer before posterior sampling; report original
   model quantities as well as adapter coordinates. No target-name branches
   enter the reusable proposer or controller.
3. Add fixed-policy antithetic/oscillation regressions. For stationary AR(1),
   `tau(q)=(1+rho)/(1-rho)` and `tau(q^2)=(1+rho^2)/(1-rho^2)`. At rho=-.9,
   means can look efficient while squares are slow. Finite-window estimates
   need only expose that separation or declare insufficient support; do not
   fit tolerances to one observed estimate. Test the exact periodic limit
   separately, where square variance is zero and no IACT may be invented.
4. Add reference-report regressions for wrong charts, dropped tails, constant
   indicators, nonfinite/short chains, quantity-family counting, and absent
   model references. Exercise the actual posterior callback/precision names.
   Include a symmetric, bounded finite bank normalized to mean zero and second
   moment `5/3`, with no observations beyond two. This isolates the tail check
   from the mean/variance checks; ordinary clipping changes both. Match funnel
   starts in model coordinates by applying the inverse partial chart.
5. Run a bounded diagnostic calibration on exact draws and known stationary
   Gaussian-copula AR(1) draws, with independent replicates and preserved chain
   axes. Rho in `{0,.9,-.9}` is a declared sensitivity grid, not a sampler
   setting. Student-t margins use the independent SciPy inverse CDF; funnel
   charts transform independent Gaussian coordinates analytically. Compare
   all results with exact probabilities; also truncate the Student-t tails
   and apply the known wrong funnel map as negative controls. Rejection or
   inconclusive outcomes both prevent a false pass, and are counted separately.

Calibration uses 32 replications, 2048 draws/chain and four chains, with root
seed 5700. These are convenience-selected cost bounds, not power-certified
sample sizes. Each replicate is one independent alarm observation; report
Clopper-Pearson uncertainty and all denominators. Do not count chain draws as
independent observations for an AR(1) probability interval. Any screening
defect triggers diagnosis rather than post-hoc threshold adjustment. Passing
regressions and calibration still do not establish robustness across models.

## Compute, provenance and continuation

Use up to 1800 seconds of the already reserved D4R4b 3600-second local-check
allowance for implementation tests and calibration together. Approximately
625 seconds were recorded before this suballocation; keep all remaining local
reserve until settlement. This adds no budget. CPU diagnostics intentionally
hide GPUs before TensorFlow import and use two CPU threads; XLA is exercised
for compatible numerical functions, with explicit graph/debug exceptions in
focused regressions. The running GPU pilot remains undisturbed.

Run focused pytest files for the new quantities, existing validation lifecycle
and pilot diagnostics. Archive the exact changed diagnostic source and use a
scoped commit before the calibration driver executes. Its manifest records
command, source hashes, environment, seed, CPU-only status, elapsed time,
plan and result. Use fresh output under
`N5/generic-adaptation/preconfirmation-01/analytic-controls-01/`. Record actual
commands and tests in a result note, then update the master and settle cost.

The skeptical audit now passes for this bounded diagnostic work: exact and
partial controls share a model, quantities have declared moments, the reference
uncertainty does not become tuning evidence, negative controls can expose false
reassurance, and the allocation has a stop. The strongest alternative explanation
for success is that synthetic stationary controls are easier than a transient
MCMC chain. Fresh public-pipeline calibration and the priced state-space
confirmation therefore remain necessary after this stage.

Precommit checks: 24 focused quantity, oscillation and validation-lifecycle
tests passed in 6.44 seconds (8.75 seconds including process startup), with
GPUs hidden. Independent finite differences checked the full model-map
Jacobian; the negative controls rejected the wrong map and refused a fabricated
zero tail MCSE. These are engineering checks. Run the same tests and the
existing proposer regressions from the committed source before calibration.
Calibration is subcapped at 1,200 seconds of the 1,800-second allowance; this
is a resource allocation, not an expected runtime or a deadlock threshold.

## Post-calibration continuation audit

The calibration finished in 23.603492 seconds; 40 committed-source regressions
and two final review regressions passed. It did not establish nominal MCSE
coverage, and antithetic MCSE availability is an open diagnostic issue. The
next smallest independent check is actual public-tuner/export/reload/posterior
wiring with the newly declared quantities. This can expose a callback, schema,
coordinate or stopping failure that an exact-reference calculation cannot.

While both GPUs are occupied, run the three small analytic fixtures as an
explicit CPU graph reference/debug exception, with GPUs hidden and XLA off
through the existing `--cpu-reference` switch. This is not a production-target
or GPU qualification run. Use unchanged feasibility settings from the October
11 bounded plan: four chains, 64-draw fixed pilot windows, fresh 32-by-64
measurement and verification repetitions, at most four primary/eight total
policies, 512--1024 warm-up and at most 1024 retained transitions per chain.
These inherited short allocations test wiring, not broad posterior adequacy.
The new model/tail quantities and matched funnel model starts are frozen in
the next source commit before execution.

Use root seed 5800, a disjoint convenience block, and fresh output
`preconfirmation-01/analytic-pipeline-01`. Subcap this run at 900 supervisor
seconds inside the existing 1800-second local-control suballocation. Known
usage is 75.253492 seconds, so this fits after settling the completed tests
and calibration; the old 1200-second unused calibration reservation is not
charged again. No new campaign allocation is needed.

```bash
env CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
  TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 TF_CPP_MIN_LOG_LEVEL=2 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/validate_mams_adaptation.py \
  --output /home/ubuntu/python/BayesFilter/docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/preconfirmation-01/analytic-pipeline-01 \
  --targets student_t funnel_noncentered residual_funnel --seed 5800 \
  --cpu-reference --max-wall-seconds 900
```

The engineering endpoint is all three target dispositions, every created and
verified member retained, correct callback/quantity identities, separate
posterior/reference outcomes and source/stream provenance. A short posterior
failure or unavailable MCSE is expected evidence, not a reason to remove a
tuning member or stop valid peers. A shared source/target/identity defect is a
continuation veto and repair trigger. No method ranking or nominal coverage
claim follows even if every diagnostic passes. The audit passes for this
bounded exception: it answers the wiring question on actual public paths,
fits the existing reserve and leaves the GPU/XLA evidence requirement intact.
