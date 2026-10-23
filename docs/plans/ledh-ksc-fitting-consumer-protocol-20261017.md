# KSC future guidance: distinguish the consumer from the fit

Status, 17 October 2026: the fitted-guide campaign is
[complete](artifacts/ledh-ksc-fitted-guides-20261017/result.md), following the
[reference/representation diagnostic](artifacts/ledh-ksc-guidance-20261017/result.md).
Neither fitted family qualifies for promotion. The independent audit passes.
The subsequent [consumer-information diagnosis](artifacts/ledh-ksc-information-20261017/result.md)
is also complete: it confirms omitted between-component variance, without
establishing a cause of score error or an improved score estimator. The next
stage is analytical moment-conditioning design, before another fitting ladder.
Both campaigns are closed; their unused time does not authorize another launch.
Work in `ledh-refactor-with-policy-fix`, checkout
`/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922`.
Preserve the numerical evidence and unrelated working-tree edits.

## Question and evidence contract

Does a target-specific fitted future message help estimate the likelihood
score in the existing LEDH mixture consumer, and can any failure be attributed
to the fit rather than to how that consumer uses an accurate message?

The target remains `sqmc_ksc_seven_mixture_q1_prior1_predict_first_v1`, with
two observations, prior N(0,1), process variance 1 and coordinates
`(Phi^-1(gamma), log(beta))=(1.5,0)`. It is the seven-component observation
approximation, not the native stochastic-volatility likelihood. Reference
likelihoods and scores come from the checked two-time Gaussian-mixture
enumeration. Preserve the ordinary, negative-tail and positive-tail conditions
`[0.2,-0.4]`, `[0.2,-8]` and `[0.2,5]` separately.

The principal comparator is unguided global LEDH with beta `(0.1,0.9,0)`;
the equal-mixture consumer is a separately reported comparator. Freeze four
fitted arms: quadratic/global, quadratic/equal, kernel-mixture/global and
kernel-mixture/equal. All four use unguided global for the primary uncertainty
comparisons; matching unguided consumers remain heuristic checks. Candidate and
comparator must share particles, cloud randomness, flow, proposal-density
correction, OT and reset. Keep N=32 and the current diagnostic controls for
this mechanism comparison. These controls are untuned hypotheses, not an
admissible canonical scope or new defaults.

The eventual downstream criterion is lower conditional score MSE against the
enumerated reference for each coordinate and observation condition, supported
by paired uncertainty. Fit loss, likelihood variance and particle ESS are
explanatory only. Invalid references, failed wiring/analytical derivatives,
nonfinite output, corrupted artifacts or missing required diagnostics stop
execution. A valid candidate's failure against a heuristic prevents promotion
but permits the planned mechanism diagnosis or another eligible fit.

The constructed heuristic set is: identity-only proposal (physical predictive
baseline); unguided global (one cheap shared map); unguided local (local
covariance without future information); unguided equal mixture (diversification
without fitting); and Gaussian observation-moment approximation (cheap
model approximation). The last deliberately uses a different likelihood and
must be labelled as a sanity comparator, never the reference. Evaluate all
five in every condition and coordinate. This is a diagnostic comparison
within one consumer, not a claim to beat tuned classical particle filters.

Preserve code hashes, exact commands, environment, seeds, device/growth/XLA
settings, budgets, all attempts, per-row results, paired summaries and a
machine-readable heuristic verdict under a fresh versioned artifact directory.
No result here establishes canonical Contract-E admission, native-SV accuracy,
cross-model transfer, HMC readiness or a production default.

## What is already established about the mechanism

For the second observation, write gamma for the AR coefficient and let the
observation-noise mixture have weights w_j, means m_j and variances v_j.
The exact future message at the first updated state is

    h(x) = sum_j w_j Normal(y_1; gamma*x + 2*log(beta) + m_j, 1 + v_j).

At gamma>0, gamma*h is a normalized mixture with centers
`(y_1-2*log(beta)-m_j)/gamma` and variances `(1+v_j)/gamma^2`.
This normalization is specific to this scalar two-time model. It does not
justify treating an arbitrary backward-information function as a density.
The terminal guide is constant. Inject future information once; keep the
current observation in the actual target weight rather than multiplying it
into the future guide again.

The checked call chain is `future_guides` in
`bayesfilter/score_study/ksc_guidance_diagnostic_tf.py` to
`make_diagnostic_programs` in
`bayesfilter/score_study/guided_mixture_diagnostic_tf.py`, then the shared
mixture proposal/analytical-recursion implementation. Existing focused wiring
and derivative checks and the 192-row source-audited run support this bounded
consumer claim; they do not establish a canonical production call chain.

The exact message supplies local quadratic coefficients to a finite LEDH map.
For a mixture these use responsibility-weighted positive precision, not the
possibly indefinite negative Hessian of log(h). The spatial log-gradient is
exact at the anchor; the finite map is not an exact draw from the globally
twisted distribution. Proposal weights still correct to the declared physical
filtering target. This is not the complete controlled-SMC twisted recursion,
so its variance guarantee cannot simply be transferred to this consumer.

There is also a separate score issue. For any positive finite likelihood
estimate Zhat with its total derivative D, and exact reference score s,

    D log(Zhat) - s = (D Zhat - s*Zhat) / Zhat.

Reducing variability of Zhat alone need not reduce this ratio's error. The
derivative numerator and its dependence on the denominator matter. Thus an
exact future message is an information control, not a downstream optimum or
an upper bound on the usefulness of fitted messages.

The completed experiment found descriptive heuristic losses for all guided
arms, including exact future guidance. This weakens the assumption that more
representation capacity is the missing ingredient. It neither proves that
the consumer is irreparable nor supplies a supported population ranking.

## Implementation sequence

1. Preserve a zero-guide replay and the exact-future control through the
   public diagnostic endpoint. Add any fitted adapter to that same endpoint;
   do not create a reduced LEDH implementation. Verify that all guide arrays
   are frozen at the base parameter while state/anchor tangents remain active.
   Finite-difference checks must reuse the frozen guide. Differentiating a
   refitted guide would be a different finite program.
2. Implement the scalar KSC controlled-SMC pilot adapter using the existing
   `ScalarGaussianControlledSMC` fitter and the actual seven-mixture observation
   density. Its first predictive variance is `gamma^2+1`, not a Gaussian
   pilot's inherited process/observation variance. Extract the next-time
   policy, integrate its transition once to obtain the first future message,
   and use a constant terminal message. Record fit residual, Gram conditioning,
   projections and admissibility. An integrable quadratic twist may have
   negative curvature; the current LEDH guide requires nonnegative precision.
   Mark such a fit ineligible. Do not silently clip it. A constrained fitter
   would be a separately defined local extension.
3. Implement an explicitly local positive-kernel mixture fit for h. The
   replicated `GaussianMixtureIWSG` is a resampling primitive, not a trained
   backward-density estimator. Calling the new fit original-paper KDM/IWSG
   training would be unsupported. Keep the distinction in its module and
   results. Use TensorFlow/TFP numerical operations, stable graph signatures
   and no pfor; no neural architecture or optimizer is required for this fit.
4. Verify that the fitted mixture and quadratic use the same guide slots and
   proposal-density denominator as their controls. Check analytical directions
   for both coordinates with coefficients held fixed, exact-mixture
   representation parity, zero-guide replay, invalid-fit handling and disjoint
   seed wiring. Reuse the existing moment reset; label it diagnostic. A
   canonical claim requires a separate full lifecycle and per-scope tuning.
5. Only after focused checks pass, instantiate the bounded fitting campaign
   below with runnable commands and fresh output paths. Preserve unsuccessful
   fits; do not choose them retrospectively using downstream final scores.

## Target-specific fitting design

For the quadratic pilot, compare 64/256 particles and one/two backward-fit
passes. These are a capacity/budget ladder, not transferred optimal settings.
Use independent forward-pilot noise and actual observation likelihoods. The
source fitter uses regression rather than a learning-rate optimizer; retain
its existing precision/projection policy with explicit diagnostics. Projection
does not automatically make its quadratic eligible for the LEDH guide.

For the kernel fit, construct an importance design density

    r(x) = 0.5 Normal(x; 0, gamma^2+1)
           + 0.5 Normal(x; m, 4*vmax),
    m = (y_1 - 2*log(beta) - sum_j w_j*m_j)/gamma,
    vmax = max_j(1+v_j)/gamma^2.

This uses known model moments and future observation, not oracle fitted
parameters. The wider component is a tail-coverage hypothesis. Draw independent
design states x_i from r and 16 transition replicates z_il from f(.|x_i).
Compute `hhat_i = mean_l g(y_1|z_il)` and normalize weights proportional to
`hhat_i/r(x_i)` in log space. Then fit

    q_fit(x) = sum_i normalized_weight_i Normal(x; x_i, bandwidth^2).

The identity `E_r[(hhat/r)*phi(x)] = integral h(x)*phi(x) dx` explains the
importance measure. It does not make the self-normalized finite fit unbiased.
Use 64/256 centers and bandwidth multipliers 0.5/1/2 times weighted standard
deviation times effective-weight-count to the power -1/5. Record the raw
variance, effective count, largest weight and normalization. Zero/nonfinite
variance or bandwidth is an invalid fit, not authority for an undocumented
ridge. The three multipliers are a declared small capacity/smoothing search,
not universal bandwidths. The fixed 16-replicate design is an initial budget
hypothesis; include exact-h weights as a labelled representation-only control
to distinguish transition Monte Carlo error from density approximation.

Validate normalized-message cross-entropy using a disjoint importance design
and transition noise, with normalization separately estimated for that design.
Use exact normalized h to report validation error in this oracle-accessible
diagnostic. A quadratic message `exp(-a*x^2-b*x-c)` has a normalized Gaussian
density only when `a>0`. Guide admissibility alone is insufficient: a constant
zero-guide control is valid for replay but has no normalized-message loss.
Treat a fitted quadratic with `a<=0` as ineligible for this density-fit
comparison, rather than assigning it a Gaussian normalization or infinite
evidence against the fitting method. For eligible quadratics, normalize their
Gaussian messages before computing the same loss. Compare scale-normalized losses per condition rather
than unnormalized likelihood amplitudes. Select each family's settings on
validation message loss only, then freeze them. Report the local derivative
and tail fit errors as explanatory diagnostics. Neither loss is a downstream
score-promotion criterion.

Use four independent fitting seeds per capacity setting during development;
use distinct calibration, validation and final seed namespaces and record the
exact table before launch. Existing October-17 clouds are exploratory and
must not enter final evaluation. Final inference is conditional on the three
fixed datasets; it does not estimate performance over random datasets.

## Executable campaign freeze — 2026-10-17

Implementation: `scripts/run_ledh_ksc_fitted_guides.py`; post-run analysis:
`scripts/analyze_ledh_ksc_fitted_guides.py`. The user directive to continue
executes this bounded development/final campaign; it does not reopen the
closed exact-message campaign. Both launches use the `tftwogpu` environment,
GPU 0, trusted escalated access, FP64/XLA, TF32 disabled and verified memory
growth. No parameter, protection setting or scientific default changes.

The complete prelaunch seed table is
`artifacts/ledh-ksc-fitted-guides-20261017/seed-design.json`. Calibration uses
651000 + 1000*condition + 10*configuration + replicate (four replicates);
validation uses 661000 + 100*condition + replicate. Final fitting uses
671000 + 1000*condition + 100*family + cluster, and final clouds use
681000 + 1000*condition + 10*cluster + replicate (eight by eight). Condition
and family ordering are recorded by the runner. Stateless stream IDs are
also in the seed table. No old final cloud or development seed is reused.

Validation uses 256 independent design points and 64 transition samples per
point, with one disjoint validation design for each of the four development
replicates. Settings are selected by average excess normalized-message
cross-entropy across all three conditions and all four replicates; the
exact-message entropy subtraction is identical across configurations and
cannot change their ordering. All twelve development fits must be eligible
for a setting to enter selection. Ties use the declared configuration order.
Exact-h weighting on both training nodes and validation nodes is reported
only as an explanatory oracle control, never used for selection. The tail
diagnostic uses |x| > 2*sqrt(1+gamma^2), two predictive standard deviations;
it has no tuning or promotion role. These finite design sizes are budget
hypotheses. Their sampling variability remains a limitation.

Fresh outputs: `artifacts/ledh-ksc-fitted-guides-20261017/development-01` and
`artifacts/ledh-ksc-fitted-guides-20261017/final-02`. Each launch is capped
at 450 seconds including imports/compilation; total GPU-process wall ceiling
900 seconds, at most two launches. CPU-only checks/analysis have a separate
180-second ceiling; independent references receive at most 60 seconds. The
previous adapter implementation tests predate this campaign; current
analysis/XLA smokes consumed 8.52 reported test seconds (reserve 15 wall
seconds conservatively). No third launch is authorized by this plan.

Commands, from the LEDH worktree (P=/home/chakwong/anaconda3/envs/tftwogpu/bin/python,
A=docs/plans/artifacts/ledh-ksc-fitted-guides-20261017):

```bash
CUDA_VISIBLE_DEVICES=-1 $P scripts/run_ledh_ksc_fitted_guides.py --stage seeds --output $A/seed-design.json
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true $P scripts/run_ledh_ksc_fitted_guides.py --stage development --output $A/development-01 --budget-seconds 450
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true $P scripts/run_ledh_ksc_fitted_guides.py --stage final --output $A/final-02 --frozen $A/development-01/frozen-selection.json --budget-seconds 450
CUDA_VISIBLE_DEVICES=-1 $P scripts/analyze_ledh_ksc_fitted_guides.py --run $A/final-02
```

Complete command logs are retained beside each output directory. Prelaunch
hashes record the runner, adapter, consumer, analysis and tests; each manifest
additionally records the loaded repository source dependency closure before
execution and verifies it afterward. Final execution rejects changed source
or seed design relative to the frozen selection. All 2,304 final cells are
required, including explicit ineligible-fit rows; no seed replacements or
successful-cell-only inference. Frozen-coefficient finite differences test
both coordinates at two step sizes for the first final fit/cloud of each
eligible family and condition.

Skeptical prelaunch audit: PASS for this diagnostic question. The physical
seven-mixture target, predict-first variance, denominator and Contract-E
consumer remain unchanged. The quadratic and kernel fits reach the shared
consumer through executable parity tests; inverse-design weighting and
once-only future integration are checked. Selection never uses final scores.
The baseline is unguided global under identical clouds and controls; four
physical heuristics and the explicitly different-target Gaussian heuristic
are evaluated conditionally. The analysis tests missing/duplicate/unpaired
cells and preserves failed fits. The uncertainty unit is eight independent
fit clusters, not 64 independent fits, and multiplicity is 24 contrasts.
CPU-XLA smoke tests pass for both fitted families and loss evaluation. GPU
compatibility and total runtime remain unproved until these launches; an
over-budget result stays incomplete. Pure numerical fits and consumers are
XLA compiled with stable signatures; host diagnostics, exact-density
reference evaluation and selection reduction are explicit eager diagnostic
exceptions, not a production implementation claim.

## Fresh campaign boundary and decision rule

The next serious campaign has a proposed ceiling of 900 GPU wall seconds,
two launches of at most 450 seconds each, and 180 CPU seconds for independent
checks/analysis. This is a new ledger, not unused time from the closed stage.
Before launch, record executable commands, the complete seed table and source
hashes in its plan/manifest; cap the runner at these limits. Routine adapter
implementation and small tests can proceed under the current user directive.

Run FP64 GPU/XLA with TF32 off as a diagnostic reference exception, trusted
GPU access and verified memory growth. Freeze OT (1,5), N=32, 16 flow steps,
two correction steps, basis size four, rate 0.1 and radius 0.25 solely to
isolate the fitting mechanism. Record this restriction; no setting is tuned
or promoted by this experiment. Do not alter protection parameters to force
an improvement. Record eligibility/projection rates before downstream metrics.

Use eight new independent fit seeds and eight independent final clouds per
fit, paired across arms in every condition. Refit only under the frozen
protocol; never use final scores to alter fit or consumer settings. Report
the 64 final rows per condition and arm, but treat the eight fits as the
independent clusters. Use paired differences in squared score error and
cluster-level uncertainty; do not treat the 64 rows as 64 independent fits.
If the budget cannot cover all frozen arms and required diagnostics, stop as
incomplete rather than dropping unfavorable arms or silently reducing seeds.

Declare simultaneous 95% intervals across all four fitted arms, three
conditions and two coordinates: 24 candidate-minus-unguided-global MSE
comparisons. Use Bonferroni-adjusted paired t intervals on the eight independent
fit-cluster means, with seven degrees of freedom and two-sided error `0.05/24`
per interval. The small cluster count and distributional assumption must
accompany the result; keep per-cluster data. An arm's improvement across all
conditions requires upper bounds below zero for its six differences. Otherwise
report condition-specific evidence or no supported ranking under these same
simultaneous intervals. Other contrasts are descriptive unless their
uncertainty and multiplicity are declared before execution. There is
no score-based candidate selection. Any descriptive heuristic loss vetoes
promotion under the inherited conservative heuristic screen; do not call that
loss statistically established harm without uncertainty support.

## Skeptical audit, defaults and stop conditions

The principal risk is an accurate message being treated as proof of a useful
proposal. The existing exact-message control already refutes that shortcut
for these descriptive results. The algebra above explains why fitting and
consumer effects must be separated. No extra capacity is justified by fit
loss alone. A second risk is fitting the future posterior and calling it h;
the explicit `1/r` measure correction prevents that target substitution.

The pre-execution skeptical audit corrected two further flaws: guide
admissibility does not imply a normalizable fitted density, and correcting
only six intervals would ignore the four fitted arms. The strict-positive
quadratic normalization check and 24-comparison uncertainty family address
these issues. Candidate ineligibility is separated from invalid execution.
With these repairs, the protocol supports routine adapter development; it
still requires the exact commands, seed table and bounded launch manifest
specified above before any serious campaign.

The target/prior/Q, enumeration reference and guide formula come from the
checked KSC diagnostic; reference agreement and coefficient/parity checks are
their early tests. Pilot sizes, replicate count, bandwidth rule, design tails
and fit passes are new hypotheses; heldout loss, importance concentration,
conditioning and exact-h controls expose failure. Frozen consumer controls
come from the completed diagnostic and owner OT instruction; validity margins,
zero-guide replay and comparison to all unguided arms expose a consumer failure.
None are transferred tuned defaults. Off/zero safeguards are not new choices
in this stage; changing a numerics-altering protection requires its own
non-harm evaluation, including healthy-output equivalence.

Stop for numerical/artifact/reference invalidity, an accepted candidate
violating the runtime guide domain, absent uncertainty diagnostics, or
exhausted compute/attempt budget. An ineligible fit found by the declared
preflight check is a candidate rejection, not a continuation veto; preserve
its failure and continue eligible arms without replacing missing inference
cells or relabelling it as evidence against KDM or controlled SMC. A supported
failure of fitted and exact guides calls for
consumer analysis rather than unbounded capacity search. Even a favorable
screen requires a new untouched, adequately powered confirmation before any
default, canonical or cross-model claim.

Terminal review must separate engineering correctness, numerical validity and
scientific interpretation; include decision and inference-status tables.
The strongest alternative explanation remains finite-particle error and the
diagnostic reset/proposal geometry. A different consumer or stronger final
uncertainty could overturn the present interpretation. The MathDev structural
matcher's inconclusive result remains a recorded limitation, not a proof.
