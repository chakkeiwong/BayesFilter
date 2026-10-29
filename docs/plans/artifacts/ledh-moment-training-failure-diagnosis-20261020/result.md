# Why the simulated-moment training did not generalize

20 October 2026. Diagnostic work on `ledh-refactor-with-policy-fix`, base commit
`4d6915cc66e701c4c7b657e094306d298fdd3e76`. No runtime algorithm, optimizer,
numerical default or selected adapter changed during this investigation.

The training loss is correct for the declared terminal-moment target. Training
also changes the proposals and their estimated moments: the controls are not
disconnected. The negative result concerns a much narrower implementation than
training the replicated controlled-SMC and KDM algorithms end to end. It tunes
three or twelve adapter coefficients around frozen messages, while requiring
half the draws to come from an LEDH component that receives negligible weight
at the later times in this replay. The optimizer makes small-minibatch progress
that does not reliably carry to validation or the untouched evaluation.

These are checked restrictions and observed failure mechanisms. Their separate
contributions to the final loss gap have not been identified experimentally.
Neither the population loss identity nor future-guided proposals in general
are refuted. There is no evidence that another Sinkhorn-iteration sweep would
address the findings below.

## What failed, and what was checked

The [completed independent evaluation](../ledh-flow-aware-proposal-repair-20261020/result.md)
used 32 simulated datasets and two independent particle replications per dataset.
It found no statistically supported gain of either selected adapter over its
own untrained adapter. Trained-minus-untrained mean loss differences were
0.01713 for cSMC, with pointwise 95% interval [-0.02724, 0.06348], and 0.00362
for the kernel arm, with interval [-0.00434, 0.01315]. Thus it did not establish
that training itself harmed the estimators either. Both trained arms had higher
pooled loss than the conditional-only control and failed the recursive-Gaussian
comparison in both predeclared observation regimes. Those failures remain
promotion vetoes. No score-accuracy conclusion follows from these moment losses.

This investigation replayed eight saved training/validation datasets with two
independent particle streams, seven arms and unchanged numerical controls:
SIR d18, T5, N64, FP64 GPU/XLA, TF32 off, OT (1,5), balance32, K=N, UKF,
dual-cap GenUT and Contract E--Chol. It did not reopen the holdout for tuning.
There are 112 complete traces, seven direct consumer-parity checks, and local
adapter finite differences on four datasets, totaling 255 kernel/endpoint calls.
All replay validity checks passed. All eight recorded runtime hashes match the
completed training run. Every trace-derived loss agrees exactly with the actual
training endpoint for the seven parity fixtures. This is executable call-chain
evidence, not inspection of an unused helper.

| Consumer operation | Executed source, relative to the worktree | Finding |
| --- | --- | --- |
| Construct candidates | `bayesfilter/score_study/physical_proposal_diagnostic_campaign.py:65` | Both guide arms fix the physical-component probability at 0.5 |
| Fit frozen messages | `bayesfilter/score_study/sir_backward_guidance_diagnostic_tf.py:191` | Controlled-SMC fit, followed by quadratic and positive-RBF exports |
| Adapt messages | `bayesfilter/score_study/moment_training_tf.py:47` | Three shared or twelve time-specific coefficients |
| Draw and evaluate densities | `bayesfilter/highdim/ledh_physical_proposal_tf.py:67` | One sampled component; both full densities evaluated at that point |
| Correct importance weights | `bayesfilter/highdim/ledh_canonical_score_tf.py:558` | Transition density divided by the full proposal, then observation likelihood |
| Reset particle cloud | `bayesfilter/highdim/ledh_canonical_score_tf.py:664` | Contract E and the shared higher-moment correction |
| Compute training loss | `bayesfilter/score_study/moment_training_tf.py:84` | Terminal post-reset states, with the label used only in the objective |
| Update adapters | `bayesfilter/score_study/moment_training_tf.py:109` | Two-sided random-direction search and same-minibatch backtracking |

## The objective does not require posterior moments or scores as labels

Write C=(theta,Y_1:T). Let Phi contain the standardized state coordinates and
their upper-triangular raw second products. Calibration fixes the centering,
coordinate scales and feature scales on separate simulations. The matrix W
includes the squared inverse feature scales and the two block weights: 1/(2d)
for first moments and 1/[2d(d+1)/2] for second moments. These reproduce
`make_moment_objective` at `moment_training_tf.py:22`.

For a fixed adapter eta, the particle and guide-fitting randomness U is
independent of the simulated terminal state X_T conditional on C. The inspected
consumer supplies observations to the filter and supplies X_T only to the loss.
With mu(C)=E[Phi(X_T)|C], expanding the square gives

\[
\begin{aligned}
E\|\widehat\mu_\eta(C,U)-\Phi(X_T)\|_W^2
&=E\|\widehat\mu_\eta(C,U)-\mu(C)\|_W^2\\
&\quad+E\|\Phi(X_T)-\mu(C)\|_W^2.
\end{aligned}
\]

The cross term is zero because, conditional on C and U, the label residual has
mean zero. The second term does not depend on eta. This proves the intended
population statement. It applies to fixed eta or to a learned eta evaluated on
independent examples; it does not make adaptively selected training loss an
unbiased estimate of generalization performance. A single latent draw labels
raw second moments, not covariance directly.

The same algebra explains noisy local decisions. For fixed candidates and
particle randomness, put delta=hat_mu_eta_prime-hat_mu_eta. On one labeled
dataset the loss change is

\[
\Delta\ell=2\delta^T W(\widehat\mu_\eta-\Phi(X_T))
                 +\delta^T W\delta.
\]

Its conditional variance due solely to the latent label is

\[
4\delta^T W\operatorname{Cov}(\Phi(X_T)\mid C)W\delta.
\]

It vanishes only in special cases, although its contribution to the expected
cross term is zero. Sharing random numbers between the two candidates is
correct and is already implemented. It cannot remove this label variability
or the variability of the finite-particle estimator across new streams.

## The trainable family is small, and the kernel arm is not the replicated KDM learner

For each active time the quadratic adapter has only a strength s=exp(eta_0)
and two shifts repeated across all nine SIR regions. If its frozen information
parameters are H and b, the code uses

\[
H_\eta=sH,\qquad b_\eta=s(b+H\delta).
\]

It cannot learn region-specific coefficients or rotate H. Bounds restrict
s to [1/4,4]. Thus it cannot continuously turn an active quadratic guide fully
off. The selected pooled cSMC adapter mainly attenuates the frozen guide:
s=0.324382. This is a learned attenuation, not a zero-guide option. No parameter
bound was hit in training.

The kernel adapter leaves mixture weights and the active-time mask fixed,
translates all centers by the same repeated two-coordinate shift, and divides
all component covariances by s. The consumer then moment-matches the
Gaussian-times-mixture product to one Gaussian per parent before drawing.
It does not sample the uncollapsed product mixture. Its final time is inactive,
as appropriate for a future-only message when no future observation remains.
The fixed half-mixture allocation nevertheless remains active there.

The fitter explicitly identifies itself as a diagnostic, not original KDM
(`sir_backward_guidance_diagnostic_tf.py:1`). It obtains a message from the
replicated Gaussian controlled-SMC fit, then fits a positive RBF approximation
to that same message. The replicated KDM training modules are not called by
this consumer. Calling this experiment end-to-end training of the replicated
KDM algorithm is wrong relative to the executed call chain. The historical
arm name is retained in old files only to preserve provenance.

Two additional representation restrictions are visible in the saved fits:

- All 136 inspected active-time quadratic fits, covering 34 training/validation
  datasets, needed positive-information projection. The median ratio of
  projected to raw message-regression error was 4.97095. For the quadratic
  physical proposal, V_eta=(V_0^{-1}+sH)^{-1} is at most V_0 in the positive
  semidefinite order. Such a family can contract the conditional covariance
  but cannot enlarge it. This statement does not apply to the mixture arm,
  whose between-component covariance can enlarge its matched covariance.
- All 136 kernel fits selected bandwidth 2, the upper end of [0.5,1,2]. The
  median kernel-to-constant message-regression error ratio was 0.85554. The
  32 centers, mixture weights and selected base bandwidths then remained fixed
  during moment training. Boundary selection warrants a capacity diagnostic;
  it does not prove that any larger bandwidth would help terminal moments.

These message-regression errors are explanatory only. They do not prove that
projection or Gaussian moment matching caused a particular terminal loss gap.

## Correct proposal weighting is not full controlled-SMC twisting

Conditional on a fixed parent, let a(x)=f_theta(x|parent)g_theta(y_t|x),
Z_0=integral a, and pi_0=a/Z_0. The actual online consumer uses

\[
q_{\beta,\eta}=(1-\beta)q_{\rm flow}+\beta q_{\rm physical,\eta},
\qquad w(x)=a(x)/q_{\beta,\eta}(x).
\]

There is no future-guide factor in the filtering target a. To see the
consequence exactly, consider the ideal pure tilted proposal
q_psi=pi_0 psi/E_pi_0[psi], with positive integrable psi. Its weight is

\[
w(x)=Z_0\,\frac{E_{\pi_0}[\psi]}{\psi(x)}.
\]

Provided E_pi_0[1/psi] is finite, direct integration gives

\[
E_{q_\psi}w=Z_0,\qquad
\operatorname{Var}_{q_\psi}(w)
=Z_0^2\{E_{\pi_0}[\psi]E_{\pi_0}[1/\psi]-1\}\ge0.
\]

The inequality follows from Cauchy--Schwarz. Pure pi_0 already gives constant
within-parent incremental weights. This is an exact limiting calculation, not
an identity for the implemented Gaussian approximation or its half-mixture.
Across parents, Z_0 also varies, so it does not imply constant weights for the
whole conditional-only filter.

In full controlled SMC the intermediate target changes too. With twisted
transition M_t^psi=M_t psi_t/(M_t psi_t), the subsequent potential has the
form g_t(M_{t+1}psi_{t+1})/psi_t, with terminal lookahead equal to one.
The replicated implementation includes precisely the lookahead and inverse
policy terms in `multivariate_controlled_smc_tf.py:541` and
`multivariate_controlled_smc_tf.py:351`. For an exact backward recursion
psi_t=g_t M_{t+1}psi_{t+1}, these intermediate potentials can be constant,
with the corresponding initial normalizing factor. The hybrid LEDH consumer
instead corrects back to ordinary filtering weights at every time. Therefore
the ideal full-twist variance argument does not establish an improvement for
this consumer. Its filtering weights are correct; omitting their correction
would change the target and would be wrong as a filter repair.

This also does not mean that a proposal guide has no effect. It changes
finite-particle bias, variance and later input clouds. For example, for IID
self-normalized importance samples from a fixed target, the asymptotic
weighted moment-error trace is

\[
\mathcal V(q)=Z^{-2}\int\frac{a(x)^2}{q(x)}
                      \|\Phi(x)-\mu\|_W^2\,dx.
\]

When the required moments exist, Cauchy--Schwarz minimizes this expression at
q proportional to a times ||Phi-mu||_W, subject to support qualifications.
It is generally not the likelihood-optimal proposal a/Z or a generic future
likelihood tilt. This is an explanatory asymptotic result, not a new training
rule requiring unknown mu. The simulated-label objective already avoids that
requirement and measures the actual finite estimator.

## The fixed allocation spends many particles on a component with negligible weight

The shared consumer samples one component per particle with beta=0.5. The
adapter cannot change this probability or repair the affine flow map. Across
the 16 replay streams, the final step assigns 33.375 of 64 draws on average
to the physical component and 30.625 to the flow component. The following
quantities are computed from actual selected labels and normalized weights,
not density responsibilities or subtraction from one.

| Arm | Mean flow-label weight mass, time 3 | Mean flow-label weight mass, time 5 | Maximum flow-label mass, time 5 | Mean terminal ESS |
| --- | ---: | ---: | ---: | ---: |
| Conditional-only control | 0 | 0 | 0 | 13.57 |
| Half-mixture without future guide | 1.09e-7 | 2.94e-17 | 4.60e-16 | 9.29 |
| Selected cSMC adapter | 5.85e-9 | 3.33e-17 | 5.34e-16 | 10.56 |
| Selected kernel adapter | 4.90e-9 | 2.51e-15 | 4.02e-14 | 10.76 |

If only m draws have nonzero normalized weights, Cauchy--Schwarz gives
ESS=1/sum(w_i^2) <= m. The replay is effectively in that situation at late
times: roughly half the draws contribute negligible terminal weight. Exact
importance correction preserves the intended target but cannot recover the
sampling effort spent on those draws. This is a confirmed allocation
inefficiency in the observed runs. It is not a complete causal attribution of
the loss gap, and ESS is not a promotion criterion.

Setting beta=1 would remove this component but would also cancel the affine
LEDH flow in this physical-output construction. It is already the
conditional-only control, not evidence that LEDH itself has improved.

## Small-minibatch descent is not evidence of population learning

The optimizer probes one normalized random direction at eta +/- 0.1u, uses
the sign of the loss difference, and backtracks up to three times on the same
two datasets and their common particle streams. It does not use gradient
magnitudes. New datasets/streams are selected for the next update. Validation
uses eight datasets with two streams each, at initialization, update16 and
the run endpoint. The initial search runs eight updates per configuration;
the selected configuration gets a 24-update extension.

| Saved training quantity | cSMC selected path | Kernel selected path |
| --- | ---: | ---: |
| Trainable coefficients | 3 pooled | 12 time-specific |
| Updates retained along selected path | 8 + 24 | 8 |
| Distinct training datasets encountered | 24 | 12 |
| Initial validation loss | 0.92294248 | 0.94161638 |
| Selected validation loss | 0.91634688 | 0.93142359 |
| Relative selected validation decrease | 0.715% | 1.082% |
| Validation loss after the 24-update extension | 0.91634688 | 0.94070066 |

Across the complete search, 175 of 176 updates were accepted and no bounds
were hit. The optimizer is moving and finding same-batch decreases. The
kernel extension loses most of its earlier validation gain; simply increasing
the update count has already failed to reliably help in this small pilot.

Local coordinate differences at eta=0 also detect real, dataset-dependent
responses. The cosine between the mean directions from two training and two
validation examples was -0.9303 for cSMC and -0.0314 for the kernel adapter.
The first-coordinate signs persist when the difference radius is reduced
from 0.1 to 0.05 and 0.025. These are four-example diagnostic observations,
not estimates of population gradient alignment or proof that no good adapter
exists. Median absolute changes of loss between two particle streams for a
fixed dataset were 0.1530 and 0.2168 for the selected cSMC and kernel arms.
These stream changes are not estimates of latent-label noise.

Together, the small data coverage, restricted search and unstable validation
response support a failure of reliable generalization in this pilot. They
do not identify a universal learning rate, prove optimizer convergence, or
guarantee that more data or more expressive guides will succeed.

## The checked reset does not erase the training moments

Across all arms, times and replay streams, the largest scaled first/second
moment discrepancy between the weighted pre-reset cloud and unweighted
post-reset cloud is 1.37e-12. The largest terminal scalar-loss change across
that reset is 1.11e-13. These are successful numerical agreement checks, not
blockers. They rule out a material loss of these particular moments at the
final reset in the checked runs. Earlier resets can still affect higher-order
shape and subsequent transitions; that broader effect has not been removed
by this check. There is no reason here to retune OT (1,5).

## Decision and next repair targets

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept the scoped code/mathematics diagnosis | Shared-consumer parity and target algebra checked | Replay finite/valid; source hashes match | No unique decomposition of the heldout loss gap | Correct the description of what was trained; design the next repair around the observed restrictions | Simulated-label training is impossible |
| Retain candidate rejection from the completed evaluation | No supported gain over own untrained adapters | Conditional-only and conditional heuristic deficits remain | Small capacity/search and finite scope | Use fresh partitions for any subsequent training; preserve the failed holdout | The research direction is rejected |
| Do not alter the reset or Sinkhorn controls on this evidence | Checked first/second moments preserved | No material reset discrepancy | Earlier higher-order effects remain possible | Keep OT controls fixed while diagnosing the proposal family | Global reset correctness is proved |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No numerical or artifact veto in the replay; previous efficacy/promotion vetoes remain |
| Statistically supported ranking | Only the already-reported pointwise heldout comparisons; no new ranking from this diagnostic |
| Descriptive-only differences | Component weight masses, ESS, local sensitivities, message-fit errors and selected validation changes |
| Default-readiness | Neither learned arm is promoted; this FP64 pilot is not production-default evidence |
| Next evidence needed | Fresh scoped training and untouched evaluation after a discriminating repair, with paired uncertainty and the existing conditional heuristic checks |

The repair priorities follow from what the learner can and cannot control:

1. State the method accurately. Separate training adapters to exported
   controlled-SMC messages from training the replicated KDM learner. If KDM
   learning is the intended experiment, wire that learner to the actual
   downstream moment computation and verify the call chain.
2. Diagnose and repair the late-time flow proposal or provide a justified
   allocation mechanism that can avoid its observed ineffective draws. Include
   an explicit no-guide candidate. Do not relabel beta=1 as LEDH success.
3. Give guide capacity and optimization a target-specific protocol: independent
   simulated datasets, repeated particle streams, adequate training coverage,
   and validation uncertainty. First test whether a shared descent signal is
   reproducible before spending a large budget on the same tiny minibatches.
4. If full controlled-SMC twisting is the intended mechanism, derive its
   intermediate target, initial normalization, potentials and final untwisting
   through the reset before implementing it. Removing the present proposal
   denominator is not that derivation.

These are repair targets, not an executed or approved new comparison. No
unknown score or posterior-moment labels are needed. The current user request
was diagnosis, which is complete.

Engineering ledger: consumer wiring and source match checked; adapter controls
are effective; no runtime edits. Numerical ledger: all replay calls valid and
reset moments agree at roundoff in this scope. Scientific ledger: objective
identity correct, restricted candidates failed prior promotion, causal shares
and efficacy of the next repair unproved.

Post-run red team: a small nonlinear SIR pilot may have poor optimization even
if the proposal family contains a useful candidate. Conversely, a stronger
optimizer cannot repair a component or target construction outside its
controls. A fresh matched experiment that retains the same allocation and
family yet finds reproducible moment gains would weaken the capacity/optimizer
explanation; a component repair with no downstream change would weaken the
allocation explanation. Neither possibility changes the checked call chain.
The weakest evidence is the four-example local sensitivity comparison. No
new independent mathematical-review service was invoked for this diagnosis.

## Execution record and artifacts

The [diagnostic plan](../../ledh-moment-training-failure-diagnosis-20261020.md)
was audited before execution. The first launch failed during diagnostic
imports, before a filter call: importing the supervision module too early
initialized TensorFlow before the memory-growth policy. Moving that diagnostic
import after campaign initialization repaired startup. This is not the
original training failure. The failed log remains `replay-01.log`; 60 process
seconds are charged conservatively because it emitted no wall-time manifest.

The successful command was:

```sh
PYTHONPATH=. TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=0 /home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/plans/artifacts/ledh-moment-training-failure-diagnosis-20261020/replay_diagnostic.py --output docs/plans/artifacts/ledh-moment-training-failure-diagnosis-20261020/replay-02 --budget-seconds 1000 > docs/plans/artifacts/ledh-moment-training-failure-diagnosis-20261020/replay-02.log 2>&1
```

It completed in 359.199960844 process seconds, exit0, on the RTX4080SUPER,
with verified memory growth and allocator peak86,975,488 bytes. The numerical
filter used GPU/XLA and its analytical parameter-score recursion. Eager
post-run feature reductions were diagnostic only. Total investigation charge
is 419.199960844 seconds over two launches, within the 1,200-second cap; parent
campaign remaining budget is 23,826.197895057 seconds. No GPU process remains.

The first attempted analysis preceded completion of the sensitivity file and
stopped. The analysis now requires `result.status=complete` and was rerun after
completion. Flow weight mass is summed directly over flow-labeled particles;
this avoids subtractive roundoff in computing a tiny mass as one minus the
physical mass. Both changes affect reporting only.

Evidence files in this directory:

- `saved-summary.json`: optimizer acceptance, coverage-related runs and frozen
  guide-fit diagnostics, produced by `summarize_saved.py`.
- `replay-02/manifest.json` and `replay-02/result.json`: exact environment,
  command, seeds, eight source hashes, numerical controls, device policy,
  call count and completion. Dataset seeds263021001--004 and263022001--004;
  particle streams277000000--015.
- `replay-02/traces.jsonl`, `consumer-parity.json`, `sensitivities.jsonl`,
  `sensitivity-probes.jsonl` and `training-coverage.json`: executed evidence.
- `replay-summary.json`: descriptive reductions produced by
  `analyze_replay.py`; `analysis.log` records the successful analysis.
- `terminal-validation.json`: script syntax, current runtime source hashes,
  trace/parity counts, optimizer counts, numerical summaries and budget
  reconciliation all pass in a CPU-only stdlib check.

