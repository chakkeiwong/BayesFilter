# q20 reverse-KL stopping and convergence audit

The inspected map learned substantially, but RKL convergence was not established.
Its last comparison between distinct checkpoints still reported improvement.
The scheduler stopped updating it because it qualified for an HMC trial. The
later zero incremental loss compared that same map with itself after checkpoint
import; it is not evidence of a plateau.

## Scope and audit contract

This is a read-only audit of saved training data and the source that produced
`direct-w16-lr0.0005-r0-beta1-u512`, the frozen map used in the September 22
Gaussian-step canary and 1,000-point score diagnostic. The question is whether
the saved record supports RKL convergence, and what explains the apparent
conflict with poor Gaussian whitening. There is no new training, target
evaluation, HMC run, or GPU initialization.

The comparators are the saved initialization and preceding distinct trained
checkpoint. Success for this audit means tracing the actual stopping predicate,
optimizer history, and assessment identities. A source/hash mismatch invalidates
the trace. Same-checkpoint loss differences cannot support a convergence claim.
Training losses, clipping, and score residuals explain possible failure modes;
they do not establish posterior validity or rank viable methods. No new
convergence or whitening threshold is introduced. The 128-record display window
is a convenience for summarizing existing history, with no decision threshold.

Skeptical audit: use the archived execution source rather than the dirty current
tree; distinguish per-scope counts from total Adam iterations; distinguish trial
nomination from convergence; inspect pre-import assessments rather than treating
reassessment as new optimization; and retain the unknown target normalizer in
the loss interpretation. These controls address the relevant wrong-baseline,
proxy-metric, stale-context, and environment risks. This saved-data audit passes
without needing new scientific computation. Stop if its identity checks fail.

## What the history establishes

The checkpoint contains **1,024 consecutive accepted optimizer updates**, all at
beta one, with 1,024 distinct random seeds. Adam's stored iteration agrees. The
first 512 updates preceded the checked backend/scope migration; another 512
followed it. Migration preserved the map, Adam state, RNG, and history but reset
`level_updates` to zero. Thus `u512` is a current-scope count, not lifetime
training effort. The migration receipt reports preserved numerical state and
value/score parity on its checked rows (maximum value error zero, score error
about 7.11e-15); it does not claim exact numerical continuation for all inputs.

The map used two dense IAF stages, two width-16 tanh hidden layers per stage,
batch size 32, Adam learning rate 0.0005, and gradient clipping at norm 10.
These are observed settings, not newly calibrated recommendations.

| Total optimizer steps | Mean recorded training-batch loss |
| --- | ---: |
| 1–128 | 292.7300 |
| 385–512 | 49.1659 |
| 513–640 | 47.4862 |
| 641–768 | 46.7703 |
| 769–896 | 46.3616 |
| 897–1024 | 45.8839 |

These are descriptive averages over fresh batches and changing parameters, not
a formal convergence test. More directly, the original assessment at current
scope update 512 compared against update 128 (total steps 1,024 versus 640).
On 768 paired held-out base draws, loss fell **1.1748299675**, with descriptive
normal-interval half-width **0.3522948182**. It explicitly recorded
`continuing_improvement=true`, `plateau_observed=false`, and
`status=hmc_trial_nominee`. The adaptively reused bank does not provide nominal
confidence coverage, but it plainly was not reporting a plateau.

The later imported checkpoint had identical `previous` and current session
hashes, `f2bf6b38c63b4f0f4cd040aee3ecc364aa1ad7031c9faeb21be3ec0ef901a985`.
Reassessment changed no optimizer state. It reused 2,304 cached rows across
baseline/previous/current evaluations and evaluated no new rows. Consequently
its incremental change and standard error were exactly zero. Its
`continuing_improvement=false` and incremented explanatory plateau counter do
not measure what further training would do. `plateau_observed` remained false.
The baseline decrease of 319.2572 remained valid as a baseline comparison.

Gradient clipping occurred in 1,021 of 1,024 updates, including 126 of the last
128. The last-window median unclipped minibatch gradient norm was 60.0070.
Clipping is therefore active and merits examination; a noisy minibatch gradient
need not vanish at an optimum, so these norms alone prove neither failure nor
nonconvergence and do not establish clipping as the cause.

## Why lower RKL loss did not establish whitening

Let `theta=T(z)`, with standard-normal base density `phi`, normalized target
`pi_theta=tilde_pi_theta/Z`, and invertible differentiable T. The pushforward
density satisfies `q_T(T(z))=phi(z)/abs(det DT(z))`, while the transformed target
is `pi_z(z)=pi_theta(T(z))*abs(det DT(z))`. Substitution gives

\[
\begin{aligned}
D_{\mathrm{KL}}(q_T\|\pi_\theta)
&=\mathbb E_\phi[\log\phi(z)-\log|\det DT(z)|
                 -\log\widetilde\pi_\theta(T(z))]+\log Z\\
&=D_{\mathrm{KL}}(\phi\|\pi_z)
 =\mathcal L(T)+\mathbb E_\phi\log\phi(z)+\log Z,\\
\mathcal L(T)
&=\mathbb E_\phi[-\log\widetilde\pi_\theta(T(z))
                 -\log|\det DT(z)|].
\end{aligned}
\]

The trainer computes a minibatch estimate of the last line. Lower loss is the
correct direction for RKL at a fixed target, assuming the supplied target score
is the derivative of its value. The raw loss includes an unknown additive
normalization offset: a loss near 46 is not a measured KL of 46, nor evidence of
a KL near zero. A decrease from a poor initialization establishes learning, not
closeness to the posterior. No independent normalizer or absolute KL estimate
was established in this audit.

At zero KL the two densities agree almost everywhere; with the relevant smooth
positive densities their scores agree as well. Finite-family optimization
stationarity is a weaker condition. Here even that stationarity was not
established. RKL training also does not directly minimize the pointwise score
residual or the curvature that determines stable HMC steps.

The separate 1,000-point diagnostic found median
`||grad log pi_z(z)+z||=7.6008`, with the residual exceeding `||z||` at 98.3% of
the tested iid standard-normal points. This is poor agreement with the
standard-normal score over the tested proposal region. Those are not posterior
draws, and reuse of the same score implementation is not independent derivative
validation. Both training and the diagnostic consume supplied target scores;
a score/value inconsistency remains an alternative explanation to investigate.

## Decision and next justified work

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| RKL convergence claim | Unsupported; the actual last distinct-checkpoint screen reports continuing improvement | Reassessment self-comparison cannot count as plateau evidence | How much more improvement is attainable | Preserve distinct optimizer checkpoints when reporting increments; label identical-map reassessment explicitly | That a specified extra update count guarantees convergence |
| Current map as a Gaussian whitening map | Poor score agreement in the saved proposal-region diagnostic | Gaussian score assumption fails on the tested points; numeric rows are valid | Score correctness, posterior-region geometry, capacity and optimizer effects | Check score/value derivatives, then continue a bounded training ladder with occasional paired checks and downstream geometry/HMC checks | Failure of NeuTra as a method or posterior correctness |
| Existing training implementation | It computed and reduced the intended RKL-form objective, conditional on supplied-score correctness | No rejected/nonfinite updates in this history | Independent score accuracy and optimization adequacy | Examine clipping, learning rate, batch noise and architecture only through discriminating checks | That reduced validation itself caused the failure |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No stored invalid update; self-comparison invalidates that increment as convergence evidence |
| Statistically supported ranking | None; this audit does not compare or rank training candidates |
| Descriptive-only differences | Loss windows, reused-bank intervals, gradient norms, clipping frequencies, and prior score residual summaries |
| Default-readiness | No new training, map, step-size, or posterior default is established |
| Next evidence needed | Distinct-checkpoint progress and independent derivative checks, followed by target-specific downstream validation |

The directly established workflow problem is stopping updates at trial
nomination while later interpreting the map as if whitening had been verified.
A bounded HMC trial can legitimately use an imperfect map; nomination must not
be described as convergence. The exact contribution of training duration,
architecture, learning rate, clipping, coverage, or a score defect remains
unresolved. The strongest alternative explanation is incorrect supplied
derivatives; the weakest evidence for optimization stationarity is the imported
self-comparison. No production code or campaign state was changed by this audit.

## Reproduction and source anchors

Run `python docs/plans/artifacts/q20-rkl-training-audit-2026-09-22/audit_saved_training.py`.
It uses only the Python standard library, hides GPUs deliberately, and writes a
new `summary.json` exclusively. It preserves input checksums, hashes of the
archived source files, update/count checks, prior and reassessed decisions, and
loss windows. Source hashes must match the frozen export. The audited repository
HEAD is `8f992b205e9a4b8a861db4064cbcedb76af52f1f`; the dirty current implementation
is not the source authority for this historical map.

- Archived execution source: [source-r2.tar.gz](artifacts/q20-1000-point-score-residual-2026-09-22/source-r2.tar.gz).
- In that archive, `neutra_training_protocol.py:287` defines trial nomination;
  `q20_production_training.py:75` compares the saved maps and line 211 skips
  further nominee updates; `q20_training_resume.py:118` clears prior assessments;
  `q20_checkpoint_migration.py:83` resets the scope count while retaining history;
  `tempered_transport_ensemble_tf.py:1138` implements the RKL update and line 134
  attaches the supplied target score.
- [September 20 admission repair](bayesfilter-ssl-lstm-q20-training-admission-repair-2026-09-20.md)
  explicitly made plateau explanatory and stopped training eligible maps.
- [Saved-data audit summary](artifacts/q20-rkl-training-audit-2026-09-22/summary.json).
- [1,000-point residual result](bayesfilter-q20-1000-point-score-residual-result-2026-09-22.md).

Follow-up: the [complete cohort and derivative investigation](bayesfilter-q20-training-gap-results-2026-09-22.md)
confirmed unfinished training across all twelve direct maps, measured terminal
scale saturation, and independently checked physical/latent scores by central
differences at twelve saved points. Those local derivative checks passed; they
supersede the unchecked-score status above only at their recorded scope.
