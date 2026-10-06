# Why the small-mixture NeuTra continuation did not qualify

This is a read-only diagnosis of the completed October 2 campaign, prompted by
the owner's question about the failures and related literature. No training,
sampling, tuning, or new campaign was launched. The execution record remains
`bayesfilter-neutra-rare-region-results-2026-10-02.md`; all numbers below are
either extracted from its preserved artifacts or explicitly derived.

## Question and skeptical audit

The question is why weighted exploration of two simple targets did not produce
a global IAF suitable for the declared NeuTra HMC checks. The exact normalized
mixture is the reference, and saved forward/RKL checkpoints are the within-fit
comparators. We distinguish probability estimation, learned-map geometry, and
finite-run HMC qualification. None substitutes for another.

The audit rejected three possible shortcuts in the explanation: treating every
failed screen as algorithmic impossibility; blaming ordinary HMC on the
3.09e-7 event (its diagnostics do not require that event); and treating finite
training or declining loss as convergence. Checkpoint metrics are descriptive,
not statistically supported rankings. Missing oracle transport controls and
uncalibrated training stopping rules remain limitations. This inspection can
identify observed failures and mathematical mechanisms, but cannot isolate
every causal contribution without another discriminating experiment.

## What worked and what failed

Umbrella/EMUS, bridge importance sampling, and fixed-level splitting passed the
declared probability and moment feasibility screens on both targets. This is
eight-replication screen evidence, not a ranking or an exactness guarantee for
finite runs. The mixture of local maps passed ordinary mode/moment checks but
had no narrow-event observations after its longer repair.

The narrow event has probability 3.093565358755601e-7. Even 65,536 independent
posterior draws give only 0.0203 expected observations, by N*p. Its absence
does not establish a defect in the MH correction. Directed estimators solve a
different problem from ordinary posterior draws.

None of twelve eligible global IAF fits passed HMC qualification. Among 41
verified kernel members, 14 had a terminal warm-up health failure because a
chain did not move; their states and target/log-acceptance values were finite.
Four reached retained sampling and exhausted 10,000 draws per chain.

The four retained failures were not identical. In the unwarped importance
seed-37 final-RKL member, all declared R-hat checks passed, but the mode
indicator had bulk ESS 347.62 and MCSE/SD 0.0536, failing the configured ESS
400 and MCSE/SD 0.03 requirements. Two other retained members had no broad-
valley observations at all. The broad event has probability 0.001349898,
distinct from the narrow event above. These facts support inadequate global
sampling evidence at the tested settings and budget, not an impossibility
claim about HMC.

Source: `artifacts/neutra-rare-region-2026-10-02/terminal-audit-r1.json` and
`campaign-r1/attempts/hmc-*/checkpoint-*/member-*/posterior.json` beneath that
artifact root. The sequential requirements and quantity-specific failures are
serialized in each posterior report; no current-code reinterpretation was
needed to recover them.

## Loss reduction and geometric repair are different

`bayesfilter/testing/neutra_rare_region_phases.py::fit_bank` selects a width/LR
arm after 256 updates, continues to 1,024 forward updates, then runs pure RKL
to checkpoints at 256 and 1,024 further updates. It preserves earlier maps,
but has no plateau/convergence stopping criterion and no forward-loss retention
term during RKL. Those update counts are campaign budget choices, not
demonstrated sufficient training. The selected forward continuation reported
zero clipping in all sixteen fits; recurrent clipping is not the observed
explanation for those particular updates.

For this normalized two-dimensional target, the saved Gaussian-base diagnostic
`log_ratio_mean` gives the Monte Carlo estimate

    KL(q || pi) estimate = -log(2*pi) - log_ratio_mean.

This follows from q(T(z))=phi_2(z)/|det J_T(z)| and
log(phi_2(z))=-log(2*pi)-||z||^2/2. Recomputing this scalar from existing JSON
requires no new samples. A concrete saved trajectory is:

| Unwarped importance teacher, seed 37 | Estimated RKL | Median random-base score residual | Maximum directed valley residual |
|---|---:|---:|---:|
| Forward checkpoint | 0.803326 | 0.962562 | 62.8383 |
| RKL 256 | 0.406533 | 0.818192 | 135.8714 |
| RKL 1,024 | 0.171176 | 0.485389 | 327.9104 |

The same qualitative pattern appears in warped splitting seed 11: estimated
RKL 0.612385 to 0.162468, directed maximum 66.4088 to 353.7574. These finite
probe summaries demonstrate that the recorded loss and directed diagnostics
can move in opposite directions; they do not estimate a uniform score bound
or establish a statistically supported method ranking.

Correct stratification does not solve this objective mismatch. If stratum k
has probability b_k, the forward objective is

    L_FKL = -sum_k b_k E[log q_theta(X) | X in B_k].

Oversampling a stratum while retaining its correct weight reduces sampling
noise. It does not remove the small b_k coefficient, nor does log-density loss
directly bound derivatives of the learned map. A large derivative of the loss
could still make that stratum important; small mass alone is not a bound on
its gradient contribution. Any deliberate reweighting or derivative penalty
would be a stated change to the training objective.

## Mode loss can look like excellent local whitening

Write pi_1(x)=(1/3)phi(x+5)+(2/3)phi(x-5). The invertible translation
T(z)=(z_1+5,z_2) generates only the right Gaussian component. Its exact
pullback satisfies

    pi_z(z)/phi_2(z) = 2/3 + (1/3) exp(-10*z_1-50),
    [grad log pi_z(z)+z]_1 = -10 exp(-10*z_1-50)/(2+exp(-10*z_1-50)),
    [grad log pi_z(z)+z]_2 = 0.

At ordinary standard-Gaussian points the residual is tiny, although one-third
of the target mass lies around latent z_1=-10. Thus random Gaussian-base
residuals cannot certify mode coverage. For well-separated components this
map has RKL approximately -log(2/3)=0.405465, a finite local solution rather
than the zero-KL global solution. This is the familiar reverse-KL mode-loss
mechanism, not proof that the optimizer must collapse.

Several saved fits have estimated RKL near this value. For example, unwarped
umbrella seed 11 ends at 0.409091 with median residual 0.086844. This is
consistent with the translated-component explanation; it does not prove that
the learned map equals the translation or that every failed map lost a mode.

## Low dimension does not imply mild transport derivatives

There is an exact transport for this benchmark: T_1(z)=F_1^{-1}(Phi(z)), with
T_2(z)=z_2 in the unwarped case. The warped case adds
0.1*(T_1(z)^2-26) to the second coordinate. Both targets have positive smooth
densities; this is not a disconnected-support impossibility.

Differentiating F_1(T_1(z))=Phi(z) gives T_1'=phi(z)/pi_1(T_1(z)). At physical
x_1=0 it is about 244,565. For the exact map the target-score and log-Jacobian
derivative terms cancel perfectly. An approximate map must approximate that
cancellation to give Gaussian latent geometry. This particular marginal
transport illustrates the difficulty; it is not a lower bound on every
two-dimensional map or a proof that the canonical IAF lacks sufficient capacity.
The full derivation and a small-KL/large-score counterexample are in
`../chapters/ch26e_rare_regions_transport.tex`.

## Related literature directly addresses these failures

Gabrié, Rotskoff and Vanden-Eijnden (2022), *Adaptive Monte Carlo augmented with
normalizing flows*, Appendix G.1, study a two-dimensional Gaussian mixture with
unit component covariance, ten-standard-deviation separation, and a 2:1 mode
weight ratio: the same mixture up to coordinates. Local-only chains initialized
in both modes fail to recover the weights; the full method initialized in only
one mode never discovers the other; the full method initialized in both modes
achieves accurate, efficient sampling. The reported independent-proposal
acceptance is 80–85%. Their demonstration concerns corrected global/local MCMC,
not a globally Gaussian pullback or NeuTra HMC qualification.

Section III.B, equations (8)–(9), explains why reverse-KL learning depends on
the current map discovering relevant target regions. Section VI.A discusses
mode collapse and underestimated tails. Section IV.E describes mixtures of
separately trained maps with learned mixture weights. Appendix G further
discusses finite-batch loss of a less-probable mode during concurrent sampling
and training, and using separate generators to protect those modes.

Inspected local paper: `.localresources/fab-coverage-followup-20260928/`
`gabrie-adaptive-flows.layout.txt`, lines 216–244, 405–431, 810–834,
1701–1728, and 2032–2045. Original code inspected:
`.localresources/flonaco-author-20260929/upstream/flonaco/sampling.py`,
`run_metropolis` (151–167) and `run_metromalangevin` (207–249), which implement
global independence MH and its composition with a corrected local move.
Paper: https://arxiv.org/abs/2105.12603.

Hoffman et al. (2019), *NeuTra-lizing Bad Geometry in Hamiltonian Monte Carlo
Using Neural Transport*, Section 5, explicitly identifies the possibility
that an insufficient map can slow tail mixing. Section 4.1.1 describes 5,000
training steps with batch 4,096 and a learning-rate schedule on its targets.
Those numbers do not establish an appropriate budget for our mixture; they
also do not justify calling our fixed smaller ladder converged. Inspected
local source: `.localresources/q20-flow-training-literature-20260923/papers/`
`hoffman-2019-neutra.txt`, lines 337–360 and 553–566.
Paper: https://arxiv.org/abs/1903.03704.

## Remaining experimental-design gaps

The exact transport was derived but not run through this campaign's downstream
procedure. That missing positive control prevents complete separation of
learned-map problems from tuner, diagnostic, initialization, and budget
limitations. Ordinary HMC does not need to measure the narrow event to pass;
the actual broad-valley and mode requirements still need a known-correct
control at the declared budget.

The single preselected replication used as each teacher also differs from the
eight-run aggregate screen. For example, the umbrella teachers assign about
0.76 to the right mode instead of the true 0.667. Correct stratification
preserves this teacher error. Passing the aggregate estimator screen does not
certify the chosen individual training bank. This cannot explain every fit:
the splitting banks' mode weights are much closer to the reference and their
global fits also failed qualification.

The next discriminating work would validate an exact-transport downstream
control, measure mode/valley behavior before and after RKL, and assess
target-specific training convergence and capacity. A subsequent objective or
architecture change needs an explicit hypothesis and evidence contract. More
undifferentiated end-to-end retries would not isolate these causes.

## Decision and inference status

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Preserve the three viable probability estimators | Declared feasibility screens pass | No reported correction/finite-weight veto | Finite runs and imperfect teacher banks | Verify the actual bank used by training | No method ranking or general-dimensional guarantee |
| Do not promote the global learned maps | 0/12 HMC fits qualify | Immobility and global information checks fail for tested members | Objective, capacity, convergence, and finite tuning budgets | Exact-transport control, then isolate training stages | No impossibility of IAF or NeuTra |
| Keep the research direction open | Literature succeeds on the same mixture with another sampling/training composition | No research-direction continuation veto established | Transfer from corrected global/local MCMC to NeuTra | Use the source method as a mechanistic reference | No claim that the full current recipe reproduces that paper |

| Inference status | Finding |
|---|---|
| Hard veto screen | Some chains immobile; no global fit qualified; clipping and nonfinite updates are not the recorded forward-continuation failure |
| Statistically supported ranking | None |
| Descriptive-only differences | Saved RKL, residuals, ESS, MCSE ratios, and teacher weights |
| Default-readiness | Failed for these learned maps |
| Next evidence needed | Known-correct downstream control and convergence/coverage evidence for each training stage |

Post-inspection red team: the strongest alternative explanation for part of
the downstream rejection is insufficient kernel search or retained length,
rather than loss of all useful map structure. One retained member passed all
R-hat checks but lacked the required information. Conversely, increasing
sample counts cannot by itself establish that training learned the desired
global geometry. An exact-control failure or successful adequately trained
canonical map would change the causal diagnosis. Current data do not quantify
how much of the failure each mechanism explains.
