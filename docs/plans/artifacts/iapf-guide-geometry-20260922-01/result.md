# Exact guides isolate fitted geometry and initial integration

The actual FP64/XLA consumer is correct on these saved linear-model fixtures:
342 evaluations passed. With full exact backward guides and a negligible
positive floor, its value matches the independent R prediction for the actual
initial sample to 5.69e-14, and post-initial ESS/N differs from one by at most
8.89e-16. Saved baseline and R-tail value/score replays match exactly. R and TF
Kalman values agree to 4.45e-14; the independent backward-message identity and
analytic initial-integration identity pass to 1.16e-14 and 1.43e-14.

The fitted coefficients have errors substantially beyond the diagonal family's
minimum KL error. This establishes a fitted-guide quality gap in the tested
consumer. It does not by itself identify an optimizer defect: Equation 15 fits
a different finite-cloud objective, and approximate future guides alter its
recursive target. A terminal-time poor fit cannot be explained by a future
message, making it the next useful isolation case.

## Mathematical interpretation

For full exact Gaussian guides N(c_t,V_t), write
Lambda_t=H'R^-1H+A'(Q+V_(t+1))^-1A and
eta_t=H'R^-1 y_t+A'(Q+V_(t+1))^-1c_(t+1), so V_t=Lambda_t^-1
and c_t=V_t eta_t; omit future terms at T. GJL Proposition 2 and equations 7-8
supply the exact future-likelihood guide. Frozen R `iapf_exact_twists`
independently supplies its Gaussian recursion.

Let a_t=log g_t(c_t)+log Fpsi_(t+1)(c_t)-log N(c_t;c_t,V_t).
The local consumer's value is
logmean_i N(c_1;A X0_i,Q+V_1)+sum_t a_t.
Replacing this sample mean with N(c_1;A m0,A P0 A'+Q+V_1) gives Kalman.
The local initial prior sample is a real difference from the paper's exactly
twisted initial distribution. It explains the remaining oracle error here;
demanding zero finite-sample error would test the wrong initial algorithm.

KL below is KL[N(c*,V*) || N(c,V)]. Its mean component is
0.5 (c-c*)'V^-1(c-c*), and its covariance component is
0.5[tr(V^-1 V*)-d+log(det V/det V*)]. The optimal diagonal comparator uses
c=c* and V=diag(V*). This comparator minimizes this KL, not Equation 15.

## Conditional results (descriptive only)

All rows use the negligible-floor diagnostic. KL summaries pool the four times
within the indicated cases; likelihood errors and heuristic losses are per
case. Heldout rows transfer a guide without refitting and are not a properly
refitted heldout iAPF comparison. Oracles use the data analytically and are not
learned candidate methods. Five rejected adaptive cells remain excluded and
rejected. The diagonal oracle's KL is a family benchmark, not a fitted objective.

| d | observations | guide | cases | median KL | max KL | median absolute log error | heuristic losses |
|---|---|---|---:|---:|---:|---:|---:|
| 2 | fitted | fitted | 8 | 0.1278 | 124 | 8.361 | 6/8 |
| 2 | fitted | KL_diagonal_oracle | 8 | 0.03434 | 0.03456 | 0.08371 | 4/8 |
| 2 | fitted | full_oracle | 8 | 0 | 2.22e-16 | 0.082 | 4/8 |
| 2 | heldout | fitted | 8 | 4.223 | 282.5 | 19.46 | 8/8 |
| 2 | heldout | KL_diagonal_oracle | 8 | 0.03434 | 0.03456 | 0.04171 | 6/8 |
| 2 | heldout | full_oracle | 8 | 0 | 2.22e-16 | 0.01519 | 0/8 |
| 5 | fitted | fitted | 7 | 5.857 | 15.82 | 0.996 | 7/7 |
| 5 | fitted | KL_diagonal_oracle | 7 | 0.1834 | 0.1979 | 0.0651 | 0/7 |
| 5 | fitted | full_oracle | 7 | 0 | 0 | 0.02929 | 0/7 |
| 5 | heldout | fitted | 7 | 6.042 | 91.79 | 1.779 | 7/7 |
| 5 | heldout | KL_diagonal_oracle | 7 | 0.1834 | 0.1979 | 0.07252 | 2/7 |
| 5 | heldout | full_oracle | 7 | 0 | 0 | 0.06722 | 0/7 |
| 10 | fitted | fitted | 4 | 6.279 | 10.1 | 1.332 | 4/4 |
| 10 | fitted | KL_diagonal_oracle | 4 | 0.4404 | 0.4776 | 0.09898 | 2/4 |
| 10 | fitted | full_oracle | 4 | 0 | 0 | 0.05628 | 0/4 |
| 10 | heldout | fitted | 4 | 9.801 | 77.05 | 25.01 | 4/4 |
| 10 | heldout | KL_diagonal_oracle | 4 | 0.4404 | 0.4776 | 0.0926 | 0/4 |
| 10 | heldout | full_oracle | 4 | 0 | 0 | 0.02487 | 0/4 |

Heuristic losses are single-draw comparisons with matched-count bootstrap,
constant-guide and one-step proposal estimators, alongside the Kalman oracle.
They veto promotion under the plan; they do not establish a stochastic ranking.
Even the full oracle loses on four of eight fitted d2 cases because its initial
integration is still random. The apparent accuracy of an oracle does not
validate a deployable fitted method.

## Decision and inference status

| Decision | Primary criterion | Vetoes | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Close consumer/oracle diagnosis | Independent identities and exact replay pass | No validity veto | Only these small linear FP64 fixtures | Capture real fitting clouds and targets | Broad correctness, TF32 clearance |
| Reject fitted-guide promotion | Large errors beyond diagonal minimum | Conditional heuristic losses | Different fitting objective; recursive contamination | Evaluate init/final objective and shape on captured inputs | Algorithm-wide rejection |
| Record initial-law gap | Exact finite-sample prediction verified | No invalidity | Paper initial law still absent locally | Keep this distinction explicit in future repairs | Paper zero variance or full replication |

| Inference status | Finding |
|---|---|
| Hard veto screen | Finite/CDF/source/oracle checks pass; five previous adaptive failures remain rejected |
| Statistically supported ranking | None |
| Descriptive-only differences | KL, likelihood errors, per-case heuristic losses |
| Default readiness | No candidate/default change; strict TF32 veto remains |
| Next evidence needed | Causal fit-input isolation, then fresh multi-seed downstream validation of any repair |

Terminal skeptical review: PASS for mechanism isolation. The strongest
alternative explanation of bad fitted geometry is a contaminated recursive
target rather than a broken consumer or diagonal restriction. Capture terminal
and earlier fitting targets to discriminate it. The weakest evidence is the
small survivor-only case set and single-draw heuristic comparison. A failed
independent message identity, changed draw/source hash, or different matched
model would overturn this interpretation. No R-paper replication, marginal
model-score, canonical LEDH, HMC, production or TF32 promotion follows.

Commands, environments, seeds, source snapshots and wall times are preserved
in the two launch records and summaries. CPU deliberately hid GPU; GPU used
RTX 4080 SUPER with verified growth, FP64, XLA and TF32 disabled. Independent
R source remained unchanged. `verification.json` and `manifest.json` preserve
terminal checks and all local output hashes. Campaign authorization continues.
