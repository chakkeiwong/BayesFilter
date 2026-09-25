# Paper-model transfer of the covariance and floor diagnosis

Status: COMPLETE.119 terminal checks,48 preflight recursions and400 final evaluations pass.
[Result](artifacts/iapf-paper-score-transfer-20260922-01/result.md): the .01 floor fails,
diagonal score/tail8 remains viable; negligible-tail8 behavior fails at d80.
Skeptical review PASS before implementation and execution.
Authorized renewed 48 CPU-hour/48 GPU-hour campaign; starting balance 45.832008
CPU /47.745145 GPU hours. Explicit independent R reference phase, with GPUs
hidden and single-threaded BLAS. Root:
`docs/plans/artifacts/iapf-paper-score-transfer-20260922-01/`.
At most three CPU worker hours, fourteen subprocess attempts of at most 600s,
including repairs, at most two concurrent workers. No GPU work in this phase.

## Question and source boundary

Does the failure from a fixed .01 peak-relative floor also occur in the paper's
actual linear model? Does a positive floor that accounts for dimension preserve
the Gaussian fit and improve its viability? This tests a mechanism and an
explicit reconstructed algorithm. Coordinate-score regression changes Eq15;
full covariance also changes the paper's diagonal restriction. Neither becomes
original-author code or full paper replication by matching the study settings.

Source: local Guarniero, Johansen and Lee (2017), §5.1, Eq15–16, Algorithm5 and
§5.2, `.localresources/papers/guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter.txt`,
lines907–950 and1021–1110, inspected for this phase. Eq16 specifies a positive
function c(N,m,Sigma) but not its numerical formula. §5.2 gives m=0,
Sigma=B=C=D=I, A_ij=.42^(abs(i-j)+1), T=100, dimensions5,10,20,40,80,
iAPF N0=1000, BPF N=10000 and FA-APF N=5000. Published SDs of Zhat/Z are
.09,.14,.19,.23,.35 for 1000 replicates of the complete iAPF. They are context,
not a pass threshold for our one-pass four-replicate diagnostic.

Frozen reference call chain: `reference_iapf_paper.R::iapf_paper_data` supplies
X1~N(0,I) and the first observation; `iapf_linear_model` preserves this timing;
`iapf_apf` implements the twisted initial proposal, retained weights and ESS
resampling with kappa=.5. The exact full backward guide must give the Kalman
likelihood even for finite N under this reference initial law. This differs
from the local TensorFlow X0 finite-integration consumer. Inspect and exercise
this endpoint, not merely a standalone fitting function.

## Frozen design and evidence contract

Two new data sets per dimension (data seed `951000 + 10*d + j`, j=1,2), one
bootstrap pilot with N1000 (`952000 + 10*d + j`), one backward fit per learned
arm, and four independent final streams (`953000 + 100*d + 10*j + r`, r=1..4).
Preserve all seeds and data. Re-seed each final method to the same pair label;
different mixtures can consume randomness differently, so this is pairing by
data/seed, not a promise of identical random variates throughout.

Learned arms: score regression with {diagonal,full} covariance crossed with
{positive .01,positive tail8} floor, plus the existing diagonal log-quadratic
QR/tail8 reference. Use the same bootstrap cloud for all five. No density
optimizer or adaptive controller is run; their separate issues are not erased.
The .01 arm preserves the earlier realized scalar .009999999776482582.

For a fitted Gaussian density phi, the tail8 rule is
`c = phi(m) * exp(-qchisq(N^-8,df=d,lower.tail=FALSE)/2)`.
Under X~phi, P(phi(X)<c)=N^-8; for N draws, the union bound is N^-7. This gives
a dimension-aware Gaussian contour justification. It does not bound floor
probability under the transition distribution or guarantee nonlinear robustness.
The exponent8 is a previously tested reconstruction hypothesis from
`reference_iapf_author_choices.R`, not an author setting or a tuned optimum.
For full covariance normalize the peak using log(det(V)). No ridge/clipping.
No positive floor may be silently replaced by zero on rejection.

Constructed heuristic adversaries, conditional on each dimension/data set:
bootstrap N10000 (uninformed reference at the paper count), FA-APF N5000
(analytically adapted classical comparator at the paper count), current-observation
guide N1000 (cheap one-step lookahead at equal particles), and full backward
oracle N1000 (exact Gaussian conditional authority). Record each count and
runtime; unequal counts are deliberate paper settings, not equal-work claims.

Primary quantities are log error and relative likelihood error Zhat/Z-1.
Preserve ratio mean, SD and relative RMSE within each data set. A near-zero
sample SD when every likelihood ratio is near zero is catastrophic error, not
replication; RMSE and mean prevent that false conclusion. Four final streams
are descriptive only. No confidence interval or ranking is claimed.
Observed loss to a cheap heuristic vetoes promotion; it does not stop this
mechanism study. The exact oracle is reported as a mathematical ceiling and
its unavoidable gap for approximate guides remains explicit.

Numerical/engineering continuation vetoes and repair triggers:

- New R core reproduces all 48 phase13 recursive coefficient sets/validity
  flags within1e-8 before paper runs. Recheck prior artifact hashes.
- Analytic target score passes an independent finite-difference check <=1e-6;
  rank-deficient clouds and nonpositive precision are rejected.
- Full/negligible-floor score fit recovers paper-model exact Gaussian backward
  coefficients <=1e-8 in every data set. Its final likelihood matches Kalman
  <=1e-8. This is a separate limit control, not a positive-floor candidate.
- Oracle/Kalman equality holds for the actual R APF endpoint; source hashes,
  seeds, finite log values, required diagnostics and artifact counts are valid.

Candidate precision/rank failure, QR rejection, or likelihood underperformance
is a retained candidate failure. Do not omit failed rows or restart with easier
data. Purely explanatory diagnostics: fit error from oracle, score residual,
whitening/skew/precision margins, floor probabilities, ESS and resampling count.
The Class C floor hypothesis is evaluated for non-harm against the Gaussian
limit: coefficient/value perturbation and observed floor use are preserved,
without selecting the exponent to optimize likelihood error. Large perturbation
rejects its claimed Gaussian-limit behavior, not necessarily all uses of a floor.

## Assumption audit and skeptical review

| Choice | Provenance/status | Justification and risk | Early check |
|---|---|---|---|
| Paper matrices/timing/ESS | Published study / baseline | Removes known local model mismatch | Exact oracle equality through APF |
| Score objective | Explicit extension | Removes density-energy escape; needs analytic coordinate score | R/TF saved parity and finite differences |
| Full vs diagonal | Diagnostic controls | Full may exploit Gaussian structure unavailable generally | Report restriction and oracle gap |
| .01 vs tail8 floor | Hypotheses | Fixed .01 can dominate; tail8 only has a Gaussian contour argument | Floor probability and coefficient perturbation |
| No ridge/precision clipping | Derived full-rank regression | Preserves exact solution, may reject nonglobal log-concavity | Relative eigenvalue guards |
| One backward pass | Mechanism isolation | Avoids adaptive stopping confound; not Algorithm4 | Mark output method explicitly |
| Two data sets/four streams | Bounded diagnostic convenience | Too small for Table1 replication or population ranking | Report means/RMSE and individual failures |
| R/CPU, frozen reference | User-authorized independent reference | Does not establish TF production or GPU readiness | Freeze hashes; record CPU-only environment |

Review PASS: the plan uses the actual study model and finite initial law, keeps
method/floor hypotheses explicit, preserves all failed candidates, uses the
paper-count classical comparators and exact oracle, and prevents near-zero
ratios from masquerading as low variance. The 1000-replicate, adaptive and
unknown-author-choice gaps remain. The new core is checked before expensive
work. Numerical failure triggers repair; candidate failure leads to the next
method diagnosis. No arbitrary promotion threshold or default change is made.

## Execution and stop rules

Create a new R diagnostic module without editing the three frozen reference
files. A Python standard-library driver preserves snapshots, SHA256, git commit,
environment, commands, wall time and per-case outputs. Run preflight, then two
case workers concurrently at most; every timeout/failure consumes budget. Each
case output directory is unique. Stop new launches if required input/source
validity fails or the remaining budget cannot reserve the next bounded attempt.
Localized repair under the same design may use the two spare case attempts.

```text
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/diagnose_iapf_paper_score_transfer.py --mode preflight
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/diagnose_iapf_paper_score_transfer.py --mode cases
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/diagnose_iapf_paper_score_transfer.py --mode results
```

Pre-mortem: exact full recovery could disguise poor diagonal filtering; small
empirical SD could disguise rare-event collapse; an R-only improvement could be
mistaken for a repaired TF route. The declared controls address each. Post-run
review must state whether the paper-model mechanism agrees with the local one,
whether tail8 is numerically negligible where claimed, and whether the diagonal
score candidate warrants an adaptive, fresh-replicate comparison. If it fails,
investigate the remaining diagonal projection/objective mismatch; do not spend
the budget repeating a known failed floor. Update the master and checkpoint.
