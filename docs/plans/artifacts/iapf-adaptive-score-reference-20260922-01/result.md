# Adaptive R reference result

The diagonal score method with the later doubling convention failed the
predeclared cheap-heuristic screen at dimension40: observed relative RMSE was
0.340 versus0.317 for the fully adapted APF. Ten replicates do not establish a
population ranking. Every other learned arm cleared the observed heuristic
screen at every tested dimension. All200 complete learner runs and200 heuristic
runs completed with finite likelihoods, and729 terminal verification checks
passed. No fit rejection or controller cap occurred.

The adaptive clouds removed the diagnosed floor distortion. At dimension80,
the first score fit's maximum coefficient difference from the same-cloud
negligible-floor fit ranged from1.25e-6 to0.188 across seeds. The minimum target
Gaussian weight reached0.0681. In every run, from the second fit onward, that
coefficient difference was zero at the recorded floating-point precision and
the target Gaussian weights were one. This answers the local mechanism
question; it does not prove that the floor is harmless on arbitrary clouds.

## Fresh final likelihoods

The table reports relative RMSE of Zhat/Z about one, using ten independent
complete learner replicates on one fresh observation set per dimension. All
continuous differences are descriptive. The bootstrap and current-observation
controls are also preserved in `conditional-summary.json`; the FA-APF is the
strongest cheap comparator in these observed samples. Final particle counts
differ between controller conventions, so the table is not an equal-cost ranking.

| Dimension | Score, later doubling | QR, later doubling | Score, early doubling | QR, early doubling | FA-APF N5000 |
|---|---:|---:|---:|---:|---:|
|5|0.0793|0.0373|0.0429|0.0252|0.0990|
|10|0.0559|0.0739|0.0603|0.0479|0.1198|
|20|0.1500|0.1207|0.1234|0.0639|0.2646|
|40|0.3400|0.1198|0.1444|0.0630|0.3167|
|80|0.4926|0.1873|0.2417|0.2263|1.6211|

The score/later-doubling ratio means are0.982,1.012,1.007,1.145,1.111 and
sample SDs are0.081,0.058,0.158,0.324,0.506. The QR/later-doubling means are
0.998,1.018,1.013,0.991,0.967 and SDs0.039,0.076,0.127,0.126,0.194. Mean,
RMSE and log-error MSE are retained alongside SD to prevent a nearly-zero
estimator from being mistaken for low-variance success.

With later doubling, score mean final N was1000 in dimensions5–40 and1100 at80;
QR used1000 in5–40 and2000 at80. Early doubling used2000 throughout for score,
and mean N2000,2000,1800,1900,2000 for QR. The paper reports mean final N1000,
1000,1000,1033,1142. These counts constrain the reconstruction but cannot identify
the authors' controller independently of their unknown fitter. See
`gaussian-mechanism.md` for the fixed-guide nonmonotonicity calculation.

The full Gaussian APF control agreed with Kalman within1.82e-12 in all50
independent runs. Forty-eight R/TensorFlow recursion fixtures still passed after
adding diagnostics; the target-state-score finite-difference error was2.71e-10,
and rank/nonpositive-precision rejection controls passed. The frozen R references
were unchanged. Executable wiring checks verified that the actual controller
made one fresh final filter call after stopping; its history likelihood was
never substituted for the reported result.

## Mathematical interpretation and remaining gap

For a quadratic log backward target, the affine state-score regression recovers
the full Gaussian coefficients on any full-rank cloud. Projecting its covariance
to a diagonal then defines a deterministic Gaussian approximation. Once the
floor no longer changes the fit, more iterations cannot improve that
approximation by moving the cloud. QR fits depend on cloud location because
they omit quadratic cross terms when fitting log values. These are different
diagonal approximations, which explains why numerical correctness of both
fitters does not imply equal importance-sampling accuracy. The derivation is
preserved in `gaussian-mechanism.md`; it does not claim that either approximation
minimizes likelihood-estimator variance.

Neither method implements the printed Eq15 density least-squares optimizer.
The R reference still lacks the authors' optimizer/floor/initialization identity,
original observation sets, and1000-replicate confirmation. The paper-table
numbers describe different unknown data sets; numerical closeness alone cannot
close these gaps. This result supports a functioning diagnostic reference,
not full paper replication, nonlinear validity, a model-parameter score, or
TensorFlow/GPU/default readiness.

## Decisions and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Close floor/cloud mechanism phase | Same-cloud effect removed from fit2 in all d80 runs | No numerical/harness veto | Unseen remote clouds | Preserve diagnostics in larger ladder | Universal floor safety |
| Retain both reconstructed fitters as diagnostics |200/200 learners complete | Score/later doubling fails d40 heuristic screen | Ten-replicate Monte Carlo variation | Larger frozen replication, no promotion | QR superiority or score-direction rejection |
| Keep controller conventions explicit | Fresh final runs and count histories checked | Cost comparability prevents pooled ranking | Unknown original early-index treatment | Scale later convention as declared reconstruction; retain early convention evidence | Author implementation identity |

| Inference item | Status |
|---|---|
| Hard veto screen | No numerical, source, cap or missing-artifact veto; d40 score/later arm fails the descriptive heuristic promotion screen |
| Statistically supported ranking | None; no ranking test was declared for ten replicates |
| Descriptive-only differences | Every RMSE/SD/mean, particle-count, runtime and tail difference |
| Default-readiness | Not evaluated; CPU R diagnostic extensions only |
| Next evidence needed | Larger independent complete-learner replication with uncertainty, while retaining baseline and source gaps |

Post-run red-team: the strongest alternative explanation for relative accuracy
is Monte Carlo variation plus different final particle counts. The zero last-fit
floor effect is limited to observed finite clouds; new remote clouds could
overturn its generalization. One observation set per dimension is the weakest
part of any population interpretation. QR's improved adaptive behavior does
not repair the ill-posed unconstrained Eq15 objective previously demonstrated.

Execution: the three commands in the reviewed plan ran successfully. Eleven
launches consumed1908.66 CPU worker-wall seconds and zero GPU seconds; GPUs
were deliberately hidden and BLAS/OMP threads were one. Per-arm R timings are
CPU time and include explanatory refits in score arms, so they are not method
speed rankings. Remaining campaign budget45.226470 CPU/47.745145 GPU hours.
Manifests preserve commands, seeds, R version, Git commit, source snapshots,
checksums, histories, fitted guides and complete logs. No workers remain.
