# The floor diagnosis transfers to the paper model; diagonal score fitting remains viable

The fixed .01 peak-relative floor fails badly in the paper's actual linear
Gaussian model. At dimensions 40 and 80, the likelihood ratios underflow to
zero while log errors remain finite and very large. Their near-zero empirical
variance would be a false success; relative RMSE is one. The dimension-aware
positive tail8 hypothesis avoids this collapse in all ten tested data sets.

| d | Diagonal/.01 | Diagonal/tail8 | Diagonal QR/tail8 | FA-APF | Full/tail8 |
|---|---:|---:|---:|---:|---:|
| 5 | 0.8447 | 0.096018 | 0.10318 | 0.11082 | 1.5039e-13 |
| 10 | 0.99266 | 0.096488 | 0.14888 | 0.20666 | 2.2737e-13 |
| 20 | 1 | 0.15658 | 0.45864 | 0.2116 | 9.0949e-13 |
| 40 | 1 | 0.21188 | 1.2866 | 0.86951 | 6.5427e-12 |
| 80 | 1 | 0.30785 | 0.98297 | 0.79601 | 0.0098741 |

These are relative RMSEs of Zhat/Z over four final streams on each of two new
data sets per dimension. Learned guides use N1000; FA-APF uses the paper's N5000.
The full structured table also includes bootstrap N10000, current-observation
N1000 and the exact full oracle. The diagonal score/tail8 arm clears the observed
cheap-heuristic screen in every dimension. QR/tail8 loses to FA-APF at d20,40,80,
and to the current-observation guide at d80. There is no statistically supported
ranking from this bounded sample. The exact oracle still defines the Gaussian
accuracy ceiling; these approximate guides are not promoted to defaults.

All400 final evaluations complete, with no numerical or fit rejection. The new
R core reproduces all48 saved TF recursions before these runs, passes the
independent target-gradient finite difference and rank/precision rejection
controls, and recovers the paper-model full Gaussian oracle to1.78e-14.
The actual R APF endpoint with the exact or recovered full/negligible guide
matches Kalman within the declared1e-8 tolerance. It samples the twisted X1
initial law, so the earlier local X0 finite-integration variance does not apply.

The tail8 floor is not universally negligible. At d80 it changes fitted means
or covariance entries by up to .0122 for the diagonal recursion and .1246 for
the full recursion on the initial bootstrap cloud. The full/tail8 arm then has
relative RMSE .00987, whereas the full/negligible control matches Kalman to
roundoff. At d<=40 the full/tail8 likelihoods are numerically indistinguishable
from Kalman at this tolerance. Observed proposal floor probabilities after
fitting are tiny (at d80 at most about6e-19); this does not measure floor effects
on the distant bootstrap points used during backward fitting. Treating that
proposal diagnostic as proof of negligible fitting distortion would be wrong.

The positive floor is defined by a Gaussian contour:
 c = phi(m) exp[-qchisq(N^-8,d,upper tail)/2].
If X has density phi, P(phi(X)<c)=N^-8, and a union bound over N draws is N^-7.
This argument concerns draws from phi, not the bootstrap cloud or transition
law. The exponent8 is an explicit reconstruction hypothesis, not an author
setting or a fitted optimum. At d80 the universal Gaussian-limit/non-harm claim
is rejected on the initial cloud. The candidate itself remains viable for an
iterative diagnostic with this limitation recorded.

The experiment uses the paper's A, identity observation/noise matrices, T100,
X1 timing and retained-weight ESS rule. It performs one score-regression pass,
not the full adaptive iAPF or the paper's Eq15 density optimizer. Published
Table1 values are standard deviations from1000 complete iAPF replicates;
our table is relative RMSE from eight one-pass evaluations per dimension.
Their numerical resemblance cannot establish replication. Original author
initialization, solver and floor remain unknown.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Limit |
|---|---|---|---|---|---|
| Continue diagonal score/tail8 as an explicit extension | Finite errors, oracle and parity checks pass | .01 floor fails; universal negligible-tail8 claim fails at d80 | Better iterative clouds, controller choices and Monte Carlo uncertainty | Fresh adaptive comparison, retaining floor diagnostics | No Eq15, author identity or paper replication |

| Inference status | Evidence |
|---|---|
| Hard veto screen | All implementations numerically viable; .01 likelihood collapse and QR heuristic losses reject their promotion |
| Statistically supported ranking | None; two data sets and four streams each |
| Descriptive-only differences | RMSE, ratio mean, SD, log MSE, floor effects and runtime |
| Default readiness | No change; independent R extension only |
| Next evidence needed | Full adaptive controller on fresh data with independent final runs and uncertainty; inspect both plausible doubling-start conventions |

Post-run review: floor distortion during initial fitting is the strongest
remaining alternative to attributing all residual error to diagonal covariance.
An adaptive fit uses different, increasingly informed clouds; it may repair this
problem or expose another failure. The next experiment must record the target's
Gaussian responsibilities during fitting and compare floor/no-floor coefficients
on the same cloud, not infer fitting safety from tiny final proposal floor use.
Frozen references were not edited. No TensorFlow production, TF32, model-score,
HMC, KDM or canonical LEDH claim follows.

Exact commands, source snapshots, seeds, observations, pilot clouds, fits, final
diagnostics and R version are in the versioned manifests and per-case RDS/CSV
files. CPU-only execution hid GPUs and used at most two single-threaded workers.

Terminal verification:119 checks plus48 preflight recursions and independent controls. CPU worker time 271.276s; remaining 45.756654 CPU /47.745145 GPU hours.
