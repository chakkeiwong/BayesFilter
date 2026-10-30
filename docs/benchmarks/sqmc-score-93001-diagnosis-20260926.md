# Why the first SQMC score has the wrong sign for seed 93001

The evidence identifies an inaccurate small-particle likelihood approximation,
shared across the Halton/Hilbert routes. The analytical derivative agrees with
finite differences of the scalar that was actually evaluated. This conclusion
is local to the checked cells; it does not certify the general implementation.

## Configuration and question

These are CPU float64, non-XLA reference/debug configurations of the repaired
P44 program with Contract E and dual-cap correction, D=3, T=2, and two flow
substeps. They are the saved bounded repair diagnostic, not a production
performance campaign. Each original N=12 route replays its saved controls and
[tuning artifact](../plans/artifacts/sqmc-repair-20260925/06-tuning-workflow/).
Changed particle counts, flow steps, and random inputs below are **UNTUNED
diagnostic interventions**. They cannot be used to promote defaults, rank
methods, or claim production accuracy. The data are fixed at seed 93001 for all
interventions; changing an input seed never regenerates observations.

The target is the derivative of the exact LGSSM log likelihood with respect to
theta[0]. The reported route quantity is the analytical derivative of its
finite particle log-likelihood program, locally at fixed discrete ancestry.
These quantities differ. The exact transition is
phi_j = 0.55*a_j*tanh(theta[0]), where a=(1,.85,.70); hence this component
measures persistence, rather than one state's mean or variance alone.

## Derivative check

| Method | Analytical score 1 | Central difference, h=1e-5 | Absolute discrepancy |
|---|---:|---:|---:|
| IID | -0.114613380 | -0.114613380 | 3.90e-11 |
| Hilbert inverse CDF | +0.377394691 | +0.377394691 | 8.45e-11 |
| Hilbert permutation | +0.109936585 | +0.109936585 | 3.84e-11 |
| Permutation cap ablation | +0.109990196 | +0.109990196 | 1.01e-11 |

Both h=1e-4 and h=1e-5 passed the predeclared derivative tolerance for all four
routes. Every original scalar and first score replayed within 1e-9. Ancestor
indices were unchanged at both perturbations in both observations. Thus the
wrong sign in these cells is not explained by an incorrect local derivative
or a finite-difference crossing of a sorting/resampling branch.

## Where the error enters

The total score is the sum of the derivatives of the two predictive
log-likelihood increments:

| Method | First observation | Second observation | Total score 1 |
|---|---:|---:|---:|
| Exact Kalman | +0.036627 | -0.185301 | -0.148674 |
| IID | +0.065732 | -0.180346 | -0.114613 |
| Hilbert inverse CDF | +0.597956 | -0.220562 | +0.377395 |
| Hilbert permutation | +0.338311 | -0.228375 | +0.109937 |
| Permutation cap ablation | +0.338311 | -0.228321 | +0.109990 |

The first observation accounts for the large positive error. Its exact
persistence-score contributions by state coordinate are -0.084918, -0.114067,
and +0.235612, which nearly cancel. The small cloud approximates this balance
poorly. The permutation's first-observation error is +0.301684; its
second-observation error is -0.043074, leaving the total error +0.258611.

All three Hilbert variants use exactly the same initial Gaussian Halton points
and joint process Halton points (sqmc_campaign_tf.py::random_inputs). The two
permutation variants use the same ancestry. Their .98/.97 correction-cap
difference is applied after the first likelihood increment, so that increment
is identical. They are not independent replications.

At the first observation, the inverse-CDF variant selects only 11 distinct
ancestors: it omits zero-based row 3 and uses row 11 twice. Those rows have
third state coordinates -0.949154 and +1.229857. The observed third coordinate
is +1.047001. The changed empirical ancestor distribution raises the
first-observation persistence score further. Duplication is an allowed
inverse-CDF resampling outcome, not by itself a coding defect.

We isolated the initial-cloud/ancestry contribution by integrating process and
observation noise analytically. For fixed selected ancestors x_i and uniform
weights, define V_j=Q_j+R_j and
l_i = log Normal(y_1; phi*x_i, diag(V)).
Then

    L_init(theta) = log[(1/N) sum_i exp(l_i)]
    d L_init / d theta[0]
      = sum_i softmax(l)_i sum_j
          (y_1j - phi_j*x_ij) * phi'_j*x_ij / V_j.

Here Q, R, and the initial states do not depend on theta[0], and selected
indices are held fixed locally. This is the exact first-observation score
for the empirical ancestor distribution, with propagation noise integrated
out; it is not the Kalman score for the Gaussian initial law.

It gives +0.219250 for the permutation cloud and +0.469998 for the inverse-CDF
ancestor cloud, versus exact Kalman +0.036627. Therefore much of the
first-observation error exists before process-noise sampling or flow
discretization. The actual corresponding increments, +0.338311 and +0.597956,
show additional error from the finite propagation/flow calculation.

## Discriminating interventions

Keep the same data, theta, and numerical controls; increase particle count:

| Method | N=12 | N=96 | N=384 |
|---|---:|---:|---:|
| IID | -0.114613 | -0.186131 | -0.034121 |
| Hilbert inverse CDF | +0.377395 | -0.118081 | -0.140830 |
| Hilbert permutation | +0.109937 | -0.176045 | -0.138671 |

Exact Kalman remains -0.148674. These single-seed observations support
finite-cloud sensitivity; they do not establish monotonic convergence, an
uncertainty interval, or a ranking. IID's nonmonotonic values are retained to
make that limitation visible. The generated larger Halton sets are not assumed
to contain the original small set as a prefix.

Increasing flow substeps from 2 to 24 at N=12 instead gives +0.326765
(inverse CDF) and +0.091805 (permutation): both retain the wrong sign.
Increasing integration resolution alone does not resolve this fixture.

Keeping data fixed and changing only the Halton input seed to 93002 gives
-0.627242 and -0.641681 respectively. The large sensitivity to the scramble
at N=12 is directly observed, with no confounding change in the observations.
Keeping the same Halton inputs but using identity parent pairing gives
+0.025648; ordering also affects the finite estimate, but removing the Hilbert
permutation does not reproduce the Kalman score.

## Decision and inference

| Decision | Primary criterion | Veto checks | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Close this local derivative-bug hypothesis | Fixed-input derivative parity passes | Replay, finiteness, and unchanged ancestry pass | Other parameter points and branch boundaries | Retain analytical score implementation | Global derivative correctness |
| Treat N=12 sign error as a real approximation failure | Distance to exact Kalman is material | These cells cannot certify accuracy | Bias/variance and broader scope not estimated | Separately planned adequate-N, exact-scope comparison | A new production setting or method ranking |
| Keep larger-N observations diagnostic | Both Hilbert routes move near Kalman for this dataset | No tuned claim exists for these changed scopes | One scramble at each size | Multi-scramble matched-data accuracy study with scope-specific tuning | Convergence proof or automatic promotion |

| Inference status | Finding |
|---|---|
| Hard veto screen | No replay, nonfinite, or local derivative-parity failure. N=12 accuracy is inadequate in the flagged case. |
| Statistically supported ranking | None. |
| Descriptive-only differences | All particle-count, flow-step, scramble, and pairing differences above. |
| Default-readiness | Not assessed; no defaults changed. All four routes remain diagnostic candidates. |
| Next evidence needed | Predeclared accuracy criteria, fixed-data independent scrambles, multiple datasets, exact-scope tuning, uncertainty, and constructed heuristic comparators. |

The strongest alternative explanation is additional systematic approximation
error in the flow/reset program. Local parity cannot exclude it, and two larger
particle counts cannot separate it statistically from random integration error.
Persistent discrepancy across larger independent scrambles would overturn a
purely small-cloud explanation. The weakest evidence is generalization beyond
this short synthetic dataset. The diagnostic rejects neither SQMC as a
research direction nor any route on general performance grounds.

## Reproduction

Plan: [diagnostic plan](../plans/sqmc-score-93001-diagnostic-20260926.md).
[Attempt 01](../plans/artifacts/sqmc-score-93001-20260926/attempt-01/) preserves
raw traces, fixed inputs, finite-difference traces, and its manifest.
[Attempt 02](../plans/artifacts/sqmc-score-93001-20260926/attempt-02/) preserves
every intervention, input set, result, and manifest. Both processes exited 0.
Full logs are saved in each directory. CPU use was deliberate: GPU devices
were hidden with CUDA_VISIBLE_DEVICES=-1 before TensorFlow import. No GPU
health inference follows from its no-device startup message.

Commands from /home/chakwong/BayesFilter-SQMC:

    timeout 300 /home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/diagnose_sqmc_score_93001.py --output docs/plans/artifacts/sqmc-score-93001-20260926/attempt-01
    timeout 300 /home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/diagnose_sqmc_score_93001_interventions.py --output docs/plans/artifacts/sqmc-score-93001-20260926/attempt-02

Measured post-import execution was 17.717 and 64.321 seconds, 82.038 seconds
total. The two 300-second process timeouts bound total launch exposure by
600 seconds, below the 900-second diagnostic allowance and the remaining
original CPU repair budget. No third attempt was needed. Each numerical
kernel has a stable explicit input signature. Separate kernels are cached per
finite configuration; TensorFlow's repeated-function-creation warning does not
mean an unbounded polymorphic trace was used.
