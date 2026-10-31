# Controlled-SMC source matrix: remaining experiments and discrepancies

Checked 29 September 2026 during the authorized master execution. The current
neuroscience experiment is unaffected by the issues below. Author code is pinned
at 2696f6d611aae172d10102cb7bfbfd314bb6b88f.

## Lorenz-96: the paper and source experiment are now mapped

Published section 6.2, equations (42)–(44), Figure 6 and Table 1 specify nonlinear
RK4 dynamics, Gaussian transitions and partially observed Gaussian measurements.
The state dimension is d, the observation dimension is d-2, initial covariance
is .01 I, transition covariance is .001 I, observation interval is .1, RK4 step
is .01, and T=100 means 101 observations including time zero.

Figure 6 uses the source Lorenz/SimulatedData.mat (101 by 6 observations), d=8,
observation variance 1e-4 and data-generating forcing 4.8801. The forcing grid is
40 equally spaced values from 2.5 to 8.5, with 100 repetitions per value.
Lorenz/cSMC_VaryF.m:72–85 uses N512, maximum four refinements, quadratic policies
and adaptive refinement. Lorenz/cSMC_Resample.m:245–258 checks minimum ESS strictly
above 90% after a refinement. It always performs at least one refinement.

The comparator is the fully adapted APF, not an untwisted bootstrap filter.
Lorenz/APF_VaryF.m:96–106 reads each cSMC repetition's iteration count and assigns
APF N=1382,2867,4505,6144 for one,two,three,four refinements respectively. These
are historical CPU-cost choices, not equal-GPU-cost settings. The APF run
therefore depends on the corresponding cSMC iteration matrix.

Table 1 uses d=8,16,32,64 crossed with observation variances 1e-4,1e-3,1e-2,
forcing 4.8801, one refinement and 100 repetitions per cell:

| d | cSMC N | APF N |
|---|---:|---:|
| 8 | 512 | 1382 |
| 16 | 512 | 2027 |
| 32 | 1024 | 4034 |
| 64 | 4096 | 11468 |

The counts occur at lines 42 and 52 of the respective Repeat_Dim*.m drivers.
Those drivers generate one new dataset per noise setting before their repetition
loop (lines 75–88); they do not load the supplied Figure 6 dataset. The preserved
VaryZetasq result matrices alone do not recover these unseeded generated datasets.
A fresh Table 1 implementation can replicate the generative design, but exact
dataset/point-value reproduction requires recovering or reconstructing that
provenance. The additional paper comparison to iAPF uses 1000 repetitions at
forcing 4.8801, N512, I3; its driver has not been identified and this comparison
is not satisfied by the two-filter Figure 6 matrix.

### Gaussian normalization must be explicit before the Lorenz port

The paper writes normalized Gaussian observation densities. The source drivers
instead define Observation.LogDensity as
`-.5 * precision * sum(residual.^2)`; see APF_VaryF.m:43 and Repeat_Dim8.m:88.
The initial policy constants likewise omit the Gaussian factor
(APF_VaryF.m:60–66). The consumer uses that supplied log-density at
cSMC_Resample.m:78 and :121, accumulates it into LogNormConst at :87–88 and
:129–130, and cSMC_VaryF.m:100 / APF_VaryF.m:108 save it without a restoration.
AnalyseResults.m:34–35 takes variance divided by squared mean of those saved
log-normalizers. Static call-chain inspection is now backed by the
[executed constant-restoration check](../plans/artifacts/controlled-smc-kdm-replication-20260928/attempt19_c1_lorenz_normalization/summary.json):
all8 paired checks passed, with exactly unchanged trajectories and a maximum
final-normalizer shift error of1.77351e-11. The
[bounded plan](../plans/controlled-smc-lorenz-normalization-check-20260929.md)
limits this evidence to fixed analytic policies without refitting.

For p=d-2 observed coordinates and H=T+1 observations, the normalized result is

    log Z_normalized = log Z_source - H*p/2 * log(2*pi*observation_variance).

This follows by multiplying every observation potential by its missing Gaussian
normalizer. That factor is independent of particle state, so normalized weights
and proposals are unchanged. For a fixed cell it also leaves variance(log Z)
unchanged, but shifts mean(log Z), and therefore changes the reported ratio
variance(log Z)/mean(log Z)^2. This is a real distinction between the paper's
normalized target and the source output, not a regression or XLA issue.
It does not establish that controlled SMC's inference idea is wrong.

The implementation must expose and preserve both the source output convention
and the normalized likelihood. Use the source convention to compare saved
author curves; use the normalized convention when making a likelihood claim.
No silent constant drop, post-hoc curve matching or cross-noise equivalence claim.

## Multimodal model: located in the manuscript supplement

The multimodal example is specified in the inspected arXiv manuscript,
Appendix F, equation (78), Figure 9 and Table 2; Appendix I.1 supplies the
Gaussian-mixture twisting formulas. Its exact identity in the final published
supplement has not been checked. It must no longer be described as an unmapped
source-only toy.

It has initial N(0,5), nonlinear mean
x/2+25x/(1+x*x)+8*cos(1.2*t), transition variance 10 and measurement mean x*x/20.
Unlike the neuroscience/Lorenz quadratic log-policy, Appendix F.1 fits a positive
RBF expansion in the natural scale by nonnegative least squares, then retains
the M largest coefficient knots. The coefficient beta equals the sample
standard deviation times the declared bandwidth factor; it is not a conventional
Gaussian bandwidth standard deviation. Source cSMC_Resample.m:194–230 matches
that fit/knot/normalizer structure. Its consumer is not implemented by the current
scalar quadratic TensorFlow core. Replacing it by that core would change the
specified method.

The Appendix F.2/Table 2 matrix is N512, I1, M16, 100 repetitions, with:

| Observation variance | RBF bandwidth factor | BPF N |
|---|---:|---:|
| .1 | .5 | 2252 |
| .5 | .4 | 4710 |
| 1 | .3 | 6553 |

Two source issues remain. README.txt names Repeat.m, but that driver is absent
from the pinned directory. Appendix F.2 says T+1=100, while Figure 9 and Demo.m /
Model_ToyMultimodal.m use T=100, and all three supplied observation files contain
101 rows. The latter is confirmed by the saved CPU-only
[source-container metadata audit](../plans/artifacts/controlled-smc-kdm-replication-20260928/source-container-audit-20260929.json),
which includes file hashes. Do not silently discard an observation or label a
101-observation source run as the literal 100-observation F.2 specification.
A future executable phase should freeze the 101-row source experiment with
this discrepancy disclosed, and separately state whether a 100-row textual
interpretation is being evaluated. The current demo also generates fresh data;
the empirical driver must explicitly consume the supplied data instead.

## Decision and next action

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Lorenz source matrix mapped | Source drivers and published experiment inspected | Constant identity passed; full regression/empirical execution remains unchecked | Table 1 data/RNG provenance; conditioning of multivariate regression | Build/test full covariance RK4/APF fitting and proposals before a cost pilot | Multivariate correctness, equal GPU cost or empirical replication |
| Multimodal supplement mapped | F/F.1/F.2/I.1 and actual NNLS source inspected | Missing Repeat driver; 100/101 timing discrepancy remains | Final supplement mapping and exact saved-result provenance | Reconstruct a bounded NNLS/RBF reference phase with fixed supplied data and disclosed horizon | Current quadratic core implements RBF policies or reproduces Table 2 |

The strongest alternative explanation for source/paper numerical differences is
now a data or normalization convention, before any claim about optimizer failure.
These findings repair the plan; they do not justify launching large unvalidated
multivariate or RBF jobs.

## Additional author-hosted supplement check

The [author-hosted manuscript and supplement](https://www.stats.ox.ac.uk/~doucet/HengBishopDeligiannidisDoucet_controlledSMC.pdf) is now stored locally as
.localresources/papers/heng-controlled-smc-author-manuscript-and-supplement.pdf,
with extracted text and a URL/SHA-256 receipt beside it. It is marked submitted
to the journal, so it is not being relabeled the publisher supplement.
The publisher supplement DOI 10.1214/19-AOS1914SUPP resolved, but returned an
HTML access challenge instead of the PDF during this check.

The author-hosted supplement gives the multimodal model in equation S33,
section 6.1 natural-scale nonnegative RBF fitting and top-M component selection,
section 6.2/Table S1 experiment settings, and section 9.1 Gaussian-mixture
twisting formulas. Checked extracted-text ranges are 2796–2843, 2880–2967 and
3436–3446 and 3529–3589. They corroborate the earlier Appendix F settings and the distinction
from quadratic regression. They also retain the discrepancy: Figure S2 says
T=100, whereas section 6.2 says T+1=100; all supplied datasets have 101 rows.
The final published-supplement identity and the exact table-generating driver
remain unresolved. This is explicit source-version/provenance debt, not
evidence that the method is invalid.

## Full-covariance source boundary

Lorenz/cSMC_Resample.m:190–230 solves normal equations and assembles full
quadratic cross terms for d>1. Its .001 precision projection applies only to
d=1 or purequadratic fits. The multivariate full-quadratic branch does not
project eigenvalues; EnsureSPD calls at :54 and :99 are commented out. A future
full-covariance port must reject nonpositive proposal precision before claiming
a valid Gaussian. Adding damping/projection would be a separately evaluated
change to source behavior, not permission to silently use the scalar repair.
The present scalar TensorFlow core does not implement this multivariate branch.


## Executed RBF conditional-kernel audit,29 September2026

Pinned author ToyMultimodal/cSMC_Resample.m lines110–119 sample component
probabilities from row n but Gaussian means from Ancestor(n). This is wrong
relative to the claimed parent-conditioned proposal. Published Def3.2/eq11
(text342–366), Algorithm1 step2(a) (453–472), and the author's full supplement
section9.1 (text3544–3588) require the same selected parent in both quantities.
For policy exp(c) sum_j w_j exp[-beta(z-k_j)^2], base N(m(x),q), define
D=1+2 beta q. Component mean is (m(x)+2 beta q k_j)/D, variance q/D,
probability proportional to w_j exp[-beta(m(x)-k_j)^2/D].
Thus ancestor conditioning follows directly from the Gaussian product, not a
performance preference. The source does not include a weight correction for
its different actual proposal.

Attempt34 confirmed98 affected particle transitions and max probability total
variation0.9999971113. Instrumentation did not change source outputs; correcting
only the row index removed the mismatch. Attempt35 passed18 corrected TF/XLA
forward checks, exact ancestry and fresh-stream validity. Full reference,
diffs and identity controls are preserved in the campaign artifacts.

The full N512/H101 NNLS reference subsequently exposed another issue.
Octave's NNLS at the third backward time exhausted2560 iterations on a
512-column matrix with numerical rank190. The default source caller allowed
100000 iterations and did not inspect convergence. A frozen independent
SciPy NNLS solve converged and passed KKT checks (attempt39), while SciPy BVLS
reported success but failed the same guard. This is an observed numerical
solver pathology; it does not establish failure of MATLAB's solver or of the
paper's convex fitting objective. Corrected-kernel and alternative-solver
results must be labeled explicitly, with no assertion that they reproduce
the provenance of the paper's saved Table2 values.

The scoped Lorenz source-forward and independent centered SVD/QR checks have
also passed since the earlier source-map table (attempt32,28 checks); the
full forcing/dimension/refinement matrices remain open. Current decisions:
../plans/controlled-smc-rbf-fitting-execution-result-20260929.md.
