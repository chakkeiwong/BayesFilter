# Random mixture benchmarks for NeuTra generalization

Status: benchmark design, added in response to the owner's October 3 proposal.
The two families below extend the fixed unwarped mixture. They have not been
implemented or executed by this change. Interval values are proposed benchmark
choices, not calibrated defaults or new compute allocations.

The subsequent [generic recovery and transfer plan](bayesfilter-neutra-generic-recovery-and-transfer-plan-2026-10-03.md)
places these families after simple-case confirmation and before dimensional
and state-space transfer for every qualifying method among FAB, Gabrié, AIS,
SMC, AFT and CRAFT. Its proposed simultaneous reliability screen covers twelve
method-family claims: 53 zero-failure trials per pair, using 106 unique shared
targets and up to 636 complete method-target trials. This supersedes the
earlier 36-per-family design for just one procedure; the illustrative 29-trial
single-family calculation below is not a six-method allocation.

## Question and skeptical design review

Does the same training and sampling procedure work on previously unseen
Gaussian mixtures, rather than depending on the original centers at (-5,0)
and (5,0), unit variances, fixed orientation and probabilities 1/3 and 2/3?
The candidate remains the configured canonical IAF with the selected practical
sample-generation and training procedure. Retain the original fixed mixture
as a regression case. An exact-sample forward fit is an explanatory control
for separating teacher error from fitting error; it is not the practical
candidate.

The design review identifies five ways this study could mislead us: new seeds
on the old density would not test target generalization; center separation
without component widths would misstate difficulty; independently drawn
triangle edges can be inconsistent; component means are not necessarily
density modes; and truth-dependent initialization or checkpoint selection
would leak the answer. The constructions and development/confirmation split
below address these risks. The design is suitable for implementation planning,
but launch still needs measured cost, sample sizes and accuracy requirements.

## Two additional target families

For K=2 and K=3, use the normalized, unwarped density

\[
p_\psi(x)=\sum_{k=1}^K w_k\,\mathcal N(x;\mu_k,v_k I_D),
\qquad v_k\stackrel{\mathrm{iid}}\sim U[c,d],\quad 0<c<d.
\]

Here v_k is a **variance**; its standard deviation is sqrt(v_k). Different
components have independently drawn widths. Begin in D=2 to permit direct
geometry inspection. No nonlinear transformation is applied to these targets.

**Two centers.** Draw S uniformly from [a,b], a random unit direction u, and
a random translation t. Set mu_1=t-Su/2 and mu_2=t+Su/2. Then their Euclidean
distance is exactly S. Draw S, translation, direction, variances and weights
independently. Record the realized parameters and random-generator identity.

**Three centers.** Draw S_12, S_13 and S_23 independently and uniformly from
[a,b], choosing 0<a<b<2a. This guarantees the triangle inequalities: its
largest possible side is less than its two smallest possible sides combined.
An explicit construction before translation and rotation is

\[
m_1=(0,0),\quad m_2=(S_{12},0),\quad
m_3=(h,\sqrt{S_{13}^2-h^2}),\qquad
h=\frac{S_{12}^2+S_{13}^2-S_{23}^2}{2S_{12}}.
\]

Subtract the unweighted centroid, apply a uniformly random planar rotation,
and add t. Thus all three pairwise distances have the requested uniform law
and the geometry is generally asymmetric. A wider interval with b>=2a would
require a different declared joint law: rejecting invalid triangles changes
the marginal distance distributions. Do not silently call those accepted
edges independent uniforms. Collinear three-center targets can be a separate
stress case; their outer distance is the sum of the two adjacent gaps.

To prevent fitting only the old mass ratio, also randomize positive weights.
A concrete proposal is

\[
V\sim\operatorname{Dirichlet}(1,\ldots,1),\qquad
w_k=\eta+(1-K\eta)V_k,\quad 0<\eta<1/K.
\]

This gives uniform weights on the simplex with a lower mass floor. Equal
weights remain a useful control, but should not be the whole test family.

| Proposed choice | Provenance and purpose | Limitation or early check |
|---|---|---|
| K=2 and K=3 | Owner-requested additional families | Centers need not correspond one-to-one with density maxima |
| Distances [a,b]=[6,10] | Convenience proposal in the old unit-variance coordinate system; includes the old distance as a boundary and ensures b<2a | A bounded stress range, not representative of every posterior; plot realized relative separation |
| Variances [c,d]=[0.5,2] | Convenience proposal spanning a factor of four around the old variance 1 | Isotropic components do not test within-component anisotropy |
| Translation coordinates U[-2,2] and planar angle U[0,2*pi) | Convenience proposal to remove the origin/axis shortcut | Keep the initialization/search policy independent of each realized translation |
| eta=0.1, Dirichlet parameters all 1 | Convenience proposal to vary proportions while every component has material mass | Does not test arbitrarily rare components; report that boundary |
| D=2 initially | Inherited economical inspection dimension | Later higher-dimensional and state-space checks are still required |

Keep these numbers configurable and freeze their final values before drawing
confirmation targets. They are not optimization results or literature defaults.
For each pair, report the width-adjusted separation

\[
\delta_{ij}=\frac{\|\mu_i-\mu_j\|}
 {\sqrt{(v_i+v_j)/2}}.
\]

Under the proposed intervals it ranges from 6/sqrt(2) to 10/sqrt(0.5), about
4.24 to 14.14. This is an explanatory geometry measure, not a guarantee of K
distinct modes. Retain all drawn targets; overlapping components are not a
reason to silently redraw an inconvenient instance. Assess a mode-count claim
only where independent geometry checks support that claim.

## Generalization protocol and exact references

Separate development and confirmation at the level of the entire target
specification psi, not only at the level of sampled points or training seeds.
Develop on one set of mixtures, freeze the algorithm and its automatic
calibration/repair rules, and assess fresh mixtures plus fresh training and
sampling randomness. Each target is trained anew; transfer here concerns the
procedure, not reuse of the same fitted neural parameters. Automatic
target-specific calibration is allowed when its rule and allowance were fixed
before confirmation. Truth-based manual retuning after seeing a failed target
would make that target development evidence.

The practical method receives dimension, batched log-density/score access,
and a declared common initialization distribution. It must not receive the
component count, true parameters, component labels, exact target samples, or
the evaluator's analytic transport. Its own mode search may infer structure
from the density. The test generator and evaluator retain the full target
specification. Development evaluations can inform the general procedure, but
the practical per-target training/stopping path must use only information
available for an unknown posterior. Exact-sample training is a separately
identified oracle control.

Let phi_k denote the kth component density and r_k(x)=w_k phi_k(x)/p_psi(x).
The identity p_psi(x)r_k(x)=w_k phi_k(x) gives exact evaluation quantities:

\[
E_p[r_k]=w_k,\qquad E_p[r_kX]=w_k\mu_k,\qquad
E_p[r_k(X-\mu_k)(X-\mu_k)^T]=w_kv_k I_D.
\]

These check component representation and within-component moments without
treating hard cluster assignments as latent mixture labels. Also evaluate
physical regions, tails and projected shape using independent exact draws.
A Voronoi cell's probability is generally not w_k. The old x_1>0 diagnostic
and fixed valley interval cannot be reused after arbitrary translation and
rotation. Exact responsibility means alone are insufficient: the old broad
Gaussian failure shows why shape checks remain necessary.

Keep distinct assessments for discovery, the estimated weighted teacher,
the forward-trained map, any refinement, and the final posterior sampler.
Preserve warm maps when RKL worsens coverage. The standard 1,000-base-point
score test remains explanatory; fresh physical-space and low-density-region
checks are needed to expose regions rarely visited by the fitted map.

## Evidence contract for the subsequent executable plan

| Role | Requirement |
|---|---|
| Primary question | Generalization of each of the six methods' frozen practical procedures across newly drawn two- and three-component mixtures |
| Comparator | Exact target identities and independent reference draws; original fixed mixture as regression, exact-teacher fit only for diagnosis |
| Primary pass criterion | Final frozen-map posterior estimates satisfy predeclared component, shape/moment, convergence and uncertainty requirements on unseen targets within the total cost allowance |
| Promotion veto | Invalid values/scores/weights; truth leakage; material component or shape error; failed final numerical or posterior checks |
| Repair trigger | Localize failure to discovery, teacher, fitting, refinement or downstream sampling; change the procedure on development targets, then use fresh confirmation targets |
| Continuation veto | Invalid reference/harness, corrupted evidence, unavailable required diagnostic, or exhausted total budget |
| Explanatory | Loss, acceptance, ESS/CESS, ancestry, relative separation, score residuals and timing; none alone establishes posterior correctness |
| Result preservation | Unique output root with realized target specifications, split and seed identities, source/version, selected settings, stage checkpoints, failures, uncertainty and total cost |
| Not established by passing | Universal mode discovery, a universal optimizer setting, method superiority, or q20 state-space validity |

Do not pool all chains or repeated seeds as independent new targets. Report
all attempted targets, failures and cost, with uncertainty across independent
targets and replication within targets. Size confirmation by its intended
precision, not a convenient count. For example, with zero failures in n
independent target-and-seed trials from a frozen family, the exact one-sided
95% upper bound on failure probability is 1-0.05^(1/n). Bounding it below 10%
requires at least 29 such trials; three successful trials would be weak
generalization evidence. These confidence/error levels are illustrative design
choices, not an allocation or a required success threshold. Correlated trials,
adaptive stopping or multiple family-wise claims need a corresponding analysis.
For the current six-method study, use the main plan's twelve-claim calculation
and paired target list. Freeze each method's procedure before any final target
results are exposed. Within-method trials must be independent; sharing a target
across methods does not create six independent targets. An ineligible target
cannot be discarded when reporting unconditional delivery in that family.

Budget rare-event precision before execution. For independent Bernoulli draws
of an event of probability p, relative standard error is sqrt((1-p)/(n*p)).
Therefore arbitrary relative precision on vanishingly rare bridge events can
make a test unaffordable even with exact independent samples. Choose required
estimands and precision with independent controls; do not silently relax them
after a learned map fails, or require every finite chain to observe an event
whose expected count is negligible.

## Implementation and transfer scope

The current `neutra_warm_start_targets_tf.py` hard-codes the fixed two-center
mixture. Adding two target names alone is insufficient. Parameterize density,
score, reference generation and target identity together; keep truth available
only to evaluation; replace hard-coded sign/valley features; and make consumer
initialization, selection and reporting accept the same frozen specification.
No new learned transport architecture is needed for these targets.

Before a campaign, verify density/score identities, covariance-versus-standard-
deviation handling, triangle distances, deterministic regeneration and the
information boundary with focused reference tests. Use batched TensorFlow/TFP
and the existing GPU/XLA training route. Record CPU reference generation
separately. Price a small development run before fixing the total matrix and
compute allowance. This design does not launch or claim to wire either family
into the master program.

Passing these families would support transfer across locations, widths, weights
and two/three-component geometries in the tested distribution. Extend dimension
separately, including D=20, but recognize that isotropic mixtures with only two
or three centers have a low-dimensional span of center differences. They do
not reproduce recurrent likelihood geometry. The subsequent transfer test must
use a reduced state-space/LSTM model with the same target-evaluation structure,
followed by q20 with its own validity and posterior checks.
