# iAPF complete-data score: executable next stage

Date: 2026-09-18. Owner request: refresh the master and execute its next step.
Master: `younis-kdm-score-master-program-2026-09-14.md`.
Output: `artifacts/younis-iapf-fisher-score-20260918-01/`, one new attempt
directory per launch. This stage follows the completed curved diagnosis.

## Question and mathematical target

Does a complete-data/Fisher score on the existing iAPF genealogy remove the
observed discrepancy between the fixed-label derivative and the physical
model score? This is a new estimator of the model score, not a correction
asserted to differentiate the existing finite likelihood program.

For the physical model, define

\[
 p_\theta(x_{0:T},y_{1:T})=\mu_\theta(x_0)
 \prod_{t=1}^T f_\theta(x_t\mid x_{t-1})g_\theta(y_t\mid x_t),\qquad
 S_\theta=E_\theta[\partial_\theta\log p_\theta(X,Y)\mid Y].
\]

The derivative inside this expectation holds the entire state path fixed.
Here the Gaussian densities have parameter-independent support and smooth,
positive variances, so differentiation under the finite-horizon integral is
valid locally in the six parameters. Fisher's identity is Poyiadjis, Doucet
and Singh (2011), section 2.1 equation (4). Their Algorithm 1 and equation (7)
carry the additive score through resampling. Local primary sources:
`.localresources/papers/poyiadjis-doucet-singh-2011-author-recovered-20260911.pdf`
and its text, lines 159--244. Section 2.3 and Appendix 1 discuss genealogy
variance; their uniform mixing assumptions are not asserted for our Gaussian
model. Author-code parity for this new extension is not checked.

For our fitted positive twists let
\(h_{t-1}(x)=\int f_\theta(z\mid x)\psi_t(z)dz\), \(h_T=1\),
\(q_t=f_\theta\psi_t/h_{t-1}\), and
\(G_t=g_\theta h_t/\psi_t\). The actual initial potential is \(h_0\).
Cancellation gives
\[
 \mu_\theta(x_0)h_0(x_0)\prod_{t=1}^T q_t(x_t\mid x_{t-1})G_t(x_t)
 =\mu_\theta(x_0)\prod_{t=1}^T f_\theta(x_t\mid x_{t-1})g_\theta(y_t\mid x_t).
\]
Thus the terminal weighted iAPF genealogy approximates the required physical
smoothing distribution, although intermediate twisted distributions differ.
This cancellation is a local derivation using the existing implementation's
Guarniero--Johansen--Lee equations (5)--(6), already inspected in the preceding
reference campaign. No categorical-label derivative is needed to compute this
different expectation. This is not a proof that an uncorrected derivative of
the sampling program equals the model score.

Initialize \(a_0^i=\partial_\theta\log\mu_\theta(x_0^i)\); gather it with
exactly the same initial ancestor indices as the state. Each transition adds
\(\partial_\theta\log f_\theta(x_t^i\mid x_{t-1}^i)+
\partial_\theta\log g_\theta(y_t\mid x_t^i)\), then the next resampling
gathers this accumulated quantity with the state's ancestor indices. Report
\(\sum_i W_T^i a_T^i\) before terminal resampling. The old value, local
derivative and cloud outputs retain their semantics and defaults.

For a Gaussian residual r=x-m and covariance C, the analytical partial is
\(m'_\theta{}^T C^{-1}r+
\tfrac12[r^TC^{-1}C'_\theta C^{-1}r-\mathrm{tr}(C^{-1}C'_\theta)]\).
Reuse the checked Gaussian algebra with zero state tangents. In the scalar
model, the initial contributions are .7(x0-m)/P and (x0-m)^2/P-1 for the last
two parameters. Transition contributions are xprev*rQ/Q and rQ^2/Q-1;
observation contributions are rR^2/R-1 and 1.15*x*rR/R. These provide an
independent closed-form test of all six components, including initial law.

## Evidence contract and intent ledger

Primary engineering criterion: independent scalar and multivariate density
checks, executable genealogy replay with nonidentity ancestors, FP64 parity
of the existing outputs, terminal-resampling invariance, stable tracing, and
actual fitted-proposal consumer wiring. Failed invariants invalidate the
implementation and trigger a bounded repair before research interpretation.
The two score estimators must be returned by one joint kernel invocation, so
their particles, labels and likelihood value are identical by construction.
Separate FP32 compilations have the precision limitation documented below.

Primary diagnostic: conditional model-score bias and squared error against a
mesh/domain-checked FP64 grid reference, plus an affine Kalman check. Compare
Fisher and fixed-label scores on identical particles, observations, parameters
and fitted coefficients. Report component means and MCSE; at N4096 report
Bonferroni-adjusted approximate Student-t 99% family intervals across the 24
fresh nonlinear data/component comparisons. Oracle numerical error is a
separate bound, not Monte Carlo evidence. Coverage is a screen, not proof of
unbiasedness or consistency. Failure rejects this finite-budget candidate,
not Fisher's identity; incorrect lineage or density algebra is a repair trigger.

Conditional situations: weak and curved nonlinear observations, two fresh
datasets each, T=2; N=256,1024,4096 and 64 independent final particle streams.
Dataset IDs 1400/1401 weak and 1410/1411 curved; affine check 1390. No selection
uses these final streams. Data and offline-fit streams are separated from the
final streams. This stage has no score-estimator hyperparameters to tune.

Constructed heuristic adversaries: (1) constant-twist/bootstrap Fisher, to
check whether fitting is needed; (2) prior proposals without resampling,
which avoid categorical resampling and are feasible at T=2; (3) EKF, using
local linearization; (4) UKF, using deterministic moment quadrature. The exact
affine Kalman reference additionally checks the zero-curvature limit. Report
each comparison separately by dataset and N. Any observed loss to a heuristic
is a promotion veto. Paired percentile-bootstrap 99% intervals (4000 draws)
describe conditional MSE differences; intervals are individual, exploratory,
not simultaneous rankings across models. No timing ranking on a shared GPU.

Continuation vetoes: invalid/nonfinite density, broken reference refinement,
wrong genealogy, missing required output or provenance, budget exhaustion.
An offline fit rejection or bound activity vetoes promotion and triggers later
fitting repair; a usable bounded fit can still answer this score-mechanism
question, and constant-twist evidence remains possible if fitting fails.

Do not conclude finite-N unbiasedness, a gradient identity with log Zhat,
HMC admissibility, long-horizon stability, default readiness, source-author
code parity, canonical LEDH validity, or cross-model scientific superiority.
Fisher's terminal path estimator can have substantial genealogy variance.
If sampling uncertainty dominates, next work is smoothing/variance reduction
with a new evidence contract; if persistent bias exceeds uncertainty, inspect
lineage, physical partials and target cancellation before enlarging runs.

## Default and assumption audit

| Choice and provenance | Reason and possible failure | Early check; status |
|---|---|---|
| Six-parameter model, T2, theta from curved diagnosis | Isolates score semantics; cannot establish long-horizon behavior | Independent density formulas, affine case; diagnostic baseline |
| Floor .001 and relative-shape fit, prior diagnosis | Freeze a previously nominated proposal, not a new default; bounds may remain active | Preserve fit diagnostics and veto promotion; hypothesis |
| Fit N16/cap128, 4 iterations, bounds +/-4, sd .2--4, 10000 fit steps | Reuses actual consumer and its known finite-budget weaknesses | Fit validity, projected gradient, bounds; inherited baseline only |
| Initial plus transition plus observation partials | Required by physical joint density; omission biases parameter components | Closed form and fixed-state finite differences; derived |
| Genealogy estimator, no added damping/clipping | Exactly the defined additive estimator, no matrix modification or new numerical control | Positive finite variances and density checks; derived |
| FP32/TF32 GPU XLA; FP64 CPU reference and FP64 offline fitting | Existing target and independent oracle precision | Explicit devices, growth, ref refinement, CPU tests; scoped choice |
| Fresh data and 64 final streams | Avoid old holdout tuning; finite replication may miss tails | Paired intervals, component MCSE, retain rows; bounded diagnostic |
| Grid 201/401 radius9 and 601 radius13.5 | Prior checked grid strategy, not unconditional accuracy assumption | Per-dataset refinement <=1e-7, tail<=1e-9; independently rechecked |

## Implementation, budget and review

1. Add optional analytical Fisher output inside the existing fitted-twist
   kernel, keeping its public default return unchanged. No duplicate filter.
2. Run focused CPU tests with GPU intentionally hidden. Then execute a bounded
   GPU/XLA diagnostic, preserving all particle-level aggregate rows, fit
   records, reference checks, source hashes, commands, device and wall time.
3. Review the results for wrong-target claims and heuristic losses. Refresh
   the master and concise checkpoint with the evidence and exact next action.

Total budget: <=4 driver attempts, <=1800 summed driver seconds, <=8 adaptive
fits, <=3000 filter-call charges; <=600 seconds of tests/probes. Compilation
counts toward driver time. Local harness repairs can retry within the same
contract/budget in a fresh directory. Fit controls are frozen for this stage;
neither failed data nor final particle streams can select new controls.

Environment: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`;
GPU UUID `GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a`, escalated commands,
`TF_FORCE_GPU_ALLOW_GROWTH=true`, verified repository memory helper.
CPU tests use `CUDA_VISIBLE_DEVICES=-1`. Exact commands will be saved in the
run manifest. Driver: `docs/benchmarks/diagnose_younis_iapf_fisher.py`.

Skeptical pre-execution review, 2026-09-18: pass after explicitly separating
the physical score from the finite-program gradient. Fixed-state density
partials must include the initial distribution, and terminal weights must
precede the final resampling; both are executable checks. The same-particle
comparison prevents a proposal or random-stream change from masquerading as
a score repair. Old fit nominations are labeled hypotheses, bound activity
cannot be hidden, and numerical reference failure stops interpretation.
Proxy fit loss cannot promote this score. Long-horizon genealogy degeneration
and approximate finite-replication intervals remain limitations. This review
is local and advisory; no independent reviewer is claimed.

Execution repair note: the first broader CPU check exposed two pre-existing
quadrature API migrations in the KDM mixture and SGQF consumers (new shared
validity outputs were passed as moments or incorrectly unpacked). Include the
bounded migration and fail-closed flag regressions, including validity in the
mixture schedule consumed by LEDH. This restores the current shared API;
no numerical formula or default is changed. The independent scalar replay
must reproduce the fixture's actual FP32-rounded constants even in FP64,
not replace them with exact decimal constants. Original failures are saved
in `cpu-tests.log`; neither issue is evidence against Fisher's identity.
The shared invalid-target return is (-infinity, zero score), so the injected
failure checks require that exact pair, not a NaN score. For the new numerical
comparison the reference parameters are the exact FP64 representation of the
executed FP32 parameter vector; a 5e-6*(1+abs(reference component)) numerical
allowance accompanies, and is kept separate from, the bias confidence bounds.
The affine 1390 check uses the same actual fitting consumer and frozen controls;
it adds one fit and only the N4096 rung. Total planned calls remain below 3000.

### Reviewed continuation after attempt01

Attempt01 stopped after 43.568943 seconds, 2 adaptive fits and 624 filter-call
charges. Both fits and the complete affine dataset are preserved; the partial
1400 particle results were not saved and must be repeated. The failed check
compared separate FP32 XLA compilations with and without the extra score.
At dataset1400/N4096, one of 4096 second-step cloud particles differed, the
log-likelihood differed by 1.62e-5 and the fixed-label score by at most 2.00e-4.
The same FP32 inputs cast to FP64 produced identical clouds and value, and
score agreement within 2.23e-16. See `gpu-parity-diagnosis.json` in the output
root. Non-XLA GPU returned unusable empty tensors in this self-built TensorFlow
environment; that nondefault route is not used for the comparison.

Skeptical repair review: exact trajectory parity across separately optimized
FP32 programs was an unjustified numerical assumption at a discontinuous
categorical threshold. It is not needed to compare the two estimators returned
by a single execution. The probe supports rounding amplified by resampling,
but does not prove the exact compiler operation responsible. Keep this failed
cross-compilation check as a precision diagnostic; do not increase its tolerance
or claim bit-identical optional/default GPU outputs. The numerical kernels,
fit controls, physical oracle, bias criteria and heuristic vetoes are unchanged.
Independent density and genealogy tests, FP64 parity and the joint execution
remain the engineering checks. This repairs the harness assumption, not a
failed physical-score criterion. No scientific continuation veto has fired.

Resume from attempt01's complete affine result and frozen dataset1400 fit,
checking their numerical source hashes. Save each completed particle rung.
The remaining four nonlinear datasets require 2304 particle calls, 48 moment
calls and 24 fit charges, exactly the 2376 remaining charges. Do not rerun the
separate-compilation comparison inside this campaign. There are 3 attempts,
1756.431 driver seconds and 6 fits remaining; the original total budget remains
unchanged. Tests and the three bounded parity probes are conservatively charged
to 240 of 600 test/probe seconds. No fit is selected on the final streams.
