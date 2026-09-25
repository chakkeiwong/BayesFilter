# C2 Mixture-UKF/APF Phase 5B: Fixed-Map RBF Representation Pilot

Date: 2026-09-04  
Governing plan: `docs/plans/bayesfilter-c2-mixture-ukf-apf-master-program-2026-09-02.md`  
Status: `EXECUTED_PASS_RBF_HYBRID_READY`  
Scope: one-step fixed-map Gaussian-reference RBF diagnostic on the C2 fixture  
Classification: `extension_or_invention_candidate_diagnostic_only`

## Why this phase follows Phase 5A

The Phase 5A Hermite pilot passed all engineering and numerical validity
checks, but its fixed-map representation showed a tail/conditioning failure:
the degree-10 exact Gram normalizer was far from the independent predictive
mixture estimate and the shell residuals did not improve monotonically with
degree.  That result rejects no generic algorithm.  It opens the next
predeclared representation arm while keeping the map, target, rows, and
normalizer comparator unchanged.

During the Phase 5A close audit, the active worktree was also found to be
missing the reviewed C2 model and Hermite-engine files needed by the recorded
driver.  The exact files were restored from the preserved `b3eaa7a9` C2
snapshot before this phase.  A current-tree CPU replay reproduced the Phase
5A `Z_T` and all three degree records to roundoff.  That replay is a
provenance/integrity check, not a replacement for the authoritative GPU
artifact.

## Research question and boundary

Does a fixed separable Gaussian RBF basis with an explicit constant channel
reduce the fixed-map tail and Gram-normalizer discrepancy observed with the
polynomial Hermite basis, on the same exact finite target?

This phase is a representation diagnostic.  It does not test proposal ESS,
recursive filtering, posterior correctness, HMC readiness, analytical total
gradients, or a production/default choice.  The C2 model is a fixture only;
no width or center setting may be promoted as model-independent evidence.

## Exact target and reference measure

Use the Phase 5 map and the exact one-step C2 target without modification:

\[
  \gamma(x)=\sum_j w_j f(x\mid x_j)g(y_1\mid x),\qquad x=m+Lu,
\]

and, with the standard-normal probability measure
\(d\mu(u)=\eta_4(u)\,du\),

\[
  h_\star(u)=\gamma(m+Lu)|\det L|/\eta_4(u),
  \qquad s_\star(u)=\sqrt{h_\star(u)}.
\]

The fitter approximates \(s_\star\) on disjoint reference-normal rows.  The
reported fitted normalizer is the exact RBF Gram contraction

\[
  Z_H=\int h(u)^2\,d\mu(u),
\]

while the primary comparator is the independent predictive-mixture identity

\[
  Z_T=\int\gamma(x)\,dx
      =\mathbb E_{J\sim w,\;X\sim f(\cdot\mid x_J)}[g(y_1\mid X)].
\]

The Gaussian-reference \(\gamma/\eta_4\) estimate remains an explanatory
tail-variance diagnostic only.  It is not the promotion comparator.

## RBF basis and analytic contractions

For centers \(c_i\in\mathbb R\), widths \(s_i>0\), and an optional constant
channel, define

\[
  \phi_i(u)=\exp\left[-\frac{(u-c_i)^2}{2s_i^2}\right],
  \qquad \phi_0(u)=1\ \text{when the constant channel is enabled}.
\]

Under \(\eta_1(u)du\), put

\[
  A_{ij}=1+s_i^{-2}+s_j^{-2},\qquad
  b_{ij}=c_i s_i^{-2}+c_j s_j^{-2}.
\]

The RBF-RBF mass entry is

\[
  M_{ij}=A_{ij}^{-1/2}
  \exp\left[-\frac12\left(c_i^2s_i^{-2}+c_j^2s_j^{-2}\right)
             +\frac{b_{ij}^2}{2A_{ij}}\right].
\]

The constant-RBF integral is

\[
  I_i=(1+s_i^{-2})^{-1/2}
       \exp\left[-\frac{c_i^2s_i^{-2}}{2(1+s_i^{-2})}\right]
      =\frac{s_i}{\sqrt{1+s_i^2}}
       \exp\left[-\frac{c_i^2}{2(1+s_i^2)}\right].
\]

With a constant channel, the complete one-dimensional mass matrix and
integral vector are

\[
  M=\begin{bmatrix}1&I^\mathsf T\\I&M_{\rm rbf}\end{bmatrix},
  \qquad I_{\rm all}=\begin{bmatrix}1\\I\end{bmatrix}.
\]

The implementation must use these closed forms under
`MassMeasure.REFERENCE_MEASURE`, symmetrize the returned matrix, and reject
other measures rather than silently changing the target.  The derivative
\(\phi_i'(u)=-(u-c_i)\phi_i(u)/s_i^2\) is implemented for future gradient
parity, but this pilot makes no adaptive-gradient claim.

## Frozen arms

The centers are the fixed normalized UKF-offset grid
\((-2,-1,0,1,2)\) on every coordinate.  Each arm includes exactly one
constant channel and changes only the common width:

| Arm | centers | width | purpose |
| --- | --- | ---: | --- |
| `rbf_w075` | `(-2,-1,0,1,2)` | `0.75` | local/narrow channel hypothesis |
| `rbf_w150` | `(-2,-1,0,1,2)` | `1.50` | moderate overlap hypothesis |
| `rbf_w300` | `(-2,-1,0,1,2)` | `3.00` | broad-tail nonconstant hypothesis |

These are diagnostic hypotheses, not tuned defaults.  The arms are compared
one factor at a time; no Cartesian center/width sweep is permitted.  The
Hermite map, target, TT rank 2, two ALS sweeps, ridge `1e-8`, and all bank
seeds are unchanged from Phase 5A.  The same fixed training/holdout/audit
partition is used so representation comparisons are paired, but no holdout
value selects an arm.

The Hermite-plus-RBF hybrid is deliberately deferred.  It requires analytic
Hermite/RBF cross-Gram entries and a separate duplicate-constant policy.  A
finite RBF-only arm must first pass the mass and target contracts; otherwise
implementing the hybrid would confound a basis algebra defect with a fit
result.

## Evidence contract

| Field | Declaration |
| --- | --- |
| Question | fixed-map RBF channels reduce representation error without changing the target |
| Exact comparator | independent predictive-mixture `Z_T` and exact C2 transition/observation calls |
| Primary validity gate | every arm has finite target, finite fit, symmetric positive RBF mass, valid map, and condition below `1e14` |
| Representation screen | held-out RMS, central/shell RMS, `log Z_H-log Z_T`, mass condition, fitted Gram condition |
| Promotion criterion | none in this pilot; screens are descriptive nominations only |
| Hard vetoes | missing source, target/map mismatch, analytic-mass parity failure, nonfinite value, non-SPD mass, invalid fitter/artifact |
| Explanatory diagnostics | audit RMS, rank, ESS carried from Phase 5 map, reference-tail standard error, elapsed time |
| Nonclaims | no arm ranking, proposal-efficiency claim, recursive claim, posterior/HMC claim, gradient claim, or default readiness |
| Artifact | fresh `phase5b-rbf-attempt01/` with manifest, records, result, command, close note, MathDevMCP and Lean sidecars |

## Default and assumption audit

| Choice | Provenance and role | Failure mode | Earliest check | Status |
| --- | --- | --- | --- | --- |
| Gaussian reference and Phase 5 map | existing fixed-map contract | measure/map mismatch | exact map and target parity | reviewed fixed baseline |
| centers `(-2,-1,0,1,2)` | normalized UKF-offset diagnostic convention | target mode lies outside grid | shell residual and center coverage | hypothesis |
| widths `.75, 1.5, 3.0` | one-factor tail/locality ladder | overlap or vanishing tails makes mass ill-conditioned | eigenvalues and condition before fit | hypotheses |
| one constant channel | broad-tail and normalization channel | duplicate/near-duplicate direction | mass eigenvalue and condition | reviewed requirement |
| rank 2, two ALS sweeps, ridge `1e-8` | paired Phase 5A fitter settings | underfit or ridge bias | fit/holdout and condition records | frozen comparison setting |
| same disjoint banks as Hermite | paired representation comparison | finite-bank noise | hashes and predictive standard error | fixed diagnostic design |
| eager ALS, XLA map/target kernels | existing fitter contract and repository backend policy | retracing or backend mismatch | manifest and CPU/GPU parity | diagnostic execution lane |

The ridge is a Class-C numerical choice.  It remains frozen for this paired
diagnostic and is not selected from the holdout bank.  Any ridge calibration
requires a new plan and a non-harm check.

## Protocol

1. Verify that the restored C2 model, Phase 5 map, exact target kernel, and
   Phase 5 bank seeds are available.  Refuse an output root that already
   exists.
2. Generate the same train, holdout, Gaussian-reference audit, and independent
   predictive audit banks as Phase 5A.  Record hashes and disjoint seeds.
3. Construct each `RBFBasis1D`, evaluate its analytic mass/integral matrices,
   and compare them with an independent high-order Gaussian quadrature check
   before fitting.  Reject the arm on any nonfinite, asymmetric, or
   non-positive eigenvalue result.
4. Evaluate the exact mapped target and common shift.  Fit the square-root
   target only on the training bank using the existing fixed-design ALS API.
5. Evaluate holdout central/shell residuals, the exact RBF Gram contraction,
   the independent predictive `Z_T`, and the Gaussian-reference tail
   diagnostic.  Do not use holdout or audit rows to update cores or choose an
   arm.
6. Emit one immutable arm record, a manifest with source hashes and
   environment/device policy, raw records, result Markdown/JSON, and the
   actual command.  Preserve every failed smoke directory.

## Required mathematical and executable checks

- MathDevMCP structural checks for the RBF mass, integral, and derivative
  identities, with the returned status preserved as structural evidence only.
- A scoped Lean file proving the algebraic symmetry of the completed-square
  exponent and the constant-channel block identities; it does not certify
  floating-point quadrature, SPD, or ALS behavior.
- Focused CPU tests for analytic-vs-quadrature mass/integral, derivative
  finite differences, constant-channel normalization, invalid-measure refusal,
  and `ProductBasis`/TT integration.
- Current-tree Phase 5A replay and the existing Phase 5/Phase 4 regressions.
- One bounded GPU/XLA run with memory growth configured before logical-device
  initialization.  CPU-only output is a reference diagnostic only.

## Skeptical plan audit

Disposition before execution: `PASS_FOR_BOUNDED_PHASE5B_RBF_DIAGNOSTIC`.

The plan survives the required audit because the exact C2 target and map are
unchanged, the predictive-mixture normalizer is independent of the fitted
basis, and all RBF algebra is checked before any fit.  The width ladder is
small and predeclared rather than selected after looking at holdout values.
The constant channel prevents an empty-tail representation, while the mass
eigenvalue/condition veto exposes collinearity instead of hiding it.  The
phase does not promote a width, and a poor RBF result only opens the deferred
hybrid or reference-law investigation.  A target/map mismatch, failed
quadrature identity, nonfinite mass/fit, corrupted artifact, or exhausted
budget remains a continuation veto.

### Pre-mortem

| Misleading result | Distinguishing check | Action |
| --- | --- | --- |
| `Z_H` agrees because the same rows were reused | predictive audit bank hash differs from train/holdout/reference banks | invalidate and rerun fresh |
| broad RBF appears accurate only through a nearly singular mass | eigenvalue and condition recorded before fit | hard-veto the arm; do not interpret residuals |
| quadrature agrees due to insufficient order | repeat at two declared orders and compare analytic values | repair formula/test before fitting |
| RBF residual falls but target normalizer is wrong | independent predictive `Z_T` and shell residual | retain as candidate failure; do not promote |
| one width wins by Monte Carlo noise | paired banks, standard errors, no ranking claim | open replicated plan only if needed |
| restored source differs from the historical run | current-tree Phase 5A replay and source hashes | stop representation interpretation until provenance is resolved |

## Budget and commands

Budget: one CPU regression and one GPU/XLA attempt, each at most one hour;
localized harness repairs may retry under the same contract in a fresh output
directory.  No recursive horizon, Student TT reference, or large-N proposal
run is opened by this phase.

Focused CPU command:

```text
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/mpl-c2-phase5b-rbf-tests \
/home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu \
python -m pytest -q tests/highdim/test_rbf_basis_tf.py \
tests/highdim/test_c2_hermite_basis.py tests/highdim/test_c2_mixture_ukf_apf_phase5a.py \
tests/highdim/test_recursive_moment_map_tf.py tests/highdim/test_c2_mixture_ukf_apf_phase4_repair.py
```

GPU/XLA command:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/mpl-c2-phase5b-rbf \
/home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu \
python docs/benchmarks/run_c2_mixture_ukf_apf_phase5b_rbf_20260904.py \
--output-root docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase5b-rbf-attempt01
```

## Close and refresh rules

At close, classify engineering correctness, numerical validity, and
scientific interpretation separately.  If all hard checks pass, report the
width records descriptively and open the hybrid only under a new bounded plan;
do not call an observed lower residual an improvement without replicated
uncertainty evidence.  If the mass formula or target wiring fails, repair that
local implementation and rerun the focused test before interpreting any arm.
If all RBF arms are valid but poor, retain the fixed-map Hermite result and
refresh the hybrid plan with analytic cross-Gram obligations.  If no finite
normalized representation remains or the exact target/map cannot be kept
unchanged, stop with the precise continuation veto.

Every close must preserve the failed attempt, actual command, source hashes,
budget, result/decision tables, inference-status table, MathDevMCP output,
Lean output, and a post-run red-team note.

## Execution close (2026-09-04)

The authoritative paired-seed GPU/XLA run completed in
`docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase5b-rbf-attempt03/`.
It passed the exact target/map, analytic-mass, quadrature, finite/SPD, fitter,
and artifact gates.  The focused CPU suite reported `22 passed, 2 warnings`.
The close note is
`phase5b-rbf-attempt03/phase5b-close-20260904.md`; the formal audit and raw
MathDevMCP/Lean sidecars are in the same directory.

The first quadrature smoke, first GPU lifecycle attempt, and unpaired-seed
attempt are preserved and classified in the close note.  Those repairs kept
the target, map, data partitions, arm set, budget, and interpretation rules
fixed.  No width is promoted.  The phase disposition is
`PASS_PHASE5B_RBF_MECHANICS` and the refreshed continuation is
`CONTINUE_HYBRID_DIAGNOSTIC`.
