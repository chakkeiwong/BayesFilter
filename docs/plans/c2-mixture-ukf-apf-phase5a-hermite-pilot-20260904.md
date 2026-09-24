# C2 Mixture-UKF/APF Phase 5A: Fixed-Map Hermite Representation Pilot

Date: 2026-09-04  
Status: `EXECUTED_PASS_HERMITE_MECHANICS_RBF_READY`  
Parent program: [`bayesfilter-c2-mixture-ukf-apf-master-program-2026-09-02.md`](bayesfilter-c2-mixture-ukf-apf-master-program-2026-09-02.md)  
Scope: one-step fixed-map representation diagnostic on the C2 fixture  
Classification: `extension_or_invention_candidate_diagnostic_only`

## Research question

After the generic recursive lagged moment map passes its independent linear
oracle and C2 call-chain checks, is the remaining discrepancy explained by the
fixed Gaussian-reference Hermite representation, rather than by the map or the
exact C2 target?  The pilot measures this question on one exact finite target
with disjoint training, holdout, and audit rows.

This is a representation diagnostic.  It is not a proposal-efficiency,
posterior-correctness, pseudo-marginal, HMC, or default-readiness experiment.
The C2 model is a test fixture; no model-specific setting may be promoted.

## Evidence contract

### Target and coordinate identity

Let the carried cloud at the first recursive C2 transition be
\(\{(x_j,w_j)\}_{j=1}^N\), with \(w_j\geq0\) and \(\sum_jw_j=1\).  The exact
one-step unnormalised target is

\[
  \gamma(x)=\sum_{j=1}^N w_j f(x\mid x_j)g(y_1\mid x).
\]

The Phase 5 map is \(x=m+Lu\), where \(L\) is the lower Cholesky factor of
the empirical predicted covariance.  With \(\eta_4\) the standard-normal
density, the exact target with respect to the reference probability measure
\(\mu(du)=\eta_4(u)\,du\) is

\[
 h_\star(u)=\gamma(m+Lu)\,|\det L|/\eta_4(u).
\]

The fitted square-root TT \(h\) is compared to \(\sqrt{h_\star}\).  Therefore
\(Z_T=\int\gamma(x)\,dx=\int h_\star(u)\,d\mu(u)\), while the fitted
normalizer is the exact Gram contraction \(Z_H=\int h(u)^2\,d\mu(u)\).
No fitted value is substituted into the exact target evaluation.  The primary
finite-bank estimate of \(Z_T\) samples the normalized predictive mixture
\(\sum_jw_j f(\cdot\mid x_j)\) and averages the exact likelihood \(g(y_1\mid
x)\).  A separate Gaussian-reference estimate of \(\int h_\star d\mu\) is
also recorded, but is explanatory only because its tail variance can be very
large for this observation model.

### Primary diagnostics

The pilot records, for degrees 6, 8, and 10 at fixed TT rank 2:

* weighted training RMS and unweighted/weighted held-out RMS of the square-root
  amplitude;
* central (`max |u_i| < 2`) and tail-shell (`max |u_i| >= 2`) held-out RMS;
* direct \(Z_T\) from an independent prior-predictive mixture audit bank and
  exact Hermite Gram \(Z_H\), with \(\log Z_H-\log Z_T\);
* the Gaussian-reference importance estimate and its relative standard error,
  explicitly as a tail-variance diagnostic;
* the maximum scaled augmented-system condition number over ALS updates;
* realized rank and finite/positive checks; and
* the fixed-map condition and Cholesky margin used to construct the target.

The direct \(Z_T\) estimate is a fixed-bank Monte Carlo diagnostic, not an
exact oracle.  Its sampling uncertainty is reported and prevents a normalizer
comparison from being treated as a proof.

### Pass, veto, and interpretation roles

| Diagnostic | Role | Rule |
| --- | --- | --- |
| exact C2 transition/observation calls, finite target values, SPD map | hard validity veto | any failure invalidates the attempt |
| complete train/holdout/audit banks and source hashes | hard artifact veto | any missing or mismatched record invalidates the attempt |
| fitter finite status and condition below `1e14` | hard numerical veto | candidate is not interpretable if violated |
| held-out RMS, shell RMS, `log Z_H-log Z_T` | representation promotion screen | descriptive nomination only in this pilot; no fixed threshold is promoted |
| Gram conditioning, rank, ESS of the audit bank | explanatory diagnostic | never a tuning target or correctness proof |

The pilot passes its engineering screen only when all hard vetoes pass for all
three degrees.  A poor representation metric is a candidate failure and opens
the predeclared RBF/hybrid follow-on; it is not a continuation veto for the
generic map program.  No ranking among degrees is claimed from this one bank.

## Fixed assumptions and provenance

| Choice | Provenance and role | Failure mode | Earliest check | Status |
| --- | --- | --- | --- | --- |
| C2 fixture and exact model factors | Phase 5 source contract | adapter mismatch | transition/process/stationary parity | reviewed baseline |
| `N=128` carried cloud and Phase 5 seed | Phase 5 bounded probe | empirical map noise | map SPD, ESS, repeatable hashes | fixed diagnostic baseline |
| first recursive transition (`t=1`) | isolates representation from multi-step feedback | not representative of later recursion | later Phase 5B recursive check | scope restriction |
| Hermite degrees 6, 8, 10 | parent program recommendation | polynomial tail/conditioning failure | shell residual and Gram condition | hypothesis ladder |
| TT rank 2, two forward sweeps, ridge `1e-8` | fixed-design fitter pilot; no validation tuning | underfitting or ridge bias | held-out residual and ridge metadata | convenience hypothesis, not default |
| 512 train, 512 holdout, 4096 Gaussian-reference rows plus 4096 prior-predictive rows | disjoint stateless banks | Monte Carlo error or reference-tail variance | bank hashes and both standard errors | bounded pilot |
| standard-normal reference measure | existing `HermiteBasis1D` contract | measure mismatch | `MeasureConvention` and identity mass | reviewed fixed measure |
| float64 TensorFlow, XLA-enabled map only | repository numerical policy | backend mismatch | eager/XLA map parity and manifest | required execution mode |
| eager fixed-design ALS | existing fitter API is diagnostic and records its branch | no claim of compiled production fitter | fitter status and source path | diagnostic implementation |

The ridge is a Class-C numerical choice.  It is frozen for comparability and
is not silently promoted.  A later ridge calibration must be a separate plan
with a non-harm criterion.

## Protocol

1. Reconstruct the deterministic Phase 5 C2 cloud and the same lagged map from
   the fixture, verifying the model transition, process covariance, stationary
   covariance, map SPD, and map forward/inverse residual.
2. Generate disjoint stateless standard-normal banks for training (512),
   holdout (512), and Gaussian-reference audit (4096), plus a distinct
   prior-predictive mixture audit bank (4096).  Record all seeds and hashes.
3. Evaluate the exact finite-mixture target \(\gamma\), map Jacobian, and
   reference density on each bank.  Form the square-root target using a
   common frozen log shift; reject any nonfinite value.
4. For each degree in `{6,8,10}`, construct the existing product Hermite basis
   and fixed-rank TT initial cores.  Fit only on the training bank with the
   existing objective-preserving scaled augmented ridge ALS route.  The
   holdout bank is never used to update cores.
5. Evaluate the fitted TT on holdout and reference-audit rows.  Contract its
   squared amplitude with the Hermite identity mass matrices to obtain `Z_H`.
   Estimate primary `Z_T` from the independent prior-predictive mixture bank
   by averaging the exact observation likelihood; retain the reference-bank
   estimate only to expose tail variance.
6. Emit one immutable record per degree plus a manifest, raw records, result
   JSON/Markdown, command, source hashes, environment, device/memory policy,
   and the exact plan hash used for the run.  Refuse to overwrite an existing
   output directory.

## Skeptical plan audit (before execution)

Disposition: `PASS_FOR_BOUNDED_PHASE5A_HERMITE_DIAGNOSTIC`.

* The comparator is an exact finite C2 factor evaluation and an independent
  prior-predictive audit estimate, not a TT-derived target.
* The primary normalizer estimate samples the predictive mixture, so it does
  not inherit the avoidable `1/eta` tail variance.  The reference estimate is
  explicitly noisy and cannot silently become a promotion criterion.
* Training, holdout, and audit rows are disjoint and stateless, so residuals
  cannot be explained by row reuse.
* The map is frozen and independently checked; representation error is not
  conflated with recursive map error.
* The fixed rank, ridge, sweep count, and row counts are predeclared.  No
  degree or ridge is selected on the holdout or audit bank.
* The plan has a continuation rule: a valid but poor Hermite result opens the
  RBF/hybrid diagnostic; only invalid target/map/artifact mechanics stop the
  program.
* The pilot does not answer whether a proposal has adequate ESS, whether the
  filter is posterior-correct, or whether an analytical total gradient is
  implemented.

### Pre-mortem

| Misleading outcome | Distinguishing check | Action |
| --- | --- | --- |
| `Z_H` appears accurate because both estimates use the same rows | independent prior-predictive audit bank and bank hashes | invalidate and rerun with fresh disjoint banks |
| reference-bank `Z_T` is dominated by one tail row | prior-predictive estimate and its standard error | use the prior-predictive estimate as primary; report the reference estimate as tail diagnostic |
| high-degree Hermite residual is blamed on the map | linear/map checks already passed; compare all degrees on the same map | open RBF/hybrid phase, do not alter map |
| ALS reports a finite fit despite an ill-conditioned design | scaled condition and raw/holdout residual records | hard veto the affected degree |
| tail residual is hidden by central RMS | explicit shell partition and shell counts | keep tail result as the representation diagnosis |
| one degree looks better due to one random bank | fixed paired banks plus no ranking claim | use additional seeded replication only in a new plan |

## Budget and commands

Budget: one GPU/XLA attempt and one CPU regression, each at most 1 hour;
localized harness repair may retry under the same contract in a fresh attempt
directory.  Changing the map, target, measure, rank, or budget requires a new
plan.

Focused CPU regression:

```text
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu \
  python -m pytest -q tests/highdim/test_recursive_moment_map_tf.py \
  tests/highdim/test_c2_mixture_ukf_apf_phase4_repair.py
```

Bounded GPU/XLA pilot:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/mpl-c2-phase5a-hermite \
  /home/chakwong/anaconda3/bin/conda run --no-capture-output -n tftwogpu \
  python docs/benchmarks/run_c2_mixture_ukf_apf_phase5a_hermite_20260904.py \
  --output-root docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase5a-hermite-attempt01
```

## Close and refresh rules

At close, write a decision table and an inference-status table.  State
separately whether the attempt validated the harness, the exact target, the
map, the fitter, or only the candidate representation.  If all hard vetoes
pass but every degree has unacceptable tail or normalizer residual, refresh
the next phase with fixed-map RBF and Hermite-plus-RBF arms.  If one degree is
descriptively promising, do not promote it; open a separate replicated and
recursive validation plan.  A Student reference measure remains deferred until
the Gaussian-reference representation question is answered.

## Execution close (2026-09-04)

The authoritative GPU/XLA attempt completed in
`docs/benchmarks/artifacts/c2_mixture_ukf_apf_20260902/phase5a-hermite-attempt01/`.
All source, map, target, bank, fitter, finite-value, and artifact checks
passed.  The focused CPU regression reported `11 passed, 2 warnings`.

The first CPU smoke exposed and repaired two implementation defects before the
authoritative run: a Gram-contraction einsum label and a normalizer branch that
used the square-root target instead of `log_h`.  A third repair changed the
primary `Z_T` audit from a high-variance `gamma/eta` reference estimate to an
independent predictive-mixture estimate.  Each repair preserved the target,
map, rows, and budget and is recorded in the close note; failed smoke
directories remain preserved.

The descriptive degree records were:

| degree | holdout RMS | shell RMS | `log Z_H-log Z_T` | scaled condition max |
| ---: | ---: | ---: | ---: | ---: |
| 6 | `0.06287` | `0.11159` | `0.00986` | `2.55e4` |
| 8 | `0.04273` | `0.07688` | `0.1531` | `1.85e5` |
| 10 | `0.05347` | `0.10621` | `2.882` | `1.70e6` |

These values are descriptive only.  Degree 10 is not promoted because its
exact Gram normalizer is unstable relative to the independent predictive
estimate.  The phase result is `PASS_PHASE5A_HERMITE_MECHANICS` with
`CONTINUE_RBF_HYBRID_DIAGNOSTIC`; the next arm must keep the fixed map, exact
C2 target, predictive-mixture `Z_T`, and disjoint-bank protocol unchanged.
