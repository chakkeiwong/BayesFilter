# SSL-LSTM q=20 seed-B predictive equivalence repair plan (2026-08-08)

## Objective

Repair the mechanical defects in the first q=20 fixed-parameter
predictive-equivalence harness, prospectively calibrate the corrected decision,
and compare the output law at the seed-B posterior-mean plug-in parameter with
the output law at the synthetic true-control parameter.

Parameter-coordinate equality is not a criterion. The 4,000 retained HMC draws
are used only to compute one posterior-mean physical parameter vector. They are
not propagated as a posterior-predictive mixture.

## Research intent ledger

| Item | Frozen statement |
|---|---|
| Main question | Does the posterior-mean plug-in produce a ten-step q=20 output law practically equivalent to the true-control output law? |
| Candidate | One posterior-mean vector from the authenticated seed-B 4-chain x 1,000-draw retained archive. |
| Comparator | The fixed q=20 synthetic generating vector `PRIOR_CENTER`. |
| Expected failure mode | Persistent mean or variance displacement; different complete-path shape/dependence; invalid forecast or uncertainty artifact; calibration unable to distinguish negligible from material alternatives. |
| Primary decision | All 20 simultaneous standardized-mean/log-variance intervals and the independent-bank joint-path MMD interval must pass their frozen practical-equivalence gates. |
| Promotion veto | Any target/archive/transport mismatch, nonfinite path, invalid covariance/interval, missing XLA/CPU provenance, failed calibration validation, or reused calibration/material seed. |
| Continuation veto | Invalid forecast harness; no viable calibration candidate; fresh validation failure; corrupted receipt. A failed material candidate is not a continuation veto for later diagnosis. |
| Repair trigger | Underpowered calibration triggers a larger prospective calibration or different predeclared design, never a post-hoc material threshold. |
| Explanatory diagnostics | Raw and standardized moments, quantiles, covariance, parameter summaries, quadratic MMD, runtime, and calibration family counts. |
| Must not conclude | Parameter identification, absolute posterior correctness, complete mode/tail coverage, model adequacy, NeuTra superiority, or default readiness. |

## Evidence contract

1. The exact target signature is
   `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`.
2. A disjoint true-control scale bank freezes ten horizon centers and sample
   standard deviations. Every later calibration and material path is
   standardized with this receipt through
   `standardize_forecast_paths(..., jit_compile=True)`.
3. MMD bandwidths are `0.5`, `1`, and `2` times the median complete-path
   distance from the first 128 draws per lane of the standardized null-only
   scale bank. The 128-draw subset is inherited from the superseded pilot as a
   bounded quadratic-distance computation and is a convenience hypothesis,
   not a q=20 default. No shifted or material path contributes to scaling or
   bandwidths.
4. The fixed-parameter output paths are iid. Four independent seed lanes retain
   provenance and satisfy the repository's authenticated cross-lane MMD API;
   block length is `1`. No HMC-dependence claim is imported.
5. Feature differences use the correct equal-arm covariance identity by passing
   `2 * influence_left` and `-2 * influence_right` to the pooled covariance
   estimator.
6. Feature margins are standardized mean `0.15` and log-variance `log(1.15)`.
   Provenance: the reviewed July scalar predictive design. Status: transferred
   working hypotheses, not q=20 defaults. Their q=20 operating behavior is
   tested before material data using negligible and material families.
7. The initial repaired-attempt MMD tolerance grid was
   `(0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.012, 0.015, 0.020)`.
   Attempt `r2` rejected that grid before material data were opened. The active
   prospective `r3` grid is
   `(0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.00175, 0.002, 0.00225,
   0.0025, 0.003)`. Provenance: the complete `r2` nomination found a median
   null MMD estimate of `-0.000535`, median shape-only estimate `0.003029`,
   and median standard errors near `0.00165`; this lower grid is pilot-informed
   only. Fresh `r3` nomination and validation seeds decide.
8. Nomination uses 20 replications. Fresh validation uses 60 disjoint
   replications at the selected tolerance. Material and audit seeds are disjoint
   from both.
9. Material status is one of `PASS`, `MATERIAL_DIFFERENCE`,
   `INCONCLUSIVE_UNDERPOWERED`, or `INVALID_HARD_VETO`. A pass alone authorizes
   one fresh-seed audit replication.
10. Stateless seeds use first word `20260808` and disjoint second-word
    namespaces: scale `100000-100003`, canary `110000-110007`, nomination
    attempt `r2` used `200000 + 100*replication + arm*10 + lane`, validation
    namespace `400000`, and Gaussian namespaces `300000`/`500000`. Active
    attempt `r3` uses fresh nomination namespace `900000`, validation namespace
    `1300000`, and Gaussian namespaces `1100000`/`1500000` under the same
    replication layout. Material remains unopened at `700000-700007`; audit
    remains unopened at `800000-800007`. These integers are convenience-chosen
    reproducibility identifiers; they carry no statistical meaning.

## Calibration families

Each replication generates two independent true-control banks and derives the
following right-arm alternatives after applying the frozen scale:

| Family | Role | Construction | Required decision |
|---|---|---|---|
| Identical law | Equivalence | Independent true-control banks | `PASS` |
| Negligible mean | Equivalence | Raw output shift `+0.05` | `PASS` |
| Negligible variance | Equivalence | Variance ratio `1.05` about the calibration center | `PASS` |
| Material mean | Material | Raw output shift `+0.20` | `MATERIAL_DIFFERENCE` |
| Material variance | Material | Variance ratio `1.25` about the calibration center | `MATERIAL_DIFFERENCE` |
| Shape-only skew | Material | Independent correlated standard-Gaussian paths with the q=20 scale-bank correlation; right paths use `(Z + 0.35*(Z^2-1))/sqrt(1+2*0.35^2)`, which has the same population marginal means and variances | `MATERIAL_DIFFERENCE` through the path statistic |

The full combined feature-plus-MMD decision is applied to every family. MMD is
not required to detect an alternative already detected by a co-primary feature.

## Calibration gates

Nomination candidate gate over 20 replications:

- each equivalence family: at least 16 `PASS`, at most one
  `MATERIAL_DIFFERENCE`, no invalid result;
- each material family: at least 16 `MATERIAL_DIFFERENCE`, at most one `PASS`,
  no invalid result;
- select the smallest candidate satisfying every family.

Fresh validation gate over 60 replications at the frozen candidate:

- each equivalence family: at least 54 `PASS`, at most one
  `MATERIAL_DIFFERENCE`;
- each material family: at least 54 `MATERIAL_DIFFERENCE`, at most one `PASS`;
- at least 54/60 simultaneous feature intervals cover the known feature truth
  for the null, mean-shift, and analytic Gaussian-skew families; variance-family
  coverage is explanatory because its centering uses the independently
  estimated q=20 calibration center;
- no invalid result.

The 54/60 requirement is derived from the reviewed 90% operating target; an
all-success 60-replication row has a one-sided 95% exact lower bound above 0.95,
while 54/60 is the predeclared count gate rather than a claim of exact power.

## Default and assumption audit

| Choice | Provenance/status | Failure mode | Early diagnostic |
|---|---|---|---|
| Posterior mean plug-in | User-requested functional; candidate | Nonlinear mean vector may not represent the posterior | Median remains descriptive only; do not switch summaries post hoc |
| Scale-bank size: 2,048 draws/lane x 2 replications | Convenience hypothesis, larger than prior diagnostic | Noisy horizon scales | Finite positive scale gate and null calibration |
| Bandwidth subset: first 128 draws/lane | Inherited bounded-compute pilot choice | Noisy median path distance | Fresh nomination and validation over the complete decision |
| Initial calibration size: 1,024 draws/lane x 2 replications | Pilot-informed bounded `r2` design; rejected | Underpowered MMD and feature equivalence intervals | Preserved `r2` nomination artifact |
| Active calibration size: 16,384 draws/lane x 2 replications | `r2`-derived 16x prospective power repair | Runtime or remaining MMD overlap | Fresh `r3` nomination and 60-replication validation stop closed |
| Material size: 2,048 draws/lane x 2 replications | Pilot-informed bounded design | Wide intervals | `INCONCLUSIVE_UNDERPOWERED`, never equivalence by non-rejection |
| Four iid lanes | Repository MMD API plus provenance | Lane grouping mistaken for HMC dependence | Explicit iid status and block length `1` |
| Scale floor `1e-8` | Convenience numerical guard, far below observed output scales | Floor use could alter the target statistic | Fail if any scale is below the floor; floor use is forbidden |
| Feature margins | Reviewed scalar design; transferred hypothesis | Wrong q=20 practical boundary | Negligible/material q=20 calibration families |
| Dense MMD tolerance grid | Derived from r5 pilot before fresh calibration | Overfitting pilot | Fresh nomination and validation seeds; material remains unopened |
| CPU/XLA | Reviewed reference execution exception | Backend mismatch or excessive cost | GPU hidden before import; XLA trace/device manifest; no production claim |

## Skeptical pre-execution audit

| Audit question | Finding and repair |
|---|---|
| Wrong baseline? | Repaired. The two arms are exactly posterior-mean plug-in and synthetic true-control output laws. Parameter distance is explanatory only. |
| Proxy promoted? | Repaired. Raw moments and plots cannot pass; simultaneous intervals plus MMD decide. |
| Missing stop conditions? | Repaired. Nomination, validation, material, and audit each have explicit fail-closed statuses and wall caps. |
| Unfair comparison? | Repaired. Both arms share target/operator/shape/dtype but use independent seed banks. |
| Hidden assumptions? | Exposed: iid lanes, scale source, transferred margins, tolerance-grid provenance, and material sample size. |
| Stale context? | r5 is pilot evidence only. Its paths, candidate outcome, and tolerance are not reused as validation or material evidence. |
| Environment mismatch? | CPU-only is explicitly a reviewed reference exception; TensorFlow/XLA remains mandatory. |
| Artifact mismatch? | Every stage has a unique output root, source/receipt hashes, seeds, and replayable command. |
| Could the run pass while misleading? | Yes if scaling used material data, if MMD alone was calibrated on a mean shift, or if two-arm covariance was understated. The repaired runner checks all three invariants. |
| Could it fail for tuning rather than science? | Yes. Calibration failure is classified separately and closes material data without rejecting the candidate or research direction. |

Audit verdict: **PASS AFTER MECHANICAL REPAIRS**. No material or audit command
may run until the fresh calibration validation receipt passes.

## Prospective power repair after r2 nomination

Attempt `r2` completed 20 nomination replications with zero invalid decisions
and 20/20 known-truth feature coverage in every family, but no tolerance passed.
At tolerance `0.008`, identical and negligible-mean families passed 20/20,
while negligible variance passed 1/20 and shape-only skew produced only 1/20
material decisions. The negligible-variance failure was feature-interval width;
the shape failure was MMD-interval width. This is calibration underpower, not a
target, forecast, covariance, or artifact invalidity finding.

The active `r3` repair increases calibration draws by 16x to 16,384 per lane.
The factor is a pilot-informed conservative hypothesis: dividing the `r2`
80th-percentile MMD half-widths by four projects values near `0.00098`, while
the median observed shape effect was `0.003029`. This projection is not power
evidence and does not guarantee admission; it justifies the fresh prospective
run. The family definitions, margins, scale bank, bandwidth rule, 20/60 count
gates, target, hardware class, and unopened material/audit seeds remain fixed.
The user-authorized 20,000-second headroom bounds this repair. `r3` nomination
has a 3,600-second cap and validation has a 10,800-second cap; all other caps
remain unchanged. A second nomination failure closes the campaign without
opening material data.

## Execution phases and caps

1. `canary`: 32 draws/lane; mechanics only; cap 900 s.
2. `scale`: disjoint true-control scale/bandwidth bank; cap 900 s.
3. `nominate`: 20 fresh calibration replications; active `r3` cap 3,600 s.
4. `validate`: 60 fresh calibration replications at the frozen candidate; cap
   10,800 s for active `r3`.
5. `material`: posterior mean versus true control; cap 900 s.
6. `audit`: fresh material seeds, only after a material pass; cap 900 s.

Artifacts write under
`docs/plans/artifacts/ssl-lstm-q20-seed-b-predictive-equivalence-repair-2026-08-08/`
with a unique versioned attempt root. Existing artifacts are never overwritten.

## Interpretation rules

- Calibration failure means the harness did not answer the material question.
- Material `MATERIAL_DIFFERENCE` rejects only the current posterior-mean plug-in
  under this target and frozen design.
- Material `INCONCLUSIVE_UNDERPOWERED` is not equivalence.
- A material and audit `PASS` support bounded replicated predictive-functional
  equivalence only. They do not prove posterior correctness or parameter
  recovery.
