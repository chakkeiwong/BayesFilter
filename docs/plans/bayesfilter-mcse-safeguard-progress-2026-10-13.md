# Optional MCSE safeguard: implementation and archived-run sensitivity

13 October 2026. The optional `lugsail_max_batch_means` precision policy is
implemented. Seventy-eight focused CPU/XLA tests pass, and reassessing all four
saved full-target MAMS members gives the same first passing sample counts as
the original policy. Fresh coverage/stopping calibration subsequently finished;
the [terminal result](bayesfilter-mcse-safeguard-result-2026-10-13.md) supersedes
the interim calibration status in this execution note. These results do not
promote an estimator default or complete N5/N8.

The controlling plan is
[the comparator/precision continuation](bayesfilter-modern-hmc-comparator-precision-continuation-2026-10-13.md).
All artifacts below are under
`docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/priced-comparison-01/precision-repair-01/`.

## Checked computation

The target quantity is the Monte Carlo variance of each declared posterior
mean. The code computes complete-batch estimates B and raw lugsail estimates
L within each chain, then pools `max(B,L)/(m*m*n)` over the m independent
chains. This is an explicitly different estimator from lugsail. Both
components must be finite and B positive. A finite nonpositive L can be
superseded; a nonfinite component remains unavailable. Reports retain B, L,
the selected estimate and every activation. Original estimators, quantiles,
tuning membership and running GPU policies are unchanged.

The arithmetic agrees with independent batch oracles, including both signs
of serial dependence, discarded remainders and chain-specific cancellation.
Tests exercise `HMCPosteriorAssessmentPolicy` through `assess_posterior` and
the public JSON report. They also preserve the historical calibration
inventory and reject changed repair-profile continuations. The CPU assembly
differs from immutable source `e68cdf1c4` in only `hmc_precision.py`, with
parent and replacement hashes in `.validation-source.json`. The copied
calibration driver and saved numerical closure hashes preserve the exact code
under test. A separate trusted GPU/XLA diagnostic then passed nine parity
cases across both signs of dependence and complete/remainder batches, plus
nonfinite and activation controls. Tensor placement was verified on GPU 1;
memory growth was configured before initialization, and peak TensorFlow
allocator use was 3,035,648 bytes. This closes the tested arithmetic-parity
question, not statistical calibration or default-readiness.

## Saved MAMS sensitivity

Every archived retained look was reassessed using its original quantities,
precision thresholds and posterior criteria. The largest absolute difference
from the original recorded precision calculations was 1.57e-13, within the
declared CPU/GPU float64 parity tolerance. No new transitions or seeds were
generated. The experiment retains the original stopping history.

| Member | Original first passing look | Maximum first passing saved look | Activated chain/quantity pairs at endpoint | Largest MCSE multiplier |
| --- | ---: | ---: | ---: | ---: |
| 000 | 6000 | 6000 | 24/72 | 1.0723 |
| 001 | 10000 | 10000 | 30/72 | 1.0497 |
| 002 | 6000 | 6000 | 29/72 | 1.1039 |
| 003 | 3000 | 3000 | 35/72 | 1.0538 |

These are descriptive sensitivity results for the same four saved members.
Reference comparisons and numerical-health records were preserved, not
reclassified. The safeguard's larger MCSEs do not establish better coverage.
The first member's largest final MCSE/SD is .0199493, close to the unchanged
.02 screen; another realization could produce a different stopping result.

## Evidence and cost

- `checks-01/`: 76 tests passed; 7.451938 seconds including startup.
- `checks-02/`: 78 tests passed; 8.577649 seconds including startup. The added
  controls exercise the actual sequential controller with warm-up exclusion
  and prevent degenerate uncertainty for all-tied paired binary outcomes.
- `binary-uncertainty-check-01/`: 24 reporting/continuation regressions
  passed before the final combined suite; 1.155720 seconds.
- `mams-sensitivity-01/`: all four members, all saved looks, numerical/input
  hashes and complete reports. The gzip archive was checked losslessly.
  Supervisor charge: 30.844274 seconds, from the new CPU reservation.
- `book-check-01/`: the standalone monograph chapter built successfully in
  3.924604 seconds. The new discussion and equation were inspected in the
  rendered PDF (pages 31–32). It is the chapter included by `docs/main.tex`.
- `gpu-check-01/`: nine CPU/GPU arithmetic comparisons and nonfinite controls
  passed; supervisor cost 7.430873 seconds from the local-check reserve.
- `calibration-supervision-01/`: completed two-attempt controller, cap 3000
  seconds; child outputs are `calibration-01/` and, if needed,
  `calibration-02/`. All 768 records are complete. CPU devices only, four chains, fresh seed 73113,
  128 replications per Gaussian/Student-t and correlation condition.

Testing and document builds use the original held 3600-second local-check
reserve. Calibration and archived-draw sensitivity use the separate new
3600-second CPU reserve. No GPU allocation was reused. Only terminal actual
charges may release unused reservations.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Unsupported conclusion |
| --- | --- | --- | --- | --- | --- |
| Keep optional implementation | Numerical oracle, public-policy and GPU parity checks pass | No tested finite/nonfinite or pooling defect | Finite-sample and stopped coverage remain uncertain | Finish fresh calibration | New default or universal conservatism |
| Preserve original MAMS results | Original-policy replay matches; four saved passing looks unchanged under maximum | No corrupted saved tensor or changed source detected | Same draws and one model/seed | Interpret full calibration with uncertainty | Independent replication or coverage validation |
| Continue fixed-HMC baseline | Fresh qualification in progress | No terminal run error observed at this note | Complete delivery within ceiling unknown | Monitor checkpoint progress and settle actual cost | Completed comparison or release |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Tested numerical and replay controls pass; nonfinite components remain unavailable |
| Statistically supported ranking | None |
| Descriptive-only differences | MCSE multipliers and unchanged saved MAMS stopping counts |
| Default-readiness | Not established |
| Next evidence needed | Complete paired fixed/stopped calibration and broader posterior validation |

The strongest alternative explanation for an apparently favorable outcome is
that larger estimated variances merely widen intervals and delay stopping.
The calibration retains those costs. A failure to repair finite-estimate
availability or continued serious undercoverage would require further
estimator/stopping work. Passing these controls would still leave the broader
model, independent-replication and release evidence requirements open.

Execution review also identified and repaired a reporting issue: a generic
paired Student-t interval can collapse when every binary-miss difference is
zero. The terminal analysis will replace those binary intervals with paired
discordance bounds derived from Clopper-Pearson intervals and a union bound.
The already running driver and raw records remain frozen; their old binary
summary intervals are superseded for interpretation. This requires no rerun
and changes no estimator, threshold or stopping decision. Continuous paired
intervals remain approximate and provide no multiplicity-adjusted ranking.
