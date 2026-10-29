# Structural Model B diagnostic result — 2026-10-26

Closed at the owner's request; the
[final report](../../structural-score-final-report-2026-10-26.md) records the
completed findings and the remaining accuracy question. No further run is
scheduled in this investigation.

The reviewed diagnostic plan is complete. The tests found no demonstrated
error in the analytical derivative at the inspected truth point. They did
identify a substantial contribution from the reset to the T2 model-score
discrepancy, finite-design variation, an unresolved persistence-score gap
at T50, and several engineering/reporting weaknesses. Production filtering
and HMC were not modified or rerun.

The model is `m_t=rho*m_(t-1)+sigma*epsilon_t`,
`k_t=.55*k_(t-1)+beta*tanh(m_t)`,
`y_t=m_t+k_t+.30*eta_t`, at truth `(rho,sigma,beta)=(.70,.25,.80)`.
Scores below are with respect to `(atanh(rho),log(sigma),log(beta))`,
not physical parameters. The initial law is the campaign's independent
`m0~N(0,.25), k0~N(0,.20)`. This diagnostic holds the data fixed.

## Derivative consistency and precision

`StructuralTarget.likelihood` calls the shared analytical Jet recurrence.
Central differences perturb q while keeping observations, initial draws,
innovations, parent choices and mixture choices fixed. The returned score
is a derivative of that finite numerical likelihood, which differs from
the exact-model likelihood. The complete call chain and reference-score
derivation are in [source-audit.md](source-audit.md).

| Horizon / execution | Largest component difference at h=.01 | At h=.001 | At h=.00003 | Interpretation |
|---|---:|---:|---:|---|
| T1 FP32 GPU/XLA | 2.20e-5 | 9.72e-5 | 7.21e-3 | Small-step subtraction is roundoff sensitive |
| T2 FP32 GPU/XLA | 3.04e-5 | 3.51e-4 | 1.14e-2 | Same pattern |
| T50 FP32 GPU/XLA | 1.30e-3 | 1.06e-2 | 3.64e-1 | Long-horizon finite differences are particularly sensitive |
| T1 FP64 CPU graph | Not run | 6.04e-8 | 4.05e-11 | Tight derivative agreement |
| T2 FP64 CPU graph | Not run | 6.81e-8 | 3.58e-11 | Tight derivative agreement |

Every retained T50 baseline row completed 50 steps; every recorded
validity check passed, and no conditional FP64 reset fallback fired.
All T50 finite-difference perturbations were valid. The T1/T2 and fresh
design endpoints were also valid; their original baseline validity traces
were not retained, so those traces are not retrospectively claimed.
The reset-isolation rows have zero fallback and all density/reset checks
pass.

FP64 minus FP32 changes T1/T2 log likelihood by `1.86e-7 / 1.02e-6`
and the largest score component by `4.83e-8 / 1.19e-7`. This is a joint
precision/backend check: FP64 runs on CPU without XLA and uses uniform
reset, while production FP32 uses the conditional fallback policy. The
healthy uniform branch is mathematically the same; these results do not
measure T50 FP64 sensitivity or certify the whole posterior domain.
FP32 remains the production default. A fixed very small FD step would
give a misleading diagnosis here.

Evidence: `attempts/diagnostic-11-final/results.json`,
`attempts/diagnostic-13-fp64-cpu/results.json`, and the complete retained
T50 ladder in `attempts/diagnostic-14-t50/results.json`.

## The short-horizon discrepancy and the reset

T2 path quadrature targets the exact nonlinear model. Refining order
28 to 32 changes its log likelihood by `5.47e-9`; the largest score
change is about `1.23e-7`. The score is itself checked over FD steps
3e-5, 1e-4 and 3e-4. This is a highly refined numerical reference,
not a closed-form or rigorously bounded exact answer.

The reset diagnostic retains normalized importance weights and their
derivatives before reset, and compares with the uniform cloud after
the complete production reset and one-step correction. It integrates
the next innovation independently by Gauss–Hermite quadrature.

| Quantity | Total LEDH minus T2 reference | Post-reset minus pre-reset prediction |
|---|---:|---:|
| Log likelihood | +0.014424 | +0.014979 |
| Score: atanh(rho) | +0.036151 | +0.030208 |
| Score: log(sigma) | +0.000826 | -0.006396 |
| Score: log(beta) | +0.012735 | +0.006954 |

These reset effects persist when conditional integration uses 64 rather
than 32 nodes: the log-density difference is unchanged in FP32, and
score-effect changes are at most `8.94e-8`. For this one data/design
pair, reset explains much of the log-likelihood and persistence-score
gap. It does not explain every component in the same direction.
Matching first and second moments does not preserve a nonlinear
predictive density or its parameter derivative. This is the substantive
approximation exposed by the test.

The accounting identity behind [t2-decomposition.json](t2-decomposition.json)
is, with Q_pre/Q_post denoting the conditional predictive log densities,

    ell_2 - ell_ref
      = (ell_1 + Q_pre - ell_ref)
        + (Q_post - Q_pre)
        + (ell_2 - ell_1 - Q_post).

It holds for the scores as well. For the value, the three terms are
`+0.0010033`, `+0.0149790`, and `-0.0015586`; for the persistence score
they are `+0.0074426`, `+0.0302078`, and `-0.0014992`. They sum to the
reported gaps. This is an attribution on the saved finite cloud, not
a replicated claim that removing reset would improve all targets.

## Particle variation and the long-horizon reference

Four fresh particle designs at each horizon use N=4032 on identical
observations. All endpoints are valid. MCSE below is across design
replications; it does not measure time-series sampling uncertainty,
HMC uncertainty, or particle approximation bias. Four replicates are
too few for strong tail or superiority claims.

| Horizon | Quantity | Design mean | Design SD | MCSE of mean |
|---|---|---:|---:|---:|
| T2 | Log likelihood | -1.254930 | .030376 | .015188 |
| T2 | atanh(rho) score | -.421184 | .019418 | .009709 |
| T2 | log(sigma) score | -.715227 | .026532 | .013266 |
| T2 | log(beta) score | -.569871 | .004835 | .002417 |
| T50 | Log likelihood | -39.681641 | .062745 | .031372 |
| T50 | atanh(rho) score | -2.351205 | .067281 | .033640 |
| T50 | log(sigma) score | -6.073619 | .073911 | .036956 |
| T50 | log(beta) score | -3.590910 | .062742 | .031371 |

There is no exact T50 oracle in this campaign. The existing independent
reference is a replicated bootstrap/Fisher path-score estimate. Its
score formulas pass the coordinate/dependency derivation in the source
audit, but finite-particle bias remains unknown. Repeated resampling
can reduce historical path diversity; the current reference does not
retain the genealogy diagnostics needed to measure that effect.

The persistence-score design mean is `-2.351205`, versus the reference
`-2.125355`: a difference of `-0.225850`. Combining the reported
replication standard errors in quadrature gives about `.041124`. The
gap is large relative to that estimated random variation, so simply
calling it Monte Carlo noise is unsupported. This is not a calibrated
significance claim with four LEDH designs, and neither MCSE accounts
for finite-particle bias. Reset bias and reference bias remain competing
explanations; the derivative checks do not decide between them.

The corresponding mean value difference is `+0.002295`; the other
score differences are `+0.008112` and `+0.005116`. A good likelihood
value at a point therefore does not settle the persistence derivative.

## Decision and inference status

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| No demonstrated derivative defect at inspected point | T1/T2 tight FP64 agreement; T50 FP32 ladder is roundoff sensitive | All retained T50 validity checks pass | T50 FP64 and other parameter points untested | Preserve derivative implementation; use appropriate precision/steps for future checks | Global score correctness |
| Reset is a measured T2 contributor | Weighted pre/post predictive difference survives GH refinement | Density/reset checks pass | One dataset/design and short horizon | Replicate this decomposition before selecting a reset repair | Reset is the sole T50 cause |
| T50 persistence gap unresolved | Difference is not explained by reported replication MCSE alone | Reference formulas pass source derivation | Finite-N LEDH and Fisher-reference bias | Refine independent reference and replicate reset attribution | Exact-model posterior is certified |
| Diagnostic runner reporting repaired | Full T50 ladder, raw perturbations and baseline traces retained | JSON and syntax checks | Earlier traces/logs were not retained | Preserve explicit metadata corrections | Historical logs were reconstructed exactly |
| Terminal-reset failure path identified | Source always resets before final validity decision | It did not fire here | Consequences outside inspected points | Design a separately tested likelihood-only terminal path | A terminal-reset bug caused these score gaps |

| Inference status | Finding |
|---|---|
| Hard veto screen | Successful diagnostic runs are finite and valid; earlier harness/runtime failures are excluded from scientific comparisons |
| Statistically supported ranking | None; no reset candidate or filter is promoted |
| Descriptive-only differences | Four-design means/SD/MCSE and one-design reset attribution |
| Default readiness | No default change; retain FP32/N=4032 and existing production controls |
| Next evidence needed | Replicated reset attribution, long-horizon reference refinement with genealogy/score uncertainty, and then a repair evaluated against the same target |

The original campaign's EKF, cubature and bootstrap comparators remain
available; this diagnostic does not rank a new method against them.
No heuristic-dominance or downstream HMC improvement claim is made.

## Execution and review

The focused structural test suite passed `19 tests` with two warnings.
Final checks passed Python compilation and CLI help, parsed all 25 saved
JSON files with non-finite JSON constants rejected, and verified that the
final diagnostic script checksum matches the summary.
The isolated checkout required a local Sylvester-op build before import.
The successful full FP32 diagnostic took 247.15 seconds; the CPU reset
check 10.95 seconds; FP64 CPU 504.97 seconds; retained T50 GPU ladder
121.41 seconds; and 64-node reset refinement 14.03 seconds. These are
whole-process wall times including compilation; they are not
steady-state kernel benchmarks or exact GPU utilization accounting.

Early failed launches did not retain complete logs or elapsed times.
Their manifests and one partial-result checkpoint remain; the next
manifest start gives an upper bound on the preceding completed attempt's
elapsed duration. The runner now refuses to overwrite an existing output
directory and records success/failure details for future launches.
The FP64 manifest's inherited FP32 label is corrected in a separate
metadata correction file rather than silently overwriting raw evidence.

Strongest alternative explanation: the T50 reference is biased rather
than LEDH, or both are. A materially different refined reference could
reverse that attribution. Weakest scientific evidence: one-data/design
reset attribution and four design replications. A single completed
HMC run with good convergence diagnostics cannot resolve either issue.

This completes the requested diagnostic plan. The next research phase
would test a reset repair and strengthen the T50 reference; it is not
an unfinished HMC sampling stage and no background run remains.
