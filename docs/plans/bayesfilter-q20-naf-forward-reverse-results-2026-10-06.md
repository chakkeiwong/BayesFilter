# q20 short-first NAF execution

## Manuscript and integration

The completed synthetic forward/reverse study is now in
`docs/chapters/ch26f_neutra_nonlinearity_results.tex`. The revision preserves
the earlier attribution mathematics and adds the native teacher design,
empirical/population objective distinction, responsibility-feature derivation,
retrospective fixed-case criterion revision, prospective random results,
finite-probe whitening limits and the q20 interpretation boundary.

Full LaTeX compilation succeeded with no unresolved references or citations.
The new results table and equations were inspected in rendered PDF pages.
An existing overfull line in the changed section was repaired by displaying
the mixture definition. Pre-existing layout warnings elsewhere in the large
monograph remain; this is a review of the changed material, not a certification
that every page is error-free. Build/review artifacts are under
`artifacts/q20-naf-forward-reverse-2026-10-06/monograph-r1`.

Integration inspection found the dirty study checkout had removed existing
FP32/FP64 transport, fixed-affine and explicit precision-conversion APIs.
Those unrelated regressions were repaired while retaining the NAF attribution
control, tighter internal inverse stopping margin, and common joint optimizer
update. Pre-repair files are preserved in `pre-precision-repair`. No historical
scientific artifact was rewritten. Seventy-five focused tests passed, covering
precision, the shared numerical authority, attribution, weighted forward/reverse
training and q20 pilot decisions. The five new pilot tests also passed separately.
CPU-only tests intentionally hid GPUs and do not establish GPU training quality.

The isolated integration build exposed two omitted figure dependencies, now
included together with their gnuplot source/data. The merged 667-page monograph
compiles without unresolved references or citations; the q20 table and chart
Jacobian derivation were inspected in the rendered output. Existing unrelated
layout warnings remain.

The broader study suite first passed 256 tests and found three integration gaps:
an identity control silently inherited the newly selected NAF (zeroing its
weight-normalized conditioner produced NaNs), a missing route-ledger entry, and
an omitted author-reference trajectory fixture. The identity control now
explicitly constructs the zero-shift/log-scale IAF identity. The two new study
routes were added to the existing shared-controller ledger without replacing
remote entries, and the saved upstream-reference fixture is included. All 91
checks covering the failed and remaining test groups then passed. The route
audit also exposed an omitted pre-existing acceptance-fixture classification;
its explicit mechanics/reference record was restored, and all six route-policy
checks passed. No sampler code or classification of its scientific evidence
was changed. Three
unrelated all-model registry checks failed earlier (two predator-prey signatures
and a frozen SIR hash); both signature mismatches and the exact SIR hash error
also reproduce on untouched origin/main. Those target files and unrelated HMC
edits are excluded from this commit, and their checks are not reported passing.

## q20 execution and local repairs

All commands use `/home/ubuntu/anaconda3/envs/tfgpu/bin/python
scripts/run_q20_neutra_pilot.py`, trusted GPU 0, TensorFlow 2.20, FP64, XLA,
verified memory growth and source snapshots. Exact arguments and resource
charges are in `artifacts/q20-naf-forward-reverse-2026-10-06/state.json`.

| Attempt | Result | GPU-process seconds | CPU-core seconds |
|---|---|---:|---:|
| price-20261006T103006 | Timed out before first target batch | 900.22 | 901.82 |
| price-20261006T104727 | Stopped after localizing backend mismatch | 640.48 | 642.40 |
| price-20261006T105808 | Target/derivative checks pass; prior teacher rejected | 414.38 | 417.22 |
| reuse-smc-20261006T111325 | All 800 saved target values replay; teacher screen passes | 141.39 | 143.76 |
| fit-20261006T111646 | FP32 construction in diagnostic control rejected before training | 35.60 | 37.75 |
| fit-20261006T111910 | Short NAF forward/reverse ladder completed | 764.70 | 795.73 |

The first two attempts inadvertently inherited the bridge's older
`compiled_custom_op` backend. Explicitly selecting the previously used q20
`tensorflow_eigh_strict_factor_cached` backend restored execution: the first
32-row call took 15.88 seconds and subsequent calls averaged 3.00 seconds.
All 4,096 prior points had finite values and scores. Two finite-difference
steps gave maximum errors, scaled by 1+|analytic score|, of 1.41e-7 and 3.53e-8.
This supports derivative consistency on the checked points, not global target
accuracy. Native debugger attachment was denied; no system policy was changed.
The backend contrast identifies a working route but does not prove the exact
native cause of the earlier stall.

Prior importance sampling is unsuitable as this pilot's teacher. Individual
512-particle populations had ESS 2.31--9.53 and maximum weight .173--.613.
None passed the declared weight screen. These particles were not used for
training. The existing SMC repair was therefore examined before paying to
regenerate populations.

Eight saved beta-one SMC populations from August 10 have the same four-parameter
T30 UKF target signature. The reuse step verified their terminal tensor hashes,
pre-resampling weight normalization and affine chart reconstruction, then
recomputed all 800 target values and scores on GPU. Saved log densities include
the chart Jacobian. After subtracting its derived half log determinant, maximum
scaled density error was 5.20e-16. The first four populations form the frozen
training partition and the last four form validation. Their ESS ranges from
87.83 to 100 of 100; each represents both observation-weight sign regions and
retains 50--65 distinct initial roots. The train/validation negative-mass means
differ by .1062, with between-population SE .04077, passing the exploratory
screen. This is approximate warm-start evidence conditional on two known
proposal-supported sign regions. No source mutation crossed their boundary;
exhaustive mode discovery and posterior correctness remain unproved.

The first fit failed while creating a diagonal Gaussian diagnostic control:
TensorFlow inferred FP32 from a Python covariance list. Explicit FP64
construction fixes the mismatch. No optimizer update occurred. Focused checks
passed, and fresh attempt `fit-20261006T111910` completed under the same
scientific and resource contract. It included prior/diagonal/full-covariance
controls, 0/128/512 forward updates, 32 reverse updates when the warm-start
coverage check passes, frozen-map reload checks, independent inverse-teacher
geometry and paired 1,000-point endpoint probes.

Integration review also preserved the remote log-domain DSF weight-underflow
repair and precision documentation. An identity-mixture FP32/XLA regression
with log weights near -119 passes; tiny positive mathematical weights must not
be rejected merely because their linear floating-point representation
underflows. Existing source-profile positional argument order is retained.
Twenty precision/pilot tests passed after API reconciliation, followed by seven
pilot tests and eight focused underflow/pilot checks after the local additions.
These intentionally CPU-only checks are engineering evidence.

The short fit is numerically viable but is not a whitening result. Held-out
teacher cross entropy fell from 18.020 at the initial map to 8.117 after 128
forward updates and 6.531 after 512. All updates were finite, batch-native,
XLA compiled and unclipped; every endpoint reloaded and retained the two
represented sign regions. The 32-update reverse phase remained finite and
retained coverage, but its cross entropy rose to 9.757 because the reverse
objective differs from teacher cross entropy. The base-draw score residual norm
(median, p99, maximum) was (32.77, 241.54, 335.93) initially, (32.04,
2041.89, 5490.36) after 512 forward updates, and (8.34, 479.51, 745.13) after
reverse updates. On inverse-mapped validation-teacher points, weighted residual
means were 16.85, 7.15 and 5.81, with maxima 92.89, 61.26 and 27.62. Thus the
reverse phase improved central teacher geometry and the residual median while
leaving very large tails. The full-moment Gaussian control had median/p99
residuals 167.18/710.91; the NAF short endpoint is descriptively better in the
center, but no ranking uncertainty was estimated.

The stage-five decision is to execute the planned second seed at 2,048 forward
and up to 128 reverse updates. Large finite residuals do not invalidate the
harness or trigger a declared continuation veto. The first fit learned its
teacher and remained numerically valid, so stopping before that planned
calibration would be premature. The long 8,192/16,384-update ladder still
requires a measured decision after this bounded second seed. Its 4,200-second
reservation fits the initial 7,200-second exposure cap together with the
2,896.77 seconds already charged.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Reject prior teacher | Weight screen fails in all eight banks | Severe weight concentration | Better full-support proposals | Use replay-checked SMC warm start | Failure of NAF |
| Admit saved SMC for a bounded warm start | Replay and replicated heuristic screens pass | No numerical/identity veto | Known-region support and finite particles | Short GPU fitting with independent validation | Exhaustive posterior coverage |
| Retry the local control repair | No training occurred in failed fit | Dtype error repaired | Unmeasured training geometry | Preserve completed short fit | q20 whitening or HMC readiness |
| Admit the short NAF pair | Finite training, coverage and reload checks pass | Tail score residuals remain very large | Teacher support and optimization tails | Execute second seed and larger bounded rungs | Correctly whitened q20 posterior |
| Preserve synthetic results | Completed 12/12 random screens | Large finite tail residuals remain | Limited geometries | New-target short tests | General posterior transfer |

| Inference status | Finding |
|---|---|
| Hard veto screen | Prior weights fail; current target derivatives and replayed SMC teacher pass their bounded checks. |
| Statistically supported ranking | None. |
| Descriptive differences | Prior/SMC weights, region masses and measured runtime; the first q20 pair completed and bounded calibration continues. |
| Default readiness | Existing q20 consumer default unchanged. |
| Next evidence | Conditional tail diagnostics, a fresh teacher partition or calibrated reverse retry, then a new decision on long training. |

Post-run review: the strongest
alternative explanation for a favorable student is incomplete teacher support.
Paired base probes plus inverse-mapped validation points can expose local
geometry errors, but both can miss undiscovered regions. A new independent
teacher disagreeing materially would invalidate a stronger coverage claim.
