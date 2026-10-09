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
included together with their gnuplot source/data. The final merged 668-page monograph
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
| fit-extended-20261006T114852 | Second seed: 2048 forward, 128 reverse updates | 2323.38 | 2358.89 |
| continue-reverse-20261006T123655 | Continued saved Adam state from reverse128 to reverse256 | 1052.64 | 1069.42 |

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
reverse phase reduced the measured teacher residual and base residual median while
leaving very large tails. The full-moment Gaussian control had median/p99
residuals 167.18/710.91; the NAF short endpoint is descriptively better in the
center, but no ranking uncertainty was estimated.

The stage-five decision was to execute the planned second seed at 2,048 forward
and up to 128 reverse updates. Large finite residuals do not invalidate the
harness or trigger a declared continuation veto. The first fit learned its
teacher and remained numerically valid, so stopping before that planned
calibration would be premature. The long 8,192/16,384-update ladder still
requires a measured decision after this bounded second seed. Its 4,200-second
reservation fits the initial 7,200-second exposure cap together with the
2,896.77 seconds already charged.

The second seed completed 2048 forward and 128 reverse updates in 2323.38
seconds (38.7 minutes), above the initial 20--25 minute estimate but within its
4200-second ceiling. The 1536-update forward block cost 690.62 seconds and the
last 96 reverse updates cost 627.34 seconds. Validation cross entropy changed
from 18.031 initially to 5.809 forward and 5.989 reverse. Paired 1000-point
residual median/p99/max changed from 3.087/342.34/1694.00 at the forward endpoint
to 1.995/74.28/671.22 after reverse refinement. Final sign mass was .457; 80.7%
of the base draws still had residual norm above one. Inverse-teacher weighted
mean residual was 3.918. All updates were finite and unclipped, and both
objective endpoints passed reload, inverse and represented-region checks.
There is no statistical cross-seed ranking: seed and forward/reverse budgets
both differ from the first pilot.

A read-only CPU inspection of the preserved tensors found all ten largest
residuals at points with estimated log(p/q)<-3. The largest residual was inside
||z||<=3, so it is not exclusively an extreme-base-tail problem. The estimate
uses the same bank's importance normalizer and is explanatory, not an exact
posterior-density certificate. This points to generated excess density that
finite empirical forward fitting does not directly penalize. The current
stage-six test continued the saved reverse128 Adam state to reverse256 at the
same .0001 rate and fresh RNG counter, with four 32-update checkpointed blocks.
Its 1800-second reservation fits the initial 7200-second pilot exposure limit.
A CPU reference check restored the saved parameters, Adam state and counter
exactly without target evaluation or training. No long forward ladder has
been launched.

The reverse continuation completed in 1052.64 GPU-process seconds (17.5 minutes).
All four 32-update blocks were finite, unclipped, batch-native and GPU/XLA.
The frozen map reloaded exactly; relative inverse error was 1.34e-10 and
log-determinant discrepancy was 6.63e-11. Its complete 1000-point probe gave
residual median/p99/maximum 1.681/57.44/279.99, compared with
1.995/74.28/671.22 before continuation on the same base bank. Validation cross
entropy was 5.7347; the negative sign fraction remained .457 against the
validation teacher's .524, passing the predeclared exploratory coverage rule.
Inverse-teacher weighted residual mean was 3.236, maximum 48.46. Importance
ESS was 294.09/1000, and 76.4% of base points still had residual norm above one.
These are descriptive measurements, not a statistically established ranking.

The initial short-test plan is complete. It used 6272.79 measured GPU-process
seconds, including failed infrastructure attempts, below its 7200-second cap.
The result passes the finite training, heldout-fit and represented-region
requirements for considering further target-specific calibration. It does not
supply a correctly whitened q20 map. No numerical, target, data or mathematical
continuation veto occurred: residual tails are a candidate limitation and a
repair signal. Completion of this bounded pilot is not rejection of NAF or
the forward/reverse research direction.

The next justified experiment is a separately priced reverse-rate/budget
calibration from the preserved forward2048 checkpoint, followed by a fresh
fitting seed and independent teacher check. The measured steady reverse cost
is about 6.53 seconds/update: a 512-update arm costs about 3343 seconds before
compilation and diagnostics. A two-arm exposure allowance of 8400 GPU seconds
would cover that measured cost plus setup margins. This is a planning estimate,
not a launched run or an optimization result. A much longer empirical forward
fit is not the next discriminating test of the observed excess-density tails.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Reject prior teacher | Weight screen fails in all eight banks | Severe weight concentration | Better full-support proposals | Use replay-checked SMC warm start | Failure of NAF |
| Admit saved SMC for a bounded warm start | Replay and replicated heuristic screens pass | No numerical/identity veto | Known-region support and finite particles | Short GPU fitting with independent validation | Exhaustive posterior coverage |
| Retry the local control repair | No training occurred in failed fit | Dtype error repaired | Unmeasured training geometry | Preserve completed short fit | q20 whitening or HMC readiness |
| Complete two short NAF fits and saved-state continuation | Finite training, heldout fit, coverage and reload checks pass | No numerical veto; large finite score tails remain | Teacher support and reverse-rate/budget calibration | Preserve checkpoints; price matched reverse calibration | Correctly whitened q20 posterior |
| Preserve synthetic results | Completed 12/12 random screens | Large finite tail residuals remain | Limited geometries | New-target short tests | General posterior transfer |

| Inference status | Finding |
|---|---|
| Hard veto screen | Prior weights fail; current target derivatives and replayed SMC teacher pass their bounded checks. |
| Statistically supported ranking | None. |
| Descriptive differences | On the second fit's common base bank, residual median/p99/max changed from 3.087/342.34/1694.00 forward to 1.681/57.44/279.99 after 256 reverse updates. |
| Default readiness | Existing q20 consumer default unchanged. |
| Next evidence | Matched reverse-rate/budget calibration, a fresh fit and independent teacher check before downstream validation. |

Post-run review: the strongest
alternative explanation for a favorable student is incomplete teacher support.
Paired base probes plus inverse-mapped validation points can expose local
geometry errors, but both can miss undiscovered regions. A new independent
teacher disagreeing materially would invalidate a stronger coverage claim.

## Terminal verification and accounting

The continuation's focused CPU-only checks passed (7 tests). The terminal audit
verified 229 result artifacts and 5176 saved source files across all attempts,
complete finite endpoint probes, GPU/XLA training records, parent-checkpoint
identity and identical local/shared charges. See `terminal-audit-r1.json`.
These checks establish record integrity and the checked mechanics, not q20
posterior correctness. No worker remains active.

Measured engineering checks/builds consumed 972.54 CPU-core seconds. An
additional 4000 CPU-core seconds was deducted as a contingency for earlier
unmetered checks and small audit/render operations. The latter is explicitly
not measured consumption or a proved upper bound. Its rationale and individual
measured charges are in `engineering-accounting-r1/result.json`. The remaining
bookkeeping allocation is 105674.49 GPU-process seconds (29.35 hours) and
92964.58 CPU-core seconds (25.82 hours), including that contingency deduction.

Final LaTeX build `monograph-integration-r1/build-r5.json` passed with no
unresolved citations/references or overfull boxes in the changed section.
Rendered q20 pages 587--588 were inspected; the earlier synthetic pages were
already reviewed. `manuscript-review.json` binds this review to the final PDF
and section hashes. Existing layout warnings elsewhere in the monograph are
not represented as repaired.
