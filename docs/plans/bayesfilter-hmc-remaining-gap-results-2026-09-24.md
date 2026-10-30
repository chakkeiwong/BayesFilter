# Remaining-gap execution results and incomplete confirmation

The engineering repair and bounded development work are complete. C2's first
confirmation attempt stopped after three per-fit timeouts: five null and four
quarter-SD fits completed, with no half-SD fit completed. The planned denominator
remains 128 null fits and 64 each with quarter- and half-posterior-SD target
shifts. Confirmation is incomplete; no campaign process is still running.
The governing documents remain the
[master program](bayesfilter-hmc-repair-master-program-2026-09-16.md), its
[A–F amendment](bayesfilter-hmc-remaining-gap-program-2026-09-24.md), and the
[execution contract](bayesfilter-hmc-remaining-gap-execution-2026-09-24.md).

## Repairs and verification

The independent quantile calculation found a real tied-cutoff rounding bug.
Interpolation could move a percentile slightly below two equal endpoints,
changing the indicator and its ESS. The runtime now uses the existing
tie-preserving percentile calculation and binds the corrected method identity
into precision policies and checkpoints. Two compatibility imports now refer
to their actual helper owners rather than the large legacy tuning facade.

The new `reference_mean` validation engine generates fresh normal-conjugate
data, runs complete public preparation/tuning/posterior assessment, and computes
an independent lugsail mean MCSE from archived model-coordinate draws. Exact
conditional truth stays in the assessor. It supports the existing process
isolation supervisor, cumulative retry budgets, source/design binding, immutable
completed assessments, and full planned denominators. An abnormal process exit
cannot promote a saved numerical alarm. A capped but completed fit remains
unavailable assessment evidence. The CLI reports the declared rate screen;
an occasional nominal null alarm does not by itself fail that screen.

Validation receipts under `artifacts/hmc-remaining-gaps-2026-09-24/` record:

| Check | Result and scope |
| --- | --- |
| Independent ArviZ reference module | 36 passed, one optional external-fixture skip. Pinned isolated reference environment. |
| Quantile/public-route repair checks | 92 passed; two concurrent Q20 consumer cases deliberately excluded from the committed-base snapshot. |
| Complete inference-validation suite | 359 passed; the remaining renderer case passed after restoring its omitted committed script. All 360 distinct cases passed across those receipts. |
| Reference-mean/isolation/design/execution checks | 60 passed, including real independent-data fits, ordinary public sampling, failed exits, source rejection and restart. |
| Final response/isolation checks | Three passed; final assessment/missingness checks: 19 passed, one public-fit case deselected because its integration check had already passed. |
| Official book | `main.tex` built successfully with owned chapter overlays; new page 433 visually inspected, in addition to earlier pages 432/474/475. |

Failed packaging/test attempts remain recorded and charged. The committed-base
snapshots exclude concurrent Q20, NeuTra core, governance and bibliography work;
these tests do not validate that separate revision. The main book remains the
guide, with the existing Markdown API reference aligned to it.

## Posterior development: all eight fits completed

[B's complete result](artifacts/hmc-remaining-gaps-2026-09-24/b-r1/development-result.json)
preserves every candidate and interval. All selected members passed their
declared posterior checks; every verified sibling remained retained.

| Model / arm | Verified members in the two fits | Selected L | Warmup / retained per chain |
| --- | --- | --- | --- |
| Gaussian baseline | 24, 24 | 3, 3 | 2000/1000; 2000/2000 |
| Gaussian candidate | 20, 25 | 3, 25 | 30000/4000; 30000/4000 |
| Beta-binomial baseline | 22, 17 | 3, 3 | 2000/1000; 2000/2000 |
| Beta-binomial candidate | 18, 16 | 3, 3 | 2000/5000; 2000/5000 |

One Gaussian candidate mean interval missed its exact reference. With two fits
per arm this is descriptive evidence, not a calibrated coverage rate or a
ranking. A foreign job acquired GPU 1; its overlap confounds the second Gaussian
candidate timing. The following beta launch failed before sampling, was charged,
and resumed unchanged on same-class GPU 2. No foreign process was interrupted.

The saved-array analyses retained all 256 Gaussian/beta slots. The quantile bug
does not change their historical median MCSEs. Direct order-statistic intervals
change only one stopped Gaussian and one fixed beta coverage event. Lugsail and
autocorrelation give identical stopped-mean coverage counts on those arrays.
These comparisons do not justify replacing either estimator or interval rule.
Five of seven archived Gaussian warmup caps pass the same screen using the full
10000-draw window, while two still fail; this nominates a window/count experiment
without proving equilibration.

## Detector activation, cost and confirmation

[C's development result](artifacts/hmc-remaining-gaps-2026-09-24/c-r1/development-result.json)
contains three independent fits plus one paired no-op execution. Baseline and
no-op have identical numerical records and draw checksums. Their standardized
mean error was -0.425, below the 2.576 alarm cutoff. The actual quarter- and
half-SD target shifts produced alarms, with standardized errors 11.758 and
19.013. All qualified after 2000 warmup and 1000 retained draws per chain.
These activations establish neither achieved null size nor detection power.

The three full-fit prices, including process startup/exit, were 216.591,
242.501 and 210.633 GPU seconds. They imply a descriptive 56,724-second price
for the entire predeclared C2 inventory. The frozen confirmation has a
65000-second numerical budget and 65100-second outer ceiling. This reserve
fits the existing allowance and includes a margin above the forecast; it is
not a runtime tail guarantee. No denominator or statistical threshold changed.

[The stopped suite](artifacts/hmc-remaining-gaps-2026-09-24/c-r1/confirmation-run-r1/suite/run_index.json)
used frozen source `c-r1/source-r7` and prospective seeds 2026092431/32/33.
Each fit runs in a fresh GPU/XLA process with verified memory growth. Final
rate bounds must use all planned slots: missing null outcomes are possible
alarms, and missing defect outcomes are nondetections. The original exact
95% screens remain null upper <=.10 and defect lower >=.80. No observed
confidence bound can stop or enlarge a cell.

The outer receipt ended at 2026-09-23 22:00:12 UTC with exit code 1 after
5077.745 GPU seconds. Each cell stopped at its first process timeout under
the existing infrastructure-stop rule: null fit 5, quarter-SD fit 4 and
half-SD fit 0 reached the cumulative 890-second per-fit ceiling. All nine
normally completed fits qualified; the five null fits produced no alarm and
the four quarter-SD fits produced alarms. These incomplete outcomes do not
establish null size or power. The full-denominator summaries correctly report
`calibration_incomplete`. A subsequent read-only inspection of the frozen
source and saved numerical chunks localized all three timeouts to active
candidate-set tuning, before posterior sampling. Numerical evidence continued
to be written shortly before termination. The checkpoints already contained
verified candidates; none of these exits was a posterior R-hat rejection.

| Timed-out fit (zero-based) | Completed tuning observations | Verified candidates already retained | Sum of completed tuning-chunk wall seconds |
| --- | --- | --- | --- |
| Null 5 | 106 | 2 | 778.055 |
| Quarter-SD 4 | 47 | 5 | 808.884 |
| Half-SD 0 | 44 | 4 | 795.212 |

These chunk times account for about 87--91% of each 890.367-second process
attempt. They exclude preparation, analysis and checkpoint overhead, and any
chunk interrupted by termination. They are wall times, not measurements of
exclusive GPU compute. For a matched chunk shape (four scalar chains,
136 transitions per chain, L=13), calls after the first cached-runner call had
median times of 0.251 seconds in completed null fit 0, 15.596 seconds in
quarter-SD fit 4 and 16.286 seconds in half-SD fit 0. Quarter-SD fit 4 itself
contained both a 0.248-second and a 16.249-second call at that same shape.
Null fit 5 likewise alternated between fast and slow calls. Changing L or
merely counting more proposals cannot by itself explain that discrepancy.
The sums and medians above come from each fit's
`tuning/numerical_chunks/*.json`; candidate counts come from
`tuning/tuning_checkpoint.json`, and elapsed totals from
`process-attempt-001-exit.json`.

Two execution-control problems are established by frozen `c-r1/source-r7`.
First, `procedures.py:327` uses `search_config=None` for native search, and its
preparation config does not receive the fit deadline. All three checkpoints
therefore record `max_wall_time_seconds: null`, although the outer supervisor
kills the process at 890 seconds. The public tuner already accepts a wall
budget through `public_timeout_budget_s`; this validation caller did not wire
it through. Second, `fit_process.py:215` breaks the entire cell on every
non-complete process receipt, conflating an exhausted local fit allowance with
an invalid harness. That is why three timeouts left only nine completed fits
out of the planned 256, despite substantial aggregate allowance remaining.

The underlying cause of the slow numerical calls remains unproved. The saved
records do not distinguish device contention, runtime scheduling or graph
execution overhead. Retracing warnings alone cannot settle that question;
the slow calls include reused runners with unchanged shapes. The three
development prices supported a point forecast, not a calibrated runtime-tail
bound. The earlier statement that the full inventory was affordable was too
strong: its estimated cost fit the allowance, but completion within it was
not established.

The next repair should propagate the remaining fit deadline through
preparation, tuning and posterior execution, preserve a partial checkpoint
before the hard process limit, and distinguish local budget exhaustion from
source/artifact/harness invalidity. After checking that a timeout is local,
remaining declared fits should continue within their original cumulative
allowances and full denominators. A bounded diagnostic comparing representative
fast and slow saved workloads, with runtime/device telemetry, is needed before
repricing or increasing any limit. Preserve every verified candidate and every
failed attempt. A new launch must not reset an exhausted fit's allowance,
select a faster subset, or treat an unfinished fit as a completed assessment.
No numerical rerun or code repair was performed for this read-only diagnosis.

C1 is still unfunded alongside C2. Subtracting fixed-comparator time from the
uncontended B observations gives price scenarios of 65,585 Gaussian and 71,604
beta-binomial GPU seconds for their respective 256-fit cells. These are
forecasts, not measured no-comparator prices. Together with C2 they imply
about 53.86 GPU hours before further learned training or exact consumer work.
Funding insufficiency does not reject the posterior-policy candidate.

## Learned geometry: priced, with a capacity issue identified

[D's result](artifacts/hmc-remaining-gaps-2026-09-24/d-r1/gpu-pricing-r1/data/result.json)
completed all 16 banana/mixture arms in 60.995 outer GPU seconds. The scalar
objective/directional-gradient discrepancies were below 2.3e-10 relative to
`max(1, gradient_norm)`. Frozen forward/logdet equality, inverse checks and
single-trace update graphs passed. There was no gradient clipping. Training
loss was still changing at 512 updates, especially for the mixture. These
short runs price this exact legacy trainer; they do not validate adequate
training or the concurrent transport-core changes.

The mixture also needs a capacity repair before a serious map-quality study.
In the frozen source, `neutra_training.py:1695` constructs one legacy dense-IAF
stage. The output mask at line 2799 allows only source degrees strictly below
an output degree; output coordinate one therefore receives no variable input.
Its transform is `theta_1 = exp(s_1) z_1 + m_1`, with constant learned
`s_1,m_1`. Standard-normal `z_1` thus gives a Gaussian first marginal, whereas
the target's first marginal is the separated two-normal mixture. Increasing
hidden width or iterations cannot make this single-stage family an exact
whitening map for that marginal. This is a derivation from the inspected masks
and transformation, not an empirical verdict against learned whitening or
evidence that partial whitening cannot help HMC.

Before spending on serious mixture training, use a supported composition with
coordinate permutations or another reviewed family capable of changing that
marginal. First test the capacity property and frozen inverse/Jacobian/score
identity; then price target-specific capacity/optimizer and longer-update
comparisons. Preserve identity, fitted diagonal/dense affine, and exact-reference
controls. The banana still needs an exact-unbending comparator and adequate
training checks. Independent model-coordinate posterior/mode/tail evidence
and fresh tuning remain required. Neither short training loss nor greater
width chooses a winning map. These are the next D requirements; no numerical
training default is promoted here.

## Decision and remaining work

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep narrow runtime repairs | Independent arithmetic and public regressions pass | No remaining scoped engineering failure | Untested concurrent edits and external fixture | Preserve corrected policy identity | Broad calibration |
| Keep posterior candidates optional | Eight development fits deliver | One interval miss; no health veto | Finite-fit coverage and selection cost | Fund full C1 later | Superiority or a default change |
| Preserve incomplete C2; timeout repair passes | Nine of 256 historical fits completed; new supervision/resume tests pass | Original failures retained; fresh GPU canary also reaches its finite allocation | Complete-fit prices under shared workload and source-stratified continuation | Price before an explicitly accounted continuation design | Null size, power or detector rejection |
| Revise learned-mixture protocol | Training arithmetic/composition pass | Single-stage marginal cannot equal target marginal | Adequate representable training and exploration | Capacity check, controls, deeper training, downstream assessment | Learned-map quality |
| Leave exact consumers open | Input inventory rechecked | Named bootstrap files and joint reference absent | Exact target/data availability | Use local MacroFinance reply and await matching inputs | Substitute-model validation |

| Inference status | Current evidence |
| --- | --- |
| Hard veto screen | Scoped regression checks pass; C2 process timeouts prevent confirmation, and exact consumers remain unavailable. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Eight posterior development fits, detector activations, runtimes and 16 learning curves. |
| Default-readiness | No new tuning, posterior, estimator or training default promoted. |
| Next evidence needed | Completed C2 bounds; full C1; representable and adequately trained maps with model-coordinate validation; exact consumer bundles. |

The strongest alternative explanation for favorable development outcomes is
ordinary seed variability and easy local geometry. Independent confirmation
can overturn the proposed detector's adequacy. Training loss can improve while
global mode exploration remains wrong. The weakest evidence is the small
development sample and the unknown adequate learned-training protocol.

The [resource ledger](artifacts/hmc-remaining-gaps-2026-09-24/reconciliation-live.json)
charges completed failures as well as successes, including the outer C2 receipt
once. Its unspent C2 allocation is reserved for diagnosis/continuation; no run
is active. Numerical, engineering and scientific dispositions remain
separate. C2's final report must be inspected before its scientific requirement
can change; the other open requirements do not inherit a C2 pass.

## Timeout amendment closeout

The [timeout repair result](bayesfilter-hmc-timeout-aware-supervision-result-2026-09-24.md)
records completed supervision, deadline propagation, cumulative accounting and
partial-resume repairs, 91 isolated-source checks and 41 terminal checks with
overlap. The official tuning chapter and aligned API reference are updated.
The bounded GPU canary observed real foreign-device activity and numerical
progress, granted its declared 90 seconds, then terminated the incomplete fit
at the hard boundary. Checkpoints and final unavailable-fit assessment survive;
trusted inspection found no remaining worker. Its enclosing 541.547 GPU seconds
are charged once, leaving 64,667.835 GPU seconds, including the unchanged
60,022.255 C2 reservation. It establishes execution behavior, not the cause of
historical slowdown or statistical confirmation. Complete-fit pricing and an
explicit continuation design remain the next research work.
