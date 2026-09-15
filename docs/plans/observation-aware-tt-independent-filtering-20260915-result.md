# A06 result: SGQF preservation on independent filtering sequences

2026-09-15. Master phases 9 and 10 are complete; see the
[phase-10 closeout](observation-aware-tt-phase10-closeout-20260915.md).

The SGQF-preserving TT candidate does **not** pass the predeclared filtering
criterion. In d1 it has observed losses to the transition proposal on ordinary
observations and to the analytical SGQF joint proposal on large observations.
Neither loss is statistically established, but either triggers the declared
heuristic promotion veto. In d4 the SGQF guide fails on one of 12 sequences,
so the full comparison is incomplete. The other 11 sequences give encouraging
descriptive results for the candidate; conditioning on successful guide
construction cannot establish its reliability across the intended population.

All 24 independent references pass their declared resolution/precision screens.
This is candidate rejection under the frozen procedure, not rejection of TT
filtering or of Gaussian conditional proposals as research directions.

## Question, frozen contract and provenance

The [reviewed A06 protocol](observation-aware-tt-master-amendment-06-independent-filtering-20260915.md)
asks whether the empirically SGQF-preserving fit improves actual filtering on
new observation sequences. Its plan review was AGREE before implementation:
`artifacts/observation-tt-defense-consumer-20260915-01/review-a06-01.txt`.
The skeptical audit checked the actual joint conditional consumer, independent
references, complete heuristic ladder, validation/audit separation and the
distinction between promotion failure and a continuation veto. The inherited
rank/degree, row mixture and optimization schedule remain hypotheses rather
than established optima. The full run froze these controls before generating
the observation sequences; source hashes still match after the run.

There are 12 independent T=20 stochastic-volatility sequences in each of d1
and d4, four particle replications per method and sequence, and N=512 particles.
The eight methods are the transition proposal, stationary-prior proposal,
SGQF current marginal, analytical SGQF joint conditional, two scalar-axis TT
methods, the original pair-block TT, and the SGQF-preserving pair-block TT.
The four simple proposals are constructed adversaries: respectively retain
model dynamics, ignore the previous particle, use the observation-conditioned
current Gaussian alone, and preserve both Gaussian conditioning and adjacent
dependence. The four TT methods complete the plain/enhanced comparison ladder.

The new candidate fits each time separately, with the previous selected
retained marginal in the next target. It uses degree/rank 3, four sweeps,
128 proximal steps, L1 weights {0, 1e-5, 1e-3}, and 1024/4096/8192
training/validation/audit rows from the frozen .2 reference + .8 joint-guide
mixture. The two fitted starts are generic and projected SGQF; validation
chooses between those fits and the analytical SGQF joint itself, with ties
favoring SGQF. The defensive fraction is the separately calibrated 1e-5.
Selections are saved before their audit rows are generated. Exact proposal
density corrections remain in the particle filter. The full-coefficient
initializer is a diagnostic d<=4 implementation, not a scalable initializer.

For each coordinate-time cell, square the particle-filter mean error against
the independent reference and divide by that coordinate's stationary variance.
Average over the four particle replications and cells within a sequence/regime,
then weight contributing sequences equally. This normalized MSE is the primary
criterion. Regimes are |y_j|/beta <=.5, between .5 and 2, and >=2, plus all
observations. At least eight sequences and 24 cells are required per contrast.

The independent observation sequence is the paired bootstrap unit. The frozen
9999 resamples, seed 920000, preserve methods and regimes jointly. Simultaneous
95% intervals use the maximum absolute centered bootstrap contrast divided by
its observed sequence standard error; zero-SE cases would get point intervals.
Only complete, adequately covered comparisons enter the maximum. The run has
16 eligible d1 contrasts and no eligible d4 contrast; none has zero SE.
This is a finite-sample bootstrap approximation, not guaranteed coverage.
Advancement requires every eligible upper limit <=0, half-width <=.01, complete
coverage of the intended comparison, valid references and no heuristic/validity
veto. A positive observed candidate-minus-heuristic mean is a promotion veto
even if its interval includes zero. ESS, log-evidence error, runtime, fitting
loss and audit results are explanatory and cannot replace this criterion.

## Execution and validity

Command actually executed:

```text
bash docs/plans/artifacts/observation-tt-independent-filtering-20260915-01/run_campaign.sh attempt-01
```

The wrapper invokes `run_observation_tt_independent_filtering.py` with
`--wall-budget-seconds 2400` and a fresh output root. One full attempt completed;
there was no full-run retry or holdout retuning. The eight-method mechanics
smoke took 19.711 seconds. The full run recorded 1611.880 seconds of monotonic
wall time, totaling 1631.592 numerical seconds against the 2700-second limit.
UTC manifest timestamps are 04:15:07.230916 to 04:44:55.929834; the monotonic
elapsed field is used for the run's time-limit accounting. Active planning,
review, implementation and reporting also consume the owner's five-hour ledger.

Environment: `/home/chakwong/anaconda3/envs/tftwogpu`, TensorFlow/TFP,
float64 numerical kernels, GPU/XLA, NVIDIA RTX 5080 UUID
`GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a`, escalated GPU access.
Memory growth was verified before GPU initialization (`a06_growth_v1`);
allocator peak was 308225536 bytes. TF32 was enabled as a device setting,
although these numerical tensors are float64. Scalar grid reference,
one-time Gaussian/projection setup and reporting are recorded exceptions.
No NumPy numerical runtime or pfor was added.

Git commit: `e7f2a88ecff49ced481b9615c0b1237b8cabe732` with the recorded dirty
tree and explicit source hashes. Data seeds are `916000+100*d+i`; fit seeds
`917000+10000*d+100*i`; particle seeds `918000+10000*d+100*i+10*r`;
reference base seeds `919000+10000*d+100*i+10*r`, with a 1000000 offset
per reference enlargement. Saved data include observations, states, A, P0,
beta and sigma. The model scope uses the old fixture's parameters, not its
observations. The manifest, controls, all sequence results and exact inputs are
in `docs/benchmarks/artifacts/observation_tt_independent_filtering_20260915/attempt-01/`.

The synthetic inference test initially exposed an empty-cell handling error.
The localized prelaunch repair excluded ineligible regimes from the maximum
and left their intervals absent, as required by the protocol. Its focused
regression then passed (`inference-tests-01.log`, `inference-tests-02.log`).
The complete eight-method mechanics smoke also passed before holdout execution.

On the completed particle runs, all 14880 saved step records contain finite
stored numerical values, including log corrections and covariance entries.
There are zero explicit false finite flags and zero bracket failures among
3368 records carrying bracket diagnostics; maximum recorded CDF residual is
1.4322e-14. A post-run reporting bug initially counted an absent `finite` field
as false. It was fixed without changing run data: 9452 records lack that optional
flag, and the reporter now checks actual stored numbers, including the serializer's
nonfinite string representation. Absence of a flag is not failure or proof of
validity. These checks concern completed runs and do not erase the guide failure.

All d1 801/1201-node references pass: maximum stationary-SD-normalized mean
difference 7.11e-16 and log-evidence difference 7.11e-15, both below 1e-6.
All d4 references pass their two-scale checks. Nine use four N=65536 runs;
sequences 4, 7 and 8 require the predeclared four N=131072 runs. Maximum selected
mean MCSE is .01735 stationary SD (limit .02), and maximum log-evidence MCSE
is .05060 (limit .10). These establish the stated resolution/MC screens, not
an exact high-dimensional filtering distribution or guaranteed absence of bias.

## Conditional filtering results

All entries are normalized mean-square errors. The d1 rows use all 12 sequences; every d4 row uses the same 11 sequences excluding sequence 8. The d4 entries describe successful-guide sequences only.

| d | Proposal | All | Near zero | Ordinary | Large |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Transition | 0.0021600 | 0.0030579 | 0.0011964 | 0.0020561 |
| 1 | Stationary prior | 0.0051131 | 0.0056793 | 0.0018754 | 0.0098854 |
| 1 | SGQF marginal | 0.0037369 | 0.0049775 | 0.0027870 | 0.0030887 |
| 1 | SGQF joint conditional | 0.0018482 | 0.0024519 | 0.0014518 | 0.0011200 |
| 1 | TT predictive | 0.0015124 | 0.0019170 | 0.0012909 | 0.0007813 |
| 1 | TT guided | 0.0016963 | 0.0018042 | 0.0013399 | 0.0017197 |
| 1 | Original pair TT | 0.0016075 | 0.0017946 | 0.0012691 | 0.0015217 |
| 1 | SGQF-preserving TT | 0.0016702 | 0.0019506 | 0.0014199 | 0.0013987 |
| 4 | Transition | 0.0066377 | 0.0093457 | 0.0041904 | 0.0065428 |
| 4 | Stationary prior | 0.0212687 | 0.0279851 | 0.0143865 | 0.0221852 |
| 4 | SGQF marginal | 0.0060415 | 0.0086580 | 0.0041741 | 0.0039220 |
| 4 | SGQF joint conditional | 0.0029703 | 0.0034087 | 0.0022337 | 0.0033048 |
| 4 | TT predictive | 0.0072739 | 0.0105720 | 0.0052669 | 0.0038474 |
| 4 | TT guided | 0.0066699 | 0.0090306 | 0.0053657 | 0.0034635 |
| 4 | Original pair TT | 0.0026151 | 0.0032617 | 0.0021797 | 0.0020670 |
| 4 | SGQF-preserving TT | 0.0023614 | 0.0029321 | 0.0019547 | 0.0018882 |

The d1 bootstrap critical value is 2.725897, with all 9999 resamples usable.
All 12 sequences contribute to each d1 regime, with 240/98/110/32 cells for
all/near-zero/ordinary/large observations. The two observed heuristic losses are:

| Comparator and regime | Candidate minus comparator MSE | Simultaneous 95% interval | Interpretation |
| --- | ---: | --- | --- |
| Transition, ordinary | +.00022344 | [-.00018662, +.00063349] | Observed loss; promotion veto, not statistically established inferiority |
| SGQF joint, large | +.00027871 | [-.00050373, +.00106115] | Observed loss; promotion veto, not statistically established inferiority |

Seven d1 intervals have negative upper limits: transition/near-zero;
stationary/all, near-zero and ordinary; SGQF marginal/all, near-zero and ordinary.
Those are the only supported directional comparisons under this bootstrap
analysis. No SGQF-joint contrast has a negative upper limit; its all-observation
contrast is -.00017804 [-.00058381, +.00022773]. The stationary/large contrast
also misses the precision requirement (half-width .010486 > .01). There is
no supported overall ranking across all methods/regimes. Comparisons among
the TT variants were not assigned inferential intervals and remain descriptive.

Every d4 contrast is ineligible because the candidate is missing on sequence 8.
No confidence interval or population ranking is issued for d4. The 11-successful-
sequence table is an explicit conditional description, not a repaired sample.
For completeness, using all 12 sequences gives all-observation MSE .0069957
for transition and .0218795 for stationary; those unmatched means must not be
compared with the candidate's 11-sequence mean. All such outcomes remain saved.

Whole-vector maximum-observation regimes, explanatory only, contain
98/110/32 near-zero/ordinary/large time points in d1 and 5/126/109 in d4.
They do not replace the predeclared coordinate-wise criterion.

Log-evidence diagnostics likewise remain descriptive. On the matched sequences,
candidate mean log-evidence error/max absolute sequence error is -.01384/.07673
in d1 and -.01068/.13624 in d4. For SGQF joint these are -.00490/.10457 and
-.01713/.30452. The complete per-sequence errors and four-replicate MCSEs are
in `result.json`; there is no inferential log-evidence ranking.

Median per-sequence fit/total times are .170/.647 seconds for d1 SGQF joint
and 5.594/6.593 for the candidate; d4 medians are .189/.665 and 43.520/45.624.
These are descriptive measurements in one ordered campaign with compiled-kernel
reuse; shared guide construction is recorded separately. They do not establish
a deployment cost or speed ranking under equal cold-start conditions.

## What failed in SGQF and what initialization accomplished

At d4 sequence 8, zero-based time 18, the guide update receives observation
[-3.1916, 2.7754, -.3507, .1547] with beta=.4. Each signed sparse-quadrature
level has positive estimated mass but a non-SPD covariance. Minimum eigenvalues
at levels 2/3/4/5 are approximately -6.68e-12, -.001137, -1.08550 and -.080229.
This is an invalid moment calculation under the frozen rule, not merely a
low TT fitting score. A valid Gaussian conditional formula cannot repair an
invalid input covariance. No clipping, ridge or replacement proposal was
silently applied. The shared guide failure invalidates the six guided methods
on that sequence; transition and stationary continue to completion.

The unchanged CPU-only diagnostic replay identifies time 18 and reproduces
the failure (`guide-failure-replay.json`, .230 seconds of numerical work).
CUDA was intentionally hidden; this replay is failure localization, not a
replacement GPU result or a candidate repair. Its frozen source/input hashes,
command and scope note are preserved. The large higher-level negative
eigenvalues weaken a pure roundoff explanation. The evidence is specific to
this signed quadrature family/level range and input, not all SGQF algorithms.

Among completed candidate fits, SGQF initialization remains useful. At d4
transitions the selector chooses the SGQF-started TT 191 times, generic TT 10
times and exact SGQF eight times; all 11 initial proposals are exact SGQF.
In d1 the corresponding transition counts are 59/145/24, with 12 exact initial
proposals. These are algorithm decisions on dependent sequential targets, not
independent trials proving one initialization superior.

Validation selection is no worse than SGQF on the selection rows by construction.
Independent audit rows reveal losses in 9/228 d1 and 1/209 d4 transition fits;
maximum H2 increases are .002248 and .002169. Thus the selection rule is an
empirical safeguard, not a population lower bound. The d4 audit comparison is
also conditional on the guide succeeding. Minimum target-weight ESS on audit
rows is 5859/8192 in d1 and 985/8192 in d4; d4 validation reaches 269/4096.
Coverage is better descriptively than the exposed A05 late-tail case but still
uneven. The recursive fitting target uses the previously retained approximation,
so a lower fitting loss alone cannot certify the true filtering law.

## Decision, uncertainty and next action

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Do not promote the frozen candidate | Fails d1 all-comparator requirement; d4 incomplete | d1 observed conditional heuristic losses; d4 shared-guide failure | Twelve sequences; finite-sample intervals; reference MC error; dependence within each fitted path | Finish phase 10 and record a reviewed amendment before any new repair experiment | No rejection of the TT research direction; no universal SGQF failure |
| Retain analytical SGQF as an explicit comparator and selection option | No supported d1 all-regime improvement over joint SGQF | Same shared guide fails in d4 | Non-Gaussian filtering approximation and quadrature robustness remain distinct issues | Any later quadrature repair uses new calibration and confirmation sequences | Gaussian-joint algebra alone does not guarantee valid quadrature moments |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Guide covariance failure on one d4 sequence; d1 observed heuristic losses block promotion. No stored nonfinite/CDF failure in completed particle runs. |
| Statistically supported ranking | Seven specified d1 metric/regime contrasts favor the candidate under the approximate simultaneous bootstrap; no overall ranking and no supported improvement over joint SGQF. |
| Descriptive-only differences | All TT-variant comparisons, successful-sequence d4 means, selection/audit counts, log-evidence differences, ESS and runtime. |
| Default readiness | Not established; this remains an optional diagnostic implementation with a d<=4 initializer. |
| Next evidence needed | A reviewed robustness/selection repair protocol, fresh scope-specific calibration, then untouched independent sequences with adequate uncertainty and the same simple-proposal adversaries. |

The engineering ledger contains passed mechanics/consumer/inference checks and
unchanged source hashes. The numerical ledger contains passing references and
completed-run diagnostics alongside a real guide failure. The scientific
ledger rejects promotion while retaining the narrow d1 directional evidence.
No evidence is moved between these ledgers to excuse a failed primary criterion.

Post-run red-team: the strongest alternative explanation for the favorable d4
table is selection of the 11 easier, successful-guide sequences. Small sequence
counts and four-run reference MCSE estimates are the weakest inferential support.
A robust predeclared guide and a fresh complete confirmation set could overturn
the present reliability verdict. Better fitting loss alone could not. The A06
observations are now exposed and cannot be reused as untouched confirmation data.
No pair total-gradient, retraining derivative, HMC, larger-dimensional,
source-faithful TT-cross, scalable-initializer or production claim is made.

The master program's terminal documentation and manuscript update are complete.
A future repair should separately examine signed-quadrature robustness and
selection/generalization against the exact SGQF joint; it requires a recorded
amendment with a new evidence contract and review. This result does not silently
activate that experiment or expand the remaining five-hour budget.

## Evidence index and terminal review

Operational directory: `artifacts/observation-tt-independent-filtering-20260915-01/`.
It contains the launch log, focused-test logs, prelaunch audit, post-run
`assemble_report.py`, `report.json`, `tables.md`, unchanged failure replay and
terminal review wrapper. The reporter constructs the same successful sequence
set for every method in its d4 descriptive table and retains all available
outcomes separately. Its source/result checksum is in `report.json`.

Run evidence: `docs/benchmarks/artifacts/observation_tt_independent_filtering_20260915/attempt-01/`
contains `run_manifest.json`, `frozen-controls.json`, `result.json` and the 24
sequence directories. Each stores data, references, method results, fit grids,
retained cores/charts and pre-audit selections when available.

Terminal interpretation review: **AGREE**, saved in
`artifacts/observation-tt-independent-filtering-20260915-01/review-terminal-01.txt`.
This bounded review checked the result interpretation, not an independent
rerun of its numerical evidence. No material revisions were requested.
The [phase-10 closeout](observation-aware-tt-phase10-closeout-20260915.md) records
the 54-page manuscript, rendered and preservation checks, final owner-budget
accounting, and the unverified separate A06 active-work suballocation. The latter
does not change the numerical result and must not be reported as a passed budget
gate.
