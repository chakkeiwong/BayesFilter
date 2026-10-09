# Acceptance repair execution and test evidence

The current engineering question is whether fixed-horizon independent trials
can survive the existing controller, health checks, recovery and retained
export without duplicating information or changing the acceptance target.
The governing plan is
[the October 2 acceptance repair](bayesfilter-hmc-acceptance-decision-repair-plan-2026-10-02.md);
the [robustness matrix](bayesfilter-hmc-acceptance-robustness-matrix-2026-10-02.md)
defines model roles and the deliberate defects the tests should catch.

## Expanded execution: current checked result

The root-cause repair and bounded validation are executed. The combined suite
passed **428 tests, zero failures and zero skips** in 657.820 seconds. Subsequent
focused checks added successful QR, residual-map and mixture regressions;
serial/threaded v7 trial collection; and a positive K0 posterior-reference
check. Four actual QR/nonlinear fresh-process recovery tests and four trusted
CPU/GPU/XLA parity tests also passed. The three GPU public-tuner checks passed
with actual chunk placement, verified memory growth, checkpoint recomputation
and member export/reload. These counts are overlapping evidence slices, not
numbers to add indiscriminately. The terminal audit records unique test IDs.

The procedure still uses the two public tuners and common candidate controller.
It retains fixed W/T, independent whole-start-bank repetitions, all verified
members, exact-pair repairs and fresh verification. The fixes also ensure that
lost native calls are charged without duplicating statistical information,
individual persisted chunks cannot lack charges, and unknown SSM fixture
parameters cannot silently describe a different model. The latter was a real
fixture defect: a purported diffuse/zero-covariance case previously could run
unchanged defaults. Unsupported fixture inputs now fail explicitly.

The maintained fast regression list now includes the new protocol/statistics,
accounting, adversarial and calibration controls. The larger numerical suite is
`docs/validation/hmc-acceptance-integration-tests.txt`. The executable coverage
inventory maps mechanisms to actual tests, not just model names. The official
chapter in `docs/main.tex`, agent reference and generated registry all describe
the same experimental policy. The full book builds; changed physical pages
420--421 were rendered and inspected. Its 78 unresolved citations elsewhere
are pre-existing book debt, not a warning-free-build claim.

### Calibration and reference checks

The first slice had 11,264 searches. The expanded slice has 16,384 searches
(32 method/cells with 512 each), and the actual M=100 slice has 1,024 searches
(four method/cells with 256 each). Raw auditors checked original denominators,
streams, contiguous cumulative rungs, observed/generated work and all retained
members. Across the applicable error cells no false membership, direction,
preparation or temporal-alert event was observed. Simultaneous per-cell error
upper bounds are .017617430 in the expanded slice and .027053429 in M=100;
these are separate multiplicity families and cannot be pooled into a stronger
bound. Both methods verified all 100 interior members in every applicable
M=100 search and verified none with misleading pooled means. Each interior
search actually visited 100 measurement and 100 verification stages.

At the admission stopping time, the betting temporal diagnostic detected all
512/512 opposing, single-start and nonlinear temporal controls. Hoeffding
reported 2/512, 191/512 and 1/512 respectively. These diagnostics are not
admission requirements, and absence of an alert is not burn-in evidence.
The earlier Hoeffding heterogeneous-start delivery failure (0/512) remains
preserved. No convenient failed comparison is removed from the report.

### Public numerical model results

All positive rows below use the unchanged [.55,.85] qualification band and
fresh verification. Runtime is descriptive and includes differing resource
contention; CPU/GPU timing is not a fair speed comparison. Trials include
measurement and verification; each complete trial has four starts and W+T=68.

| Fixture | CPU confirmation | GPU confirmation | Interpretation |
| --- | --- | --- | --- |
| K0 location SSM | Verified/exported; earlier 85.958 s | Not part of this three-case GPU slice | Analytic positive control; new retained mean/variance check also passes |
| QR LGSSM | 128 trials; 34,816 transitions; 50.247 s | 128 trials; 113.036 s | Actual Kalman value/score path, public tuner and checked replay |
| Nonlinear SSM | 192 trials; 52,224 transitions; 205.379 s | 192 trials; 306.540 s | Declared sigma-point approximate likelihood; no exact nonlinear posterior claim |
| Exact-whitened funnel | 128 trials; 34,816 transitions; 56.759 s | Not part of this three-case slice | Analytic supplied-map positive control; no learned-map claim |
| Residual-whitened funnel | 256 trials; 69,632 transitions; 152.835 s | 256 trials; 303.816 s | Supplied imperfect map with controlled residual curvature |
| Separated mixture | 128 trials; 34,816 transitions; 43.807 s | Not part of this slice | Qualified local acceptance; separate posterior check correctly fails |

The mixture's four chains remained in the left mode in all 512 assessed draws
per chain, after a separately archived 128-draw diagnostic prefix. Observed
left mass was 1.0; exact mass is .300000115. Coordinate R-hats were approximately
1.00559 and 1.00490, yet the declared binary mode quantity correctly prevented
posterior qualification. Tuning files remained byte-identical and membership
remained verified. This is a direct regression against conflating local tuning
with global inference; no iid interval is claimed for these correlated draws.

The K0 positive posterior check uses 256 separately archived discarded draws,
2,048 assessed draws per chain, an independent Gaussian posterior calculation,
and four estimated MCSE tolerances for the mean and centered second moment.
It passed its posterior-only modern R-hat and ESS checks. Those window sizes,
ESS floor 128 and four-MCSE tolerance are declared regression hypotheses, not
proofs of sufficient burn-in or finite-sample coverage. Matched SBC and
replicated posterior coverage remain separate work.

Near-unit/small-noise K2/K3, centered funnel, Cauchy and simplex regressions
correctly abstain at a deliberately one-trial cap. The larger inventory also
has bounded four-trial development pricing, including K5/K6. A null
`expectation_met` from pricing is not a passed robustness test. K6 has no
independent joint posterior oracle. Native hard-support -infinity rejection
is still not valid v7 evidence; smooth transformed positive/simplex controls
do not close that separate support-telemetry gap.

### Failures, repairs and accounting

Failed attempts remain preserved. Missing QR provenance/K6 data in a source
snapshot were repaired with exact dependency copies and new output directories.
An early recovery fixture lacked required binding arguments; after repair, a
combined recovery command hit its process ceiling, so all four checks were
run in separately bounded processes with unchanged numerical settings.
The later QR regression accidentally copied epsilon without the corresponding
geometry and starts. Its .968 acceptance correctly failed delivery; copying
the full already frozen configuration fixed the test, and the same regression
seed then passed. There was no band relaxation or search for a lucky seed.

Artifacts are under
`artifacts/hmc-acceptance-decision-repair-2026-10-02/expanded/continuation-01/`:
`final-regression-01`, `model-regression-02/03`, `layout-regression-01`,
`posterior-reference-01`, `ssm-recovery-03`, `calibration-audit-01`,
`model-ladder-01/02/03`, `mixture-posterior-01`, `gpu-parity-01`,
`gpu-delivery-01`, and `book-full-01`. Manifests/configs/launchers preserve
commands, Python environment, seeds, source identities and original attempts.
CPU reference runs deliberately hid GPU devices and bounded TensorFlow/BLAS
threads. GPU runs used trusted execution, XLA, TF32 recorded, float64 inputs,
and verified memory growth before logical-device initialization. GPU delivery
peaked at 613,120 / 628,992 / 412,928 TensorFlow allocator bytes for the three
fixtures; this is allocator evidence, not the device memory limit shown in logs.

CPU charges remain the already debited 28,800 + 43,200 + 28,800 conservative
seconds. The shared ledger has 70,536.184 CPU worker seconds left. The last
allocation allows at most 7,200 elapsed worker seconds with four-CPU affinity;
all its failures and focused retries are included. GPU parity measured 22.691 s
against a 600 s reservation; public GPU delivery measured 727.615 s against
1,800 s. Retain these conservative charges rather than reclaiming capacity
implicitly. The additional charge files leave 67,612.317 GPU seconds uncommitted
after the active SSM reservation and protected allowance. Do not debit the
same nested workers twice or overwrite that active service's ledger.

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep v7 available explicitly | Engineering, controlled-law calibration and scoped public-model delivery pass | No unresolved observed code invariant in completed checks; failed fixtures repaired and preserved | Delivery probability and cost on broader prepared targets | Replicate frozen real-model configurations; retain original denominators and compare against declared references | A new default or arbitrary-target robustness |
| Keep inference separate | K0 known-answer positive and missed-mode negative controls pass | Mixture fails posterior assessment despite good coordinate R-hat and acceptance | General posterior calibration and diagnostic power | Matched simulator/reference SBC and coverage checks where supported | Acceptance or R-hat proves convergence |
| Retain existing defaults | v5/v6 semantics unchanged; official documentation agrees | No default-promotion decision made | Full P4 release evidence remains | Review measured scope-specific evidence before any default change | Completion of every scientific validation gate |

| Inference status | Checked conclusion |
| --- | --- |
| Hard veto screen | No observed raw-law/count/replay/device/reference invariant violation in final results; incomplete attempts are not passes |
| Statistically supported ranking | Only the declared synthetic-cell delivery/detection comparisons with simultaneous intervals; no real-model or sampler superiority claim |
| Descriptive-only differences | Real-model acceptance, cost, wall time and one-seed delivery; CPU/GPU timing under contention |
| Default readiness | Experimental v7 retained; not promoted |
| Next evidence needed | Multi-seed real-model delivery/cost; matched posterior coverage/SBC; separate priced temporal assessment if needed; explicit hard-support telemetry before admitting native support rejections |

Post-run red team: the strongest remaining alternative is that the prepared
small fixtures are easier than BGS or larger SSMs. Whole-bank repetitions can
still be too expensive or encounter additional health vetoes at higher R.
The tests rule out the tested information/accounting/identity errors, not those
cost and geometry limitations. An independently reproduced reference mismatch,
duplicate stream, numerical-health failure or calibrated false-decision excess
would overturn the relevant claim. No untested model gains eligibility from
its name in the inventory.

## Earlier P0--P3 execution record

Earlier P2 result: the combined regression passed **260 tests** with no
failures or skips, including both public exact-score tuners retaining and
exporting two qualified Gaussian candidates. The first held-out calibration
completed **11,264 synthetic searches**. No false qualification or wrong
direction assertion was observed; all seven applicable betting-method delivery
cells passed their predeclared criterion. Those results preceded the expanded execution below; do not use this historical
paragraph as the current status.

## Completed engineering checks

P0/P1 implemented the framework-free trial policy/preflight, bounded-score
intervals, all-observation trial means, per-start temporal diagnostics, and a
shared health evaluator that leaves v5/v6 decisions intact. The combined P1
regression passed 173 checks. A subsequent P2 regression passed 142 checks,
including both exact and position-field recovery, trial resets, cumulative
evidence, altered-record rejection, loss between evidence and receipt,
duplicate streams, and a cap protecting fresh verification.

Two additional Gaussian CPU/XLA integrations passed through the ordinary and
fixed-transport public entry points. Each retained and exported both nominated
interior kernels, reloaded the declared last verification-trial endpoint, and
preserved tuning-seed exclusion. The parameter choices have an analytic
equilibrium-acceptance nomination in the test; actual finite-start qualification
still used independent measurement and verification. This is an engineering
positive control, not a posterior or state-space validation result. The final
combined regression includes the later seed-lineage changes and both positive
exports: `p2/final-regression-02/pytest.xml` records 260 passed, zero failures,
zero skips, in 255.862 seconds. The preceding combined attempt reached its
300-second process ceiling without a terminal JUnit report; it supplies no
claim that the unfinished tests passed.

One earlier regression attempt was invalidated by an implementation edit
during a fresh-process checkpoint test. The source-identity check correctly
stopped it. The unmodified-source rerun passed. The initial P0 subprocess
environment was also replaced with a whitelist, and its failed logs were
redacted. Subsequent reports must not print inherited environment credentials.

## P3 development pricing before held-out calibration

The new diagnostic driver supplies independent known-law trial vectors to the
actual candidate-set controller. It preserves all candidates, supports opposing
per-L repairs, and counts whether any falsely qualified member is retained.
It is not a numerical adapter and cannot issue retained-runner authority.
Thirty-one focused protocol, binomial-reference and controller-driver checks
pass before this pricing command.

The exact development configuration is
`artifacts/hmc-acceptance-decision-repair-2026-10-02/p3/development-pricing-config.json`.
It freezes eleven cells, both the proposed bounded betting method and the
Hoeffding reference, two searches per cell, T=65, W=3, M=8 and the declared
4/16/64/256/1024 repetition ladder. These are development fixtures from the
master's power/cost hypotheses; the two-search count is only for implementation
and cost diagnosis. It cannot establish error rates, delivery rates or a ranking.
Nonstationary Markov truth includes the discarded prefix and finite horizon.

The command uses the `tfgpu` Python, four-CPU affinity, two intra-op threads,
one inter-op thread, `CUDA_VISIBLE_DEVICES=-1`, memory growth enabled and
`BAYESFILTER_PRELOAD_CUSTOM_OP=0`. It invokes:

```text
python -m bayesfilter.testing.acceptance_decision_validation calibrate \
  --config docs/plans/artifacts/hmc-acceptance-decision-repair-2026-10-02/p3/development-pricing-config.json \
  --output docs/plans/artifacts/hmc-acceptance-decision-repair-2026-10-02/p3/development-pricing-01
```

The inner wall ceiling is 180 seconds and the enclosing process ceiling is
240 seconds, convenience bounds charged inside the eight-core-hour P0--P3
allocation. A schema, truth, independence, nonfinite-numerical or denominator
error invalidates the affected evidence and triggers repair. A legitimate
inconclusive candidate does not invalidate the driver. A timeout preserves
raw rows and original planned counts; it is not a failed scientific candidate.

Skeptical audit: direct trial-vector laws and persistent-path laws have their
own exact finite-trial expectations. The Hoeffding reference targets the same
quantity as betting, so their error events are comparable. Historical v5/v6
compatibility and failed lugsail reports have different decision meanings and
remain separately labeled baselines; this pricing slice does not close that
comparison or the full P3 gate. Error rates, useful delivery and work must be
reported conditionally by cell. The two development searches are not holdout
evidence. No acceptance band, confidence allocation or bet grid will be selected
using this pricing output. The artifacts therefore answer the implementation
and affordability question without silently becoming policy-promotion evidence.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Continue P2/P3 | Focused engineering tests pass | No known failing invariant in the last completed regression | Final combined regression and held-out calibration remain | Price the frozen known-law cells, then execute the fundable predeclared calibration | Robustness on arbitrary models or default readiness |

| Inference status | Current evidence |
| --- | --- |
| Hard veto screen | Invalid trials, duplicated streams and reconstruction failures remain explicit |
| Statistically supported ranking | None |
| Descriptive-only differences | Development runtime, trial counts and decision frequencies |
| Default readiness | Not established; legacy default unchanged |
| Next evidence needed | Held-out error/delivery calibration; relevant SSM delivery/cost and GPU parity |

## Frozen held-out known-law slice

The pricing run completed all 22 method/cell combinations, 44 searches total,
in 7.626 seconds. Its decision and work counts are descriptive only. Simple
linear scaling gives about 32.5 minutes for 512 searches per cell, including
the pricing run's compilation overhead; this is a planning estimate, not a
runtime guarantee. The Hoeffding reference did not deliver on the heterogeneous
valid-start cell at this replication cap, which must stay visible in the
conditional report rather than being pooled away.

The frozen next slice uses `p3/heldout-config.json`: the same eleven cells,
methods, bands, error allocations, M=8, T/W and repetition ladder, with 512
searches per cell and a disjoint seed namespace. It has a 3,540-second inner
wall ceiling, 3,600-second process ceiling and 3,610-second service ceiling.
With four-CPU affinity the process allocation is at most four core-hours.
These convenience ceilings allow roughly twice the measured scaled runtime
and remain within the initial eight-core-hour P0--P3 allocation after the
earlier bounded development commands. GPU devices are intentionally hidden.

The twelve-file source snapshot is `p3/heldout-source-01`; its ordinary SHA-256
manifest records the exact controller, statistics, generator, CLI and import
dependencies. The output is `p3/heldout-01`. The service is
`bayesfilter-hmc-acceptance-calibration-20261002-r1.service`. This snapshot lets
subsequent implementation work proceed without invalidating the calibration
source. The run manifest, raw search rows, per-cell progress and terminal
result preserve the original planned denominator, completed trials, all
retained candidates, supported direction errors and false membership.

The pass/fail interpretation remains the main plan's: simultaneous error-rate
intervals are compared to the declared family levels, and useful-delivery lower
bounds must reach .80 on declared easy cells. Exact qualification-boundary
cells need not deliver. Separate search and verification error allocations are
not a joint .05 guarantee. A small number of observed errors alone does not
prove calibration, and this slice does not close M=100 stress, historical
comparator interpretation, temporal-diagnostic power, real SSM delivery or GPU
parity. No default promotion follows from this slice alone.

Skeptical audit before launch: independent search replications and fresh stage
streams are explicit, finite-start truth includes W/T, repeated-look spending
is method-specific, all-member retention uses the actual controller, and a
simple same-target concentration baseline is present. The development output
changed neither scientific settings nor holdout seeds. Wall exhaustion yields
missing allocations, never imputed failures or successful candidates. The
slice therefore answers its limited calibration/delivery question under the
declared bounded random-vector laws.

## Terminal held-out result and audit

All 22 method/cell combinations finished in 978.435 seconds without a budget
stop. Each has 512 of its 512 planned searches. Across all combinations there
were zero searches with a false verified member, a wrong search direction, or
a wrong verification assertion. For each of these rates the simultaneous
Clopper--Pearson interval is [0,.0158232], rounded outward, using the prespecified
.05 family allocation across 88 cell/event intervals. These observations do
not prove the mathematical bound or generalize to untested stochastic laws.

| Cell | Betting useful delivery | Hoeffding useful delivery | Terminal interpretation |
| --- | --- | --- | --- |
| Interior | 512/512 | 512/512 | Both initial members verified in every search |
| Heterogeneous valid starts | 512/512 | 0/512 | Hoeffding delivery fails at this cap; its candidates remain inconclusive |
| Misleading pooled mean | N/A | N/A | Every candidate requires preparation review; no member verified |
| Preferred lower boundary | 512/512 | 512/512 | Broad-band qualification does not require narrow-band equivalence |
| Preferred upper boundary | 512/512 | 512/512 | Same two-band distinction |
| Qualification lower boundary | N/A | N/A | All candidates remain inconclusive at cap |
| Just outside qualification | N/A | N/A | No member verified; betting reports 99 preparation violations and 925 inconclusive candidates, Hoeffding 1,024 inconclusive candidates |
| Rare extremes | 512/512 | 512/512 | Both initial members verified despite nonconstant law |
| Opposite L repairs | 512/512 | 512/512 | Both L families repaired and verified in every search |
| Persistent rho=.995 | 512/512 | 512/512 | Independent trial rows preserve the declared finite-horizon mean |
| Initial transient | N/A | N/A | Supported high-acceptance proposals in every search; epsilon-insensitive law never yields a qualified child |

For a 512/512 delivery result the simultaneous interval is [.9841768,1],
rounded outward. For 0/512 it is [0,.0158232]. The former exceeds the declared
.80 lower-bound requirement. The latter fails it. N/A means the predeclared
useful-delivery event does not apply, not a missing search or a zero success
rate. The raw review also checks both opposite-L families, although that cell's
original delivery statistic required at least one verified child. No new
criterion was substituted for the original held-out statistic.

The exact raw-law target is each start's finite-trial mean under the declared
W=3/T=65 protocol. The transient target is approximately .910701, not its
stationary .70 limit. Wrong-direction and wrong-membership events use this
finite-trial truth. The evidence therefore addresses the specified acceptance
assertions, not stationary acceptance or posterior exploration.

The existing `complete_trials` field counts distinct trial vectors observed by
the controller. It is not the number generated: the diagnostic generator
creates all 1,024 vectors for each new candidate/stage before feeding prefixes.
`p3/terminal-review-01/result.json` independently reconstructs observed and
generated counts from raw rows, validates consecutive nonoverlapping trial
ranges, and preserves both. Observed-vector work may price a future numerical
implementation, but neither this count nor synthetic generator wall time is
actual HMC cost. For example, the betting interior median is 1,024 observed
vectors per complete two-candidate search. At four starts and W+T=68, that
would require 278,528 transitions before accounting for leapfrog cost. P4 must
measure affordability rather than infer it from this cheap synthetic run.

The calibration used a frozen source snapshot. The current statistics moved
pooled summaries, covariance, interval intersections and temporal contrasts
into stable TensorFlow graphs after that snapshot. A deterministic CPU/XLA
comparison checked both interval methods, three stages, five rungs and ten
fixtures: **300 comparisons, identical decisions, 34,820 compared numeric
fields, maximum absolute difference 7.22e-16**. It also checked an odd-length
trial summary. The predeclared tolerance was 1e-12 absolute plus 1e-12 relative.
This supports transfer of the tested numerical calculations, not a formal
floating-point theorem or GPU parity. The exact command, environment, source
hashes, data checks, plan and result are in `p3/terminal-review-01/manifest.json`,
`intent.md`, `review.py` and `result.json`. The diagnostic took 25.935 seconds
with GPU devices intentionally hidden; its CUDA no-device log is not a GPU
readiness result.

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Continue experimental betting-policy validation | Tested error events show no violation; all seven applicable delivery criteria pass | Raw counts, original denominators and checked CPU numerical parity pass | M=100 stress, adversarial laws, temporal power and real-model cost remain | Complete the remaining P3 cases and P4 state-space/funnel integrations | Universal calibration, posterior correctness or default readiness |
| Keep Hoeffding as a correctness/cost reference | No observed assertion error; heterogeneous-start delivery fails | No evidence corruption found | More repetitions may recover delivery but increase cost | Preserve failed delivery and compare measured model work | Practical suitability at the present cap |

| Inference status | Terminal evidence |
| --- | --- |
| Hard veto screen | No failed raw-count, source-check or parity invariant in this slice; synthetic laws do not test numerical HMC health |
| Statistically supported ranking | Betting has higher qualification delivery on this heterogeneous-start cell under the declared simultaneous intervals; no general sampler ranking |
| Descriptive-only differences | Other observed trial counts, runtime and candidate-state frequencies; no inferential cost comparison was predeclared |
| Default readiness | Not established; v5/v6 remain unchanged and v7 is experimental |
| Next evidence needed | Many-candidate/adversarial calibration, per-start temporal diagnostic power, actual SSM/funnel delivery and cost, GPU/XLA parity |

Post-run red team: the strongest alternative explanation for easy delivery is
that the chosen bounded laws are easier than real finite-start HMC, especially
under residual geometry or health vetoes. Passing Gaussian public integrations
does not settle that issue. Real-model underdelivery or unaffordable work would
overturn practical-readiness expectations without falsifying the synthetic
statistics. The weakest current link is the unmeasured P4 cost/delivery. The
budget ledger already charged the full eight-core-hour P0--P3 allocation;
the calibration and terminal diagnostic are inside that charge. This repair
used no GPU time and did not change the separate running SSM source or budget.
