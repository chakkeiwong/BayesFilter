# HMC master repair execution

Date: 2026-09-17. Program:
[HMC repair master](bayesfilter-hmc-repair-master-program-2026-09-16.md).
Detailed review:
[code and guide audit](bayesfilter-hmc-tuning-code-guide-audit-2026-09-17.md).
Status: the bounded M0--M6 execution, terminal audit and guide build are complete.
All launched numerical workers have ended. M7 remains open with its next work
specified in the master program; broad calibration is not declared complete.

## Outcome and scope

The public tuning procedure remains a shared candidate-set search. It retains
every verified pair, uses fresh evidence for each repair, and keeps posterior
diagnostics separate. R-hat, ESS, mode occupancy and MCSE are not tuning
requirements. The bounded repair adds an explicit exploration-cap option,
useful preparation failure causes, accurate incomplete-work reporting, and
declared posterior mode quantities. It also fixes additional controller,
retained-replay and assessor problems found by counterexample during execution.

No numerical default was promoted. The experiments below do not establish
general automatic-tuning calibration, posterior accuracy across all targets,
or discovery of unknown modes. Remaining work is explicit in M7 and the audit.

## Evidence obtained

| Experiment | Observed result | Interpretation |
| --- | --- | --- |
| Preserved empty search, same geometry and starts | Ceiling .5: zero verified members among six candidates. Explicit ceiling 1.0: 18 verified members among 100 candidates. Both bounded schedules ended. | The wider finite domain contains useful measured hypotheses for this development case. Identities/streams differ; this is not a matched-trajectory comparison or default validation. |
| Preserved missed-mode posterior | Original local checks passed. Reassessment with `P(x<0)` fails; estimated probability is zero and lugsail precision is unavailable. | Reproduces and rejects this false precision claim for the declared quantity. No sampling was rerun and no unknown-mode guarantee follows. |
| Actual acceptance screen with synthetic marks | All 640 declared single-pair searches completed, 64 per mean/dependence cell. | Measures this operational screen's decision rates, not HMC correctness or sequential confidence coverage. |
| CPU ordinary pilot | Two datasets, four independent fits complete; every selected member passed posterior checks. | Cost and availability development check; separate from fresh SBC. |
| Fresh CPU ordinary SBC | Six of six datasets, 12 of 12 independent fits complete. All fits found verified members and all selected posterior checks passed. | No discrepancy detected in this very small normal-conjugate design; insufficient for a general calibration claim. |
| GPU/XLA ordinary pilot | Two datasets, four fits complete after one checkpoint resource retry. All selected posterior checks passed. | Demonstrates this frozen implementation/device path and measures cost. The failed attempt remains charged. |
| Fresh GPU/XLA ordinary SBC | Four of four datasets and eight of eight independent fits complete. Every fit found verified candidates and every preselected posterior passed its declared checks. | No discrepancy detected in this small normal-conjugate development design. All 138 verified members are preserved; eight were assessed and 130 remain unassessed for posterior accuracy. |
| Mixture, single-mode versus known-mode dispersed starts | Two replications per regime; all four searches complete. The regimes retained 41 and 39 verified candidates respectively. Only the two preselected members per regime underwent posterior assessment. All four reached the 10,000-transition warmup cap with no retained output. | The declared posterior policy does not grant readiness for these cases. This does not establish a sensitivity rate, an accuracy ranking, or successful mixing. The other 76 members remain unassessed for posterior accuracy. |

All artifacts are under
[`hmc-repair-master-2026-09-16/`](artifacts/hmc-repair-master-2026-09-16/).
The key saved diagnoses are
[`comparison.json`](artifacts/hmc-repair-master-2026-09-16/saved-search-r1/comparison.json),
[`saved-mode-r1.json`](artifacts/hmc-repair-master-2026-09-16/saved-mode-r1.json),
and the complete
[CPU aggregate](artifacts/hmc-repair-master-2026-09-16/fresh-cpu-r1/master-fresh-cpu-aggregate.json)
and [GPU aggregate](artifacts/hmc-repair-master-2026-09-16/fresh-gpu-r2/master-fresh-gpu-aggregate.json).

The fresh CPU parameter, bounded-radius and log-likelihood rank p-values are
0.8885, 0.403 and 0.875, against the declared within-family threshold 0.05/3.
With only six datasets and two independent fits per dataset, those
non-rejections do not establish accuracy. The exact interval for complete
dataset availability is approximately [0.541, 1]; its width alone rules out a
precise availability claim.

The corresponding GPU rank p-values are 1.0, 0.136 and 1.0, with the same
within-family threshold and four datasets. Its complete-dataset availability
interval is approximately [0.398, 1]. CPU and GPU cohorts are reported
separately; neither the pilots nor changed-source runs are pooled with them.

### Acceptance-screen calibration

| True mean | Independent marks: qualified/64 | Persistent marks (.9): qualified/64 |
| --- | ---: | ---: |
| .54 | 0 | 0 |
| .64 | 29 | 20 |
| .70 | 64 | 28 |
| .76 | 18 | 37 |
| .86 | 0 | 0 |

At mean .70, the exact binomial intervals are [0.944, 1] for independent marks
and [0.314, 0.567] for persistent marks. The persistent cell has 36 inconclusive
outcomes at the evidence cap. Zero of 64 has interval [0, 0.056], not zero
uncertainty. The concentration, dependence regimes, means, seeds and evidence
rungs were fixed in the plan before launch. Qualification slightly outside the
practical band is consistent with the interval-compatibility rule; it is not
itself a violation of that rule. Continuous differences and unexplored regimes
remain descriptive. There is no stochastic candidate ranking.

The mode-assessor repair was checked on saved records without changing them.
[`saved-mode-denominators-r2.json`](artifacts/hmc-repair-master-2026-09-16/saved-mode-denominators-r2.json)
now shows the declared probability as unavailable in two of two replications
for each regime. This corrects the report denominator; it supplies no new
posterior samples or accuracy evidence.

## Engineering checks and source provenance

The terminal controller/replay/validation regression batch passed 231 tests.
After the additional assessor repair, its focused batch passed 32 tests.
The final nonfinite preparation-reporting regression failed before its repair;
the following preparation/documentation batch passed 33 tests. Failed numerical
diagnostics now remain explicit in strict JSON without masking their original
exception.
Replacing a simplified fixture with the actual public bootstrap-round type
also reproduced loss of its `classification` field in the progress summary.
That field is now preserved; the final focused preparation batch passed eight
tests. Both failing counterexamples remain saved.
Earlier batches passed 83, 231, 124 and 12 tests. These overlap and must not be
summed as unique coverage. JUnit files, including the deliberate failing
counterexamples, are preserved under the artifact root.

The tests reproduce malformed-observation recovery, later shared invalidity
against an older member, truthful search-cap reporting, both preparation
failure stages, global quantity precision/denominators, candidate-local failure,
geometry composition, fresh verification, retained continuation, public route
conflicts and validation aggregation. They include real small Gaussian
transitions and independent numerical formulas. They do not test realistic
consumer targets at scale. Two commands referenced nonexistent test files and
ran no tests; their corrected executions are recorded separately.

All long workers used immutable
[`source-r1/`](artifacts/hmc-repair-master-2026-09-16/source-r1/), source identity
`83671e17a3a1321a5aece9419214ea013f1c154bc08e2fab8f9695d49dfe8d7f`, based on
Git `d86dadf68ea57772642c6802990f46a3c6a04c30` plus the recorded working changes.
Subsequent terminal repairs and concurrent work mean that this snapshot is
historical relative to the final checkout. Its scientific observations remain
attached to that exact implementation. The new failure-path regressions test
the terminal implementation; they do not upgrade older numerical runs.

GPU workers used visible device 1 (RTX 4080 SUPER), TensorFlow 2.20.0, TFP 0.25.0,
Python 3.13.13, XLA, float64 target calculations and recorded TF32 policy.
Memory growth was configured and verified before logical-device initialization.
CPU reference/test processes deliberately hid GPUs. Each job records its
resolved design, seeds, command, environment, source, runtime, result checksum
and attempt history. No performance ranking is made from these concurrent runs.

## Repairs, retries and decisions

The CPU pilot cost 750.449361600 worker-seconds. One fresh shard reached its
800-second process cap after all four member results were saved; a 150-second
extension completed finalization in 15.149446783 seconds. The GPU pilot reached
its 1,500-second cap; a 1,000-second extension finished the same data and streams
in 629.680483735 seconds. The resource extension script preserves prior charges,
checks frozen source/design identity and uses existing numerical checkpoints.
It does not replace failed datasets or count retries as new replications.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep finite expansion optional | Saved-failure mechanism and public-path regressions pass | No relaxed acceptance/health rule | Limited models and development streams | Broader fresh ordinary fits with measured resource margins | Universal epsilon cap or new default |
| Keep declared global posterior quantities | Saved false precision is rejected; missing outputs stay visible | All four new mode posteriors fail readiness | Few replications; other modes may be unknown | Target-specific global exploration evidence | Tuning candidate failure or successful mixture inference |
| Retain screen calibration as diagnostic | All 640 searches recorded | No synthetic result receives numerical authority | Dependence/model/process regimes not covered | Actual-HMC and nonstationary screen calibration | Nominal sequential confidence coverage |
| Accept reproduced engineering repairs | Failure-before/fixed-after checks pass | Malformed data and shared invalidity cannot grant replay | Unenumerated failure histories and consumer dependencies | Preserve regressions and stage further refactoring | Whole-library correctness |
| Leave broad validation open | Small normal-conjugate fits only | Missing realistic references and weak-defect power | Cross-target reliability and calibration | M7 with concrete target/reference and power design | Production/default or all-model readiness |

| Inference status | Verdict |
| --- | --- |
| Hard veto screen | Known shared execution invalidity blocks replay; missing/invalid precision blocks declared posterior readiness. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Candidate counts, runtimes, small SBC non-rejections and start-regime outcomes. |
| Default-readiness | Not established; defaults unchanged. |
| Next evidence needed | Broader automatic preparation, sensitivity to subtle real-kernel defects, stopping-time coverage and matched realistic consumer references. |

The failures do not invalidate the research direction. The preparation-ceiling
case triggered a finite search repair; timeouts triggered resource repair; the
mode cases reject the assessed posterior candidates and motivate global
exploration work. They do not revoke otherwise valid tuning membership.

The strongest alternative explanation for the fresh normal-conjugate success
is that this easy target and these streams happen to fit the optional envelope.
Further target failures would overturn any broader availability interpretation.
The weakest evidence is statistical power and realistic-model transfer, not
whether the regression tests execute. The master program keeps these gaps open.

## Guide and final accounting

The Markdown reference and LaTeX tuning chapter were updated. The final clean
staged build produces the [563-page official guide](../main.pdf)
with tuning and diagnostics citations resolved. Physical pages 413, 420, 426
and all four coverage-table pages 427--430 were visually inspected: the changed
text, code names, references and table columns remain readable. Three pre-existing
citation keys in another chapter remain unresolved (`Afshar2015`,
`Gorinova2020`, `Pakman2014`); no full-book warning-free claim is made.

The official source is [docs/main.tex](../main.tex). Following the owner's
September 17 clarification, the verified PDF is installed at `docs/main.pdf`,
and active links use that location. The dated artifact directory preserves the
historical build and its evidence; it is not a separately maintained guide.

The [build manifest](artifacts/hmc-repair-master-2026-09-16/guide-r1/build-manifest.json)
records the clean stage, command, source hashes, 19.367497-second elapsed time,
PDF/log hashes and rendered pages. The
[coverage render record](artifacts/hmc-repair-master-2026-09-16/coverage-render-r1.json)
preserves all 23 prior report sources plus the six master reports. Source status
was recomputed against the terminal checkout: frozen experiments are historical
relative to the later repairs. Their observations remain useful for their exact
source; the generated table does not count them as current-source executions.

| Charge | CPU worker-seconds | GPU worker-seconds |
| --- | ---: | ---: |
| Prior campaign, carried forward | 78,475.826496 | 40,224.738057 |
| Master indexed jobs, including failed attempts | 3,190.574827 | 7,240.845734 |
| Master tests, saved diagnostics, final build and other overhead | 1,758.616474 | 120.000000 |
| Cumulative charge | 83,425.017797 | 47,585.583791 |
| Remaining from each original 86,400-second allowance | 2,974.982203 | 38,814.416209 |

The continuation used 4,949.191302 CPU seconds and 7,360.845734 GPU seconds,
within its own 7,200/10,800-second ceilings. Cumulative charges are 23.173616
CPU hours and 13.218218 GPU hours. The original allowances are not reset.
JUnit suite times total 588.981 seconds, including failures and zero-test
commands. The saved diagnoses and final book build are measured separately.
Untimed imports, earlier builds, inspections, coordinator work and final
reporting receive an explicit conservative charge of 1,000 CPU and 120 GPU
seconds. These two overhead charges are accounting estimates, not measured
device utilization or statistical evidence.

The [execution inventory](artifacts/hmc-repair-master-2026-09-16/execution-evidence-r1.json)
preserves commands/designs, attempt histories, fit selection, test files and
source identity. The [terminal accounting](artifacts/hmc-repair-master-2026-09-16/terminal_audit.json)
and independent [saved-record reconciliation](artifacts/hmc-repair-master-2026-09-16/terminal-integrity-r1.json)
agree: 19 complete indexed results, 64 tensor checksums, zero invalid artifacts,
no running/reserved work and no exceeded allowance. Both pilots and all fresh
SBC datasets are complete in their separate inventories. These checks establish
record consistency; the statistical limits above remain.

Terminal decision: accept the reproduced engineering repairs and close this
bounded execution. Continue M7 from the true remaining ledger with a resolved
power/target/reference design. The known mode failures reject those assessed
posteriors; they do not reject HMC as a research direction or revoke valid
tuning membership. Further default calibration, realistic consumer references,
global exploration and staged preparation extraction remain unfinished.
