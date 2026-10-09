# HMC execution-budget repair result

The C1 cost increase exposed a real execution mismatch: its development wrapper
enabled leapfrog graph reuse, but the isolated validation executor omitted that
option. Explicit propagation now reaches the public CLI, cell coordinator,
isolated child and numerical pipeline. The choice is recorded and cannot change
silently on resume. The paired GPU checks preserve every saved numerical array,
candidate decision and posterior decision.

C2 also had an allocation problem. Its final quarter/half fits received only
214.792/36.620 seconds from their cells despite declared cumulative fit
allowances of 750 seconds. Unused time remained in the enclosing reservation.
The optional sequential sharing policy now carries unused settled-cell time
forward, preserves unstarted allocations and charges all previous attempts.
Receipts distinguish a cell cutoff from an exhausted fit. A progress-only
preparation directory is archived before an eligible paid restart; it is never
represented as a resumable numerical checkpoint.

## Evidence and interpretation

All six development fits used the same frozen ordinary-HMC source, GPU UUID,
TF/TFP GPU/XLA path and verified memory growth, with profiling off. The two
matched pairs used identical designs and seeds. Runtime is descriptive; the
primary engineering criterion was exact numerical/evidence parity.

| Target and development identity | Graph reuse off, seconds | Graph reuse on, seconds | Matched evidence |
| --- | ---: | ---: | --- |
| Gaussian, first | 671.072 | 272.021 | Exact parity: 82 tensors, 150 observations, 21 verified members, geometry, seeds and posterior decisions |
| Beta-binomial, first | 584.443 | 248.072 | Exact parity: 92 tensors, 165 observations, 11 verified members, geometry, seeds and posterior decisions |
| Gaussian, second | Not run | 268.592 | Complete independent development fit |
| Beta-binomial, second | Not run | 254.480 | Complete independent development fit |

The declared max-of-two cost rule gives **134,904.326 GPU seconds (37.473
hours)** for all 512 C1 confirmation fits, versus the earlier unrepaired
314,166.757-second forecast. The comparison between those two forecasts also
changes profiling and source, so it is not a controlled performance comparison.
The paired rows above isolate graph strategy while preserving the numerical
procedure. Two pairs do not establish a runtime-tail guarantee or sampler ranking.

The full C1 confirmation launched at **2026-09-25 15:02:36 Shanghai time** as
`bayesfilter-hmc-c1-repaired-20260925-r1.service`. Its original 256 Gaussian and
256 beta-binomial fits, seeds, sample policies, acceptance/health tuning rules,
candidate retention and pointwise .90 delivery/coverage/joint screens are
unchanged. Cell allocations are 85,496 and 79,984 seconds, with optional sharing.
The 165,600-second enclosing limit leaves 30,695.674 seconds above the observed
forecast. Systemd bounds the entire process group at 165,610 seconds plus one
second of shutdown. The coordinator and first child confirm both execution
flags, profiling off, selected-GPU placement, XLA and memory growth. Final
scientific assessment remains pending.

Original-source C2 recovery completed the quarter fit in 51.395 seconds.
The half fit first encountered an unfinished preparation directory; that failed
attempt was preserved and charged. Its next attempt used the remaining original
allowance. At the final stop it had 67 numerical observations and five verified
candidates, but no posterior assessment. Its cumulative attempts cost 750.366
seconds, including the recorded shutdown overrun. Twenty-five of 26 workload
samples reported another GPU process. The tuning was making progress; neither
the cell-allocation defect nor contention alone explains its total cost.

| C2 cell | Completed/planned fits | Alarms | Conservative exact 95% interval | Original rate screen |
| --- | ---: | ---: | --- | --- |
| Null baseline | 128/128 | 3 | [.004860, .066966] | Pass |
| Quarter defect | 64/64 | 64 | [.943991, 1] | Pass |
| Half defect | 63/64 | 63 | [.915990, .999604] | Pass; one unavailable |

Thus C2 has **255/256 complete** and passes the three conservative rate screens,
but it fails the declared complete-study requirement. The missing slot remains
in the denominator. Its allowance is exhausted and is not reset. Original
evidence checksums pass; the historical 9/256 study is not pooled.

## Verification and source boundaries

The latest regressions cover CLI-to-child propagation, actual isolated Gaussian
and beta-binomial array parity, budget dispatch, resume identity, cumulative
costs and preservation of interrupted preparation. Missing imported fixtures
caused two initial isolated-test failures; those were repaired and the affected
checks passed. The final implementation review also found and fixed a resume
edge case: a later interrupted reservation could escape accounting when
`max_jobs` truncated the dispatch scan. All interrupted reservations now settle
before that filtering, and their spending cannot be reclaimed by an earlier
retry. The focused regression repeats the retry to test idempotence.

There are **70 distinct passing current regression cases**, plus two successful
terminal-accounting checks for existing and missing receipts. These checks
preserve previous debug charges and charge a stopped confirmation only once.
The official book builds without undefined references or citations; physical
pages 433–434 were rendered and inspected. The API reference matches its
execution-policy description. No numerical or inference default was changed.

The active confirmation uses immutable `source-confirmation-r1`, copied from
tested source-r4. Pricing used source-r3; their only difference is classification
of a preparation-budget exception in the retry archival branch, which none of
the pricing fits entered. The later interrupted-reservation regression runs on
source-r5 and repairs the working tree. The fresh confirmation has no resumed
coordinator and no automatic retry, so that branch cannot affect this run.
Its source must not be edited or silently changed for a later resume.

The repair used 767.645 seconds of the old C2 reservation and 2,302.358 new GPU
seconds for six-fit pricing. CPU receipts, including failed checks and the book,
are reconciled in the old ledger. Before confirmation, the additional grant had
175,300.798 seconds remaining; 165,611 seconds are reserved for its full cgroup
ceiling and shutdown. The remaining reservation and final settled balance are
kept separate. Unrelated NeuTra/Q20 and book changes remain outside this source.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep explicit graph reuse | Both full-fit pairs have exact numerical and evidence parity | No mismatch in tested pairs | Runtime tails and other targets | Execute full unchanged C1 inventory | Universal speedup or a new sampler default |
| Keep budget repair | Dispatch, retry and cumulative-accounting regressions pass | No unresolved tested accounting failure | Untested operational failures | Preserve receipts and audit terminal execution | Every fit finishes within the budget |
| Retain C2 as incomplete | Conservative rate screens pass; 255/256 fits complete | Completion requirement fails | Last missing posterior outcome | Preserve failure and exhausted allowance | C2 closure or rejection of the scientific procedure |
| Launch C1 | Measured complete-workload forecast fits existing ceiling and grant | Source/hardware checks pass | Coverage, delivery and future runtime | Review all 512 planned outcomes at termination | Calibration from development or launch success |

| Inference status | Evidence |
| --- | --- |
| Hard veto screen | No paired numerical mismatch; the final C2 fit is unavailable after its full allowance. |
| Statistically supported ranking | None; graph strategy preserves the tested numerical outputs. |
| Descriptive-only differences | Development runtimes and resource telemetry. |
| Default-readiness | No inference, numerical or learned-map default is promoted. |
| Next evidence needed | Full C1 delivery, coverage and joint intervals; terminal source, health, missingness and accounting review. |

The strongest alternative explanation for the original cost increase is the
combination of profiling, seed-dependent search and machine load. The paired
checks isolate graph strategy on two development identities but do not explain
every historical timing. The weakest evidence is the two-prices-per-target
runtime forecast; late costly fits may still exhaust the ceiling. A numerical
mismatch or bad downstream assessment in confirmation would overturn any
broader readiness claim. The repair establishes tested execution behavior;
posterior calibration awaits the full confirmation.

Exact commands, source hashes, seeds and hardware are preserved under
`artifacts/hmc-budget-debug-2026-09-25/`: `confirmation-launch.json`, each
enclosing `manifest.json`/`execution.json`, child manifests, `verification.json`,
`gpu-prices-r1/data/result.json`, `c2-terminal-fit-diagnosis.json` and the
confirmation output. The governing plan is
[the budget-debug plan](bayesfilter-hmc-budget-debug-2026-09-25.md).
