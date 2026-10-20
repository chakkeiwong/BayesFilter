# Fixed-HMC continuation: interim inspection, 14 October 2026

Update at 23:56 China time: member 7 passed after 2,000 warm-up and 8,000
retained transitions per chain, bringing delivery to six of eight. Member 8
has two committed chunks and is running its third. The 23:53 audit in
`closeout-preparation-01/interim-audit-01.json` verifies 1,738 source/input hashes,
1,464 tensor checksums and 61 chunks. A detached read-only observer will audit
and settle the cohort after the numerical service exits; 41 focused regression
tests cover its bookkeeping and failure handling. The [closeout plan](bayesfilter-modern-hmc-cohort-closeout-2026-10-14.md)
records the evidence contract and next complete randomized-HMC baseline. No
further numerical allocation has been made. The earlier inspection below is
preserved with its capture time; it is not the latest candidate count.

At 22:58 China time, five of the eight verified fixed-HMC candidates have
passed their declared posterior and finite-target reference checks. One has
finished at its retained-sample cap without sufficient precision. The seventh
is sampling and the eighth remains queued. The detached service
`bayesfilter-modern-fixed-resume-20261014-01.service` remains active.
This is an interim result for the full predator–prey SGQF target and one seed.

| Candidate | L | Epsilon | Retained draws per chain | Current disposition |
| --- | ---: | ---: | ---: | --- |
| 1 | 1 | 1.0 | 10,000 | Posterior and reference checks passed; carried unchanged |
| 2 | 2 | 0.5 | 10,000 | Precision and reference screens not met |
| 3 | 2 | 1.0 | 6,000 | Posterior and reference checks passed |
| 4 | 4 | 0.5 | 3,000 | Posterior and reference checks passed |
| 5 | 4 | 1.0 | 5,000 | Posterior and reference checks passed |
| 6 | 8 | 0.5 | 3,000 | Posterior and reference checks passed |
| 7 | 8 | 1.0 | Pending | Nine committed chunks including warm-up; next chunk running |
| 8 | 16 | 0.5 | Pending | Queued |

The second candidate passed numerical health, R-hat and bulk/tail ESS checks.
Its maximum modern R-hat is 1.001273, minimum bulk ESS is 4,183 and minimum
tail ESS is 4,256. The failed quantity is `second_5`, the square of the sixth
source coordinate. Its lugsail MCSE/SD ratio is 0.030111 versus the frozen
0.02 threshold. The separate reference screen is 0.128873 versus 0.1. The
absolute reference discrepancy is 0.015698 and the estimated MCSE is 0.029844;
the failed uncertainty-adjusted screen does not establish posterior bias.
This candidate therefore fails the declared posterior delivery criteria at
10,000 retained draws per chain. Its verified tuning membership remains valid.

No implementation or harness failure has been identified in this inspection.
Preserve the failed terminal result and continue the remaining candidates.
Changing the cap, precision estimator, reference tolerance or tuning parameters
to rescue this result would change the frozen comparison. Further precision
research belongs to a separately declared phase after the complete cohort is
audited and the remaining budget is settled.

The standard-library audit at
`docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/priced-comparison-01/comparator-continuation-01/progress-audit-02/result.json`
passes 1,738 original source/input checksums and 1,392 tensor checksums across
58 committed chunks. It verifies the complete candidate inventory, original
member-index seeds, policies, frozen numerical imports, completed decisions,
reference rules and first-member carry-forward. It does not recompute numerical
diagnostics or establish convergence. Its command is `python` followed by the
adjacent `audit.py`; execution took 1.055 seconds and is charged to the held
local reserve. The run consumed approximately 5.22 hours at inspection, with
approximately 41.78 hours still reserved under its 47-hour inclusive cap.
No unused reservation is released before terminal settlement. The run's launch
manifest preserves the exact numerical command, source, environment, GPU
identity, memory-growth policy and original seeds. The active scientific plan
is [the October 14 continuation plan](bayesfilter-modern-hmc-posterior-resume-2026-10-14.md).

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Continue the active cohort | Five delivered, one terminal failure, two unfinished | No recorded hard numerical veto in completed members; source and checkpoint audit passes | Unfinished members and repeated-look precision uncertainty | Finish members 7–8, audit terminal cohort, settle costs and refresh the next funded phase | All-member success, posterior bias, method superiority or release readiness |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Completed members have no reported hard numerical veto; active members remain unassessed |
| Statistically supported ranking | None; one target/seed and an unfinished cohort |
| Descriptive-only differences | Retained counts, R-hat, ESS, MCSE margins and runtime |
| Default-readiness | Not established; no default changes |
| Next evidence needed | Terminal full-cohort assessment, followed by the master's broader paired-model/seed and stopping-coverage work |

The strongest alternative explanation for the second member's failure is
Monte Carlo variability or slow estimation of a second moment, rather than an
incorrect invariant target. Independent replicated estimates with calibrated
uncertainty could distinguish these explanations. The current diagnostic and
reference screens have repeated-look limitations, and a passing local screen
does not establish global exploration. No resource or scientific continuation
veto has fired for the remaining cohort.
