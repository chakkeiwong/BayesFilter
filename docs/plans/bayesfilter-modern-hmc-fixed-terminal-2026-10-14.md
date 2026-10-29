# Fixed-HMC SGQF comparator: terminal result

14 October 2026. The experimental frozen-design fixed-HMC comparator reached
its total wall-time allocation at **11:21:01 China time**. Tuning completed:
all eight primary candidates passed independent acceptance/health verification
and were retained. One member completed the declared posterior and independent
finite-target reference screens; the second was interrupted and six had not
started posterior sampling. The comparison and master program remain incomplete.

This run used immutable source `e68cdf1c4c6b75c16f92b86a3363a1e24690d22c`,
the full predator–prey SGQF finite target, the same prepared geometry, starts,
reference and posterior criteria as the completed MAMS cohort. The design hash
is `a3177daa578051447b7b25c8ab4b506124b16a213a7696d8b1b3a9f43bcfb306`.
It used TensorFlow/TFP on GPU 1 with XLA and verified memory growth. The
executed command, environment, data/source hashes and seeds are retained in the
[run manifest](artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/priced-comparison-01/comparator-01/fixed_hmc-run/manifest.json).
The tuning evidence is in that run's `search/tuning/tuning_checkpoint.json`;
all eight retained-member exports are in `search/members/`. NUTS was excluded.
This is a one-target, one-seed feasibility result, not a performance ranking or
production/default promotion.

## Saved results and audit

The [terminal audit](artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/priced-comparison-01/comparator-01/terminal-audit-01/result.json)
checked all 197 declared source/input hashes, 896 saved qualification calls,
the complete verified-member inventory and its exported identities, and 456
tensor checksums across 19 committed posterior chunks. All 1,540 files under
the original numerical run were unchanged. This was a CPU artifact audit;
it did not rerun the sampler or independently recompute numerical diagnostics.
Its recorded command and source are in `terminal-audit-01/audit.py` and
`result.json`.

| Candidates, in creation order | Terminal evidence | Disposition |
| --- | --- | --- |
| 1: L=1, epsilon=1 | 2,000 warm-up and 10,000 retained transitions per chain; posterior checks and all 18 finite-target reference screens passed | Completed under the declared rules |
| 2: L=2, epsilon=0.5 | Seven committed 1,000-transition chunks per chain; no terminal posterior assessment | Resume committed prefix and finish assessment |
| 3–8 | Tuning verified; posterior work not started | Preserve all six in the continuation inventory |

For the completed member, the largest reported modern R-hat was 1.003064 and
the largest normalized reference-screen value was 0.072443, below its declared
0.1 threshold. No recorded posterior hard veto fired. Warm-up draws were
excluded from posterior estimates. These are posterior checks; they did not
determine tuning membership. The seven incomplete members cannot be classified
as posterior passes or scientific failures from the available evidence.

The stop reason was `supervisor_budget_exhausted`, followed by SIGKILL at the
allocation boundary. The systemd service therefore reports failed even though
this was the intended resource stop. TensorFlow retracing warnings appear in
the log; they are an explanatory performance diagnostic, not the recorded
termination cause. No new numerical default follows from those warnings.

The audit found a reporting defect: interruption finalization inherited
`posterior_assessment=not_run` from the search summary despite preserving the
posterior member rows. That label was wrong. The current driver now sets it
from the interrupted posterior status, and uses `not_run` only when no posterior
report exists. Three focused CPU regressions pass, including stale-field
overwrite, preservation of a completed member through supervisor timeout, and
interruption before posterior work. The immutable executed source and original
reports remain unchanged; the audit records the corrected interpretation.

## Decision and remaining work

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close this bounded attempt as incomplete | 8/8 tuning verified; 1/8 posterior/reference delivery | Total allocation exhausted; no recorded posterior veto for the completed member; seven members unassessed | Seven pending outcomes, one target/seed and uncalibrated repeated-look MCSE coverage | Integrate and test full-cohort checkpoint continuation, then resume under a funded bounded allowance | All-member completion, superiority, broad validity or release |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Completed member passed declared screens; no posterior verdict for the seven unfinished members |
| Statistically supported ranking | None, including comparison with the four completed MAMS members |
| Descriptive-only differences | Counts, margins and elapsed time; the fixed-HMC cohort is resource-censored |
| Default-readiness | Not established |
| Next evidence needed | Seven remaining posterior assessments, broader matched models and independent seeds, uncertainty and consumer/release checks |

The [settlement](artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/priced-comparison-01/comparator-01/terminal-audit-01/settlement.json)
charges 79,365.406805 seconds including the outer supervisor. This is
1.171799 seconds beyond the child reservation due to termination/supervision
overhead; it is charged against the precision study's released balance.
The unallocated balance is **1,579.592954 seconds (26.3 minutes)**. The original
3,600-second local-check reserve remains held, with known charges of
2,046.093070 seconds plus unmetered short inspection overhead. No active GPU
reservation remains and no new GPU run was launched. The remaining balance
does not constitute a priced allowance to finish the seven-member cohort.

The next implementation task is to connect the existing checked retained-member
loader and durable posterior-chunk replay to this frozen-design campaign's
layout. Its current launcher has no integrated resume option. Test completed
member carry-forward, interruption mid-chunk, repeated interruption, identical
member-index-derived seeds, unchanged posterior/precision/reference policy,
complete candidate inventory and rejection of changed or corrupt inputs. Resume
from a fresh directory while preserving the original evidence and source; do
not retune the eight verified candidates or apply the optional new MCSE estimator
to this comparison. Reconstruct reports from the saved prefix and recompute only
uncommitted sampling work. Price and fund the remaining posterior work before
launch. These steps preserve the existing scientific question and criteria.

N5's wider state-space/nonlinear models and seed replications, estimator
calibration, and N8 consumer/documentation/release closure remain open. N6 and
N7 retain their separate conditional scopes. A resource stop does not reject
the fixed-HMC method or the research direction. The strongest alternative
explanation for the one successful member remains that local diagnostics and
the finite set of reference quantities miss other posterior features. Complete
reference comparisons and independent replications would test that concern;
the weakest current evidence is the incomplete cohort and lack of replication.
