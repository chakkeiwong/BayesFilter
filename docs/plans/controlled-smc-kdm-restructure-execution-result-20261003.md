# Replication restructuring: review, repairs and execution

Date: 3 October 2026. Master:
[controlled-SMC/KDM replication](controlled-smc-kdm-replication-master-program-20260928.md).
Checkout: /home/chakwong/BayesFilter, branch surrogate-hmc, base commit
89065bc6354801cb368d5e163ba49fd9c4372d10. Unrelated working-tree changes are preserved.
Review: Codex self-review of the source anchors, master, dispatcher, phase
specifications and executed artifacts; no independent reviewer was used.

The master now follows the agreed order: reproduce the original author
baseline, establish a TensorFlow implementation with no worse numerical
behavior on source-valid cases, then consider Task 3/E0 enhancements.
The ready preparation and cost-pilot phases were executed. **Neither full
replication is complete.** A full source command is prepared, but its CPU
allocation is not available. The entire roadmap is not yet executable without
further phase specifications.

## Skeptical review and repairs

| Finding | Repair and verification |
|---|---|
| A restructured document alone did not stop the old dispatcher from launching historical numerical-repair phases. | The active phase list is explicit; run and wait-run reject historical phases. Old reports remain readable. Preflight exercised rejection before attempt creation. |
| Source execution, statistical reproduction, port parity and robustness could be conflated by a generic PASS. | Separate C1/C2/P0/P1/K/J verdicts; all current replication verdicts remain INCOMPLETE. Metadata and cost-pilot PASS cannot promote them. |
| Source settings and experiment coverage were incomplete. | C0 verifies 52 selected source/data files against pinned Git blobs, preserves relative paths and records dependency/paper hashes. Table 1 is included in the source matrix; missing multimodal driver and the horizon discrepancy remain explicit. |
| Copying files by basename or inferring the repository root from a copied runner could select the wrong source. | Snapshots retain repository-relative source paths; reference runners receive an explicit root. |
| Cost-matched APF could be treated as a fixed-particle comparator. | Preserve the author's particle-count lookup from each matching cSMC refinement count. Verify all grid/repetition keys and matching counts. |
| The original paper's comparators could accumulate additional replication prerequisites. | Keep the paper's comparators and correctness references. Additional heuristic competition belongs to practical-quality claims. A faithfully reproduced negative result is still a replication result. |
| Matching author double precision or repairing our translation could be mislabeled an enhancement. | Matching author FP64 and genuinely equivalent algebra remain replication work, with endpoint checks. New objectives, constraints, projections, floors or stopping semantics are E0. |
| KDM experiment configs imported generated YAML files that were missing. | Verify 17 author files and run the author's generator in a fresh source snapshot. Validate all five generated sets and trial indices 0 through 10. This does not recover missing Appendix B settings or perform training. |
| An inherited nonlinear toy dataset could be mistaken for Appendix B. | Record that the Cappe model differs from Appendix B's linear transition/bimodal observation equations. It cannot close that replication. |
| Saved native matrices were loaded but not compared with the CSV case records. | Add exact shape, value and finite-real classification checks. All 12 completed pilot cases pass; 12 deliberately corrupted records/shapes are rejected. |
| A full source launch needed far more CPU time than remained. | Measure costs before launching; prepare a 72-hour bounded specification and reject launches without that allocation. No GPU-to-CPU conversion or new grant is recorded. |

The audit checked baselines, proxy metrics, source/runtime differences, stop
conditions, comparison fairness, hidden defaults and whether commands answer
the stated question. It found enough material defects to require the repairs
above before execution. After repair, the bounded source preparation and cost
pilot were suitable for their stated questions. The full command remains
conditional on its resource allocation, and original MATLAB/Octave numerical
equivalence remains unverified.

This review inspected the local published controlled-SMC paper's relevant
method and experiment sections, the manuscript/supplement and pinned author
drivers; for KDM it inspected Appendix B's equations and training description
alongside the local author configs, generator and dataset implementation.
Exact source anchors and remaining unknowns are in the C0/K0 phase plans and
the linked source audits. No new literature-based claim of algorithmic failure
or superiority is made.

## Executed work

All attempts are preserved under
[the campaign directory](artifacts/controlled-smc-kdm-replication-20260928/).
Each contains its command, source snapshot, manifest, logs and structured
result. CPU references deliberately hid GPU devices.

| Attempt | Command phase | Outcome | Charged CPU wall seconds |
|---|---|---|---:|
| 139 | c0_source_lock | Metadata PASS: 52 source/data files verified | 0.165314 |
| 140 | c1_source_cost_pilot | Three drivers completed; BPF hit the 45-second driver limit | 76.915210 |
| 141 | c1_source_cost_pilot | BPF-only retry completed; nine prior completed cases reused with provenance | 114.912470 |
| 142 | k0_source_config | Metadata PASS: five config sets, 11 trials each | 0.264755 |

The actual dispatcher invocation for each phase was:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase PHASE

Attempts 140 and 141 together consumed 191.827681 CPU wall seconds, within
the 210-second pilot budget. Across the four attempts, the charge was
192.257750 CPU wall seconds and zero GPU seconds. No numerical job is running.

The combined pilot accounts for three selected grid cells for each of Lorenz
cSMC, Lorenz APF, neuroscience cSMC and neuroscience BPF: 12 cases in total.
Their returned log normalizers are finite and real. The timeout is a resource
failure, not evidence of a numerical failure or a defective paper.
A finite final log normalizer does not certify every intermediate covariance,
density or domain condition; these cases are not thereby certified as
source-valid port comparison cases.

The author KDM generator produced task counts 55, 55, 55, 44 and 22 across its
five config sets, with 11 distinct trial indices each. Original enable/disable
flags are preserved. No dataset generation, checkpoint training or
TensorFlow/PyTorch learning comparison was performed by that phase.

## Prepared full source continuation

[The full-source plan](controlled-smc-kdm-c1-full-source-20261003.md) specifies
the original Figure 6 cSMC/APF and Figure 4 right cSMC/BPF matrices:
12,000 cases with 100 repetitions per grid cell. Prepared source-driver copies,
diffs and commands are in
[full-source-preparation-20261003-02](artifacts/controlled-smc-kdm-replication-20260928/full-source-preparation-20261003-02/).
They are preparation artifacts, not numerical results.

The sparse pilot projects approximately 30.50 CPU wall hours by mean case time
and 67.90 hours under the conservative scheduling rule. Neither is a statistical
confidence bound. Octave timers and Python process wall time disagree in some
cases; the full runner additionally records process CPU time. Runtime variation
over unsampled cells remains a limitation.

The proposed cap is **72 additional CPU wall hours**, with per-driver limits
summing to 69 hours. Existing CPU balance is 93.880125 seconds. The inherited
GPU balance is 115,949.876987 seconds (32.21 hours), of which 2,998.576636 seconds
remain in the active tranche. These are different resources. No additional
grant has been recorded. Attempt allowance 142 is exhausted by the four
authorized restructuring attempts.

A full-run preflight rejected the requested reservation before creating an
attempt. On owner authorization, record the explicit additional CPU grant and
raise the attempt allowance to 144 as specified in the full-source plan.
Then run:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c1_source_full

This allocation covers only those four matrices. It does not fund Table 1,
multimodal experiments or KDM training. The plan freezes source statistics and
bootstrap uncertainty before the full run; it does not equate a successful
process with reproduction of the paper's comparative findings.

## Checks and interpretation

[Restructuring preflight](artifacts/controlled-smc-kdm-replication-20260928/restructure-preflight-20261003.json)
records syntax, driver-instrumentation round trips, inactive-phase rejection,
source dependencies and the unavailable-budget guard.
[Source result integrity check](artifacts/controlled-smc-kdm-replication-20260928/source-result-integrity-check-20261003.json)
records four native-matrix checks covering 12 cases and 12 rejected negative
controls. It reads existing output; it does not rerun or alter source results.
[Final metadata and launch checks](artifacts/controlled-smc-kdm-replication-20260928/restructure-final-checks-20261003.json)
pass all seven check groups: active links, phase files/dependencies, syntax,
prepared scope, historical launch rejection, the isolated resource guard and
the checkpoint transition after full-source completion. Status and checkpoint
are refreshed. The isolated fixture creates no attempt or subprocess and
grants no resource. An initial metadata-import failure was repaired by adding
the scripts directory to the diagnostic import path; no experiment was affected.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Accept restructuring repairs | Master and active dispatcher follow source → port → enhancements | No known remaining scope/launch defect in the implemented tranche | Later phases still need executable specifications | Preserve this order in C2/P/K work | Entire master or either replication complete |
| Accept C0 metadata and K0 config generation | Pinned-file verification and generated-config validation pass | No metadata mismatch observed | K0 scientific specification remains partial | Recover missing source settings/dependencies | Learning replicated |
| Accept cost-pilot completion | All 12 scheduled cases accounted for | Initial timeout repaired within budget; finite-real output screen passes | Sparse cost coverage; clock discrepancy; full numerical validity unchecked | Allocate and run the prepared full source matrices | Statistical ranking, robustness or TensorFlow parity |
| Defer full numerical launch | Required resource reservation unavailable | Budget is a continuation boundary | Actual full-run cost | Obtain the specified CPU allocation | Numerical method failed |

| Inference status | Current evidence |
|---|---|
| Hard veto screen | No nonfinite returned pilot logZ; one timeout, successfully retried. Full source-validity and port validity are not established. |
| Statistically supported ranking | None. Pilot cells are not a repeated-run comparison. |
| Descriptive-only differences | Timings, cost projections and finite returned values. |
| Default readiness | Not evaluated by this source-reference tranche. |
| Next evidence needed | Full repeated source outputs; source-valid fixed-input checks; predeclared downstream port non-inferiority; independently trained KDM comparisons. |

## Remaining work and strongest alternative explanations

1. Execute the prepared Figure 6/Figure 4 source matrices and assemble C2.
   Original MATLAB/Octave equivalence and published-draw availability remain
   separate limitations even if the Octave runs finish.
2. Specify and cost the Table 1 generative-design rerun. Exact original unseeded
   datasets are unavailable.
3. Resolve the missing multimodal Repeat.m, the 100/101-observation discrepancy
   and the distinction between original source behavior and a corrected
   mathematical route. Reconstructing a driver is not unchanged author code.
4. Finish P0's source-preserving Lorenz fit/consumer parity investigation.
   Freeze justified metric-specific tolerances and non-inferiority margins
   before untouched P1 comparisons. Coefficient drift alone is not a verdict;
   downstream discrepancies cannot be dismissed as merely conditioning.
5. Recover KDM Appendix B's missing numerical setup and specify the bearings
   pretraining, MDPF/A-MDPF, NLL/MSE and comparator dependencies. Cost actual
   independent training before allocating the remaining GPU budget. A prior
   single-trial cost does not establish that the complete training scope fits.
6. Implement and verify the TensorFlow learning call chain and downstream KDM
   parity. Reference IWSG/mixture-Kalman checks and checkpoint evaluation do not
   establish learning replication.
7. Keep original iAPF incomplete/deferred and E0 downstream. Neither is closed
   by relabeling the current controlled-SMC/KDM evidence.

The strongest alternative explanation for pilot success is simply that selected
cells avoided difficult source behavior. The weakest scheduling evidence is
the small number of timed cells plus the clock discrepancy. Full source failures
would change the source outcome, not authorize silent damping or projection.
A mismatch caused by Octave rather than our TensorFlow translation would require
runtime localization before attributing it to the port. No result here shows a
fundamental defect in either paper.

The next blocked numerical action needs an expanded CPU allocation under the
repository's Execution Tiers and bounded-compute policy. That is the reason
for the approval request; no review-token or procedural approval chain is being
introduced.
