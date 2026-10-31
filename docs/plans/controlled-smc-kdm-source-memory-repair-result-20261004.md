# Source-memory repair and executable continuation: review and execution

The source reference was interrupted by an Octave process-memory failure, and
the master lacked an executable C2 continuation. Both infrastructure problems
are repaired. The full retry then exposed an original-source Lorenz covariance
failure; independent neuroscience drivers continue. Full controlled-SMC and KDM
replication remains incomplete.

## Checked failure and repair

Attempt143 completed 1989/4000 Lorenz cSMC cases before its child returned -9.
The preserved kernel evidence records an Octave OOM kill at 40638348 KiB
anonymous RSS. The old manifest did not record the child PID; attribution uses
matching type, timing and signal. Accumulation across nested source calls is the
leading explanation; the exact Octave garbage-collection mechanism is not
proved. This does not show a mathematical failure of the paper.

Each original source case now runs in a fresh CPU Octave process. Both RNG
states and accumulated matrices carry forward; the numerical author functions,
source data, grids, repetition counts, particles, refinements and full seeds
are unchanged. Each child has an 8 GiB address-space limit. PIDs, exit codes and
cumulative maximum child RSS are recorded. Prior results remain preserved.

The C2 analyzer now checks complete native matrices and CSV rows, hashes, settings,
APF pairing and all 12000 cases before the already declared bootstrap analysis.
The continuation waits for the real predecessor deadline, handles failure and
duplicates, and reserves its CPU budget. Runtime and timing remain descriptive:
fresh process startup changes execution overhead and cannot establish the
author's original performance.

## Verification and execution

- Attempt144: PASS, 24 source evaluations across all four actual drivers.
  Likelihoods, ESS/refinements where available, carried matrices and both final
  RNG states match exactly between uninterrupted and isolated execution.
  CPU charge 364.4518109320197 seconds; cumulative child peak 2018228 KiB.
- Software: 24 focused tests PASS, 2.922 wall seconds (checks-04). Includes actual dispatcher to
  analysis subprocess, native MAT disagreement, bootstrap behavior, queue
  serialization/failure/duplicate handling, and retry/C2 budget preservation.
- Attempt145: RUNNING; source supervisor 4131909; cap252891.45682043504 CPU
  seconds (about70.25h). Original source attempts together remain capped at72h.
  1989 completed Lorenz prefix cases independently checked exactly against
  attempt143 at this review.
- C2 queue: BLOCKED after the Lorenz failure; obsolete supervisor4132309 was
  stopped after PID/identity/command verification. No numerical job was
  interrupted. New queue starts reject known source failures immediately.
- Frozen failure replay: WAITING, supervisor395864, queued at
  2026-10-04T08:01:31Z. At most 60 CPU seconds after the active source job,
  from the existing allocation. No GPU allocation is consumed.
- Neuroscience cSMC: all 2000 cases completed. Its BPF comparator is running;
  the live checkpoint records current counts and remaining budget.

The primary manifests preserve the exact commands, snapshots, dependencies,
environment, seeds and accounting. The full-source run preserves the original numerical functions; only copied
driver instrumentation and the process schedule changed. The separate replay
captures matrices only after the original covariance exception and rethrows it.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Restart original source | Exact four-consumer output/RNG check PASS | No guard, parity or input veto in the check | Later grid cases, complete runtime and exact retention mechanism | Continue attempt145; stop on corruption, prefix mismatch or execution failure | Full matrix or paper replication |
| Block full C2 on the source failure | 24 software checks PASS; frozen statistics implemented | Lorenz inputs incomplete; actual C2 blocked | Full statistical results unavailable | Preserve the block and diagnose the source case | TensorFlow parity or KDM training success |
| Replay the frozen source failure | Reviewed code and exact input hashes; numerical run queued | Replay mismatch or missing capture invalidates diagnosis | Fit, inversion and runtime contributions unresolved | Execute after independent neuroscience work | Numerical repair acceptance or paper refutation |
| Preserve remaining replication work | Source and localization findings retained | P0/P1 and KDM evidence missing | Source-valid downstream tolerances; missing training details | Finish scoped specifications using existing findings | E0 robustness work is required for replication |

| Inference status | Assessment |
|---|---|
| Hard veto screen | Attempt143 incomplete after OOM; isolation check passes; attempt145 Lorenz source rejected covariance at grid36/repetition38 |
| Statistically supported ranking | None from this infrastructure check |
| Descriptive-only differences | Timing and memory measurements |
| Default readiness | No new default or port promotion |
| Next evidence needed | Frozen source failure replay and runtime/fit diagnosis; completed independent neuroscience baseline; source-valid port and independent KDM training remain pending |

## Red-team note

The strongest remaining alternative is that process isolation changes a hidden
cross-case state outside the saved outputs/RNG. All four consumer checks and
the live interrupted-prefix comparison address it; any later mismatch rejects
the repair. The bounded check is weakest in its limited grid coverage. MATLAB
equivalence, full TensorFlow parity, Table1 exact datasets, the missing multimodal
driver/horizon, and KDM initialization/training schedule remain separate gaps.

Evidence: attempt144_c1_source_isolation_check/summary.json,
attempt145_c1_source_full/manifest.json, queue-c2_source_results.json, and
source-continuation-review-20261004-01/{checks-04.json,prelaunch-review.json,launch-verification.json,c2-queue-disposition.json}.

## Full source failure found after launch

Lorenz cSMC stops after3537/4000 completed cases, at grid36/repetition38 (forcing7.8846153846). Octave mvnrnd reports covariance minimum eigenvalue -0.00458167. The process exits1 at124064 KiB peak child RSS, so this is not the previous OOM. All1989 common cases matched the interrupted run exactly. Frozen inputs/RNG and logs are in source-continuation-review-20261004-01/lorenz-source-failure/.

Checked call chain: cSMC_VaryF selects full quadratic fitting (line79). ApproxDP solves the normal equations (cSMC_Resample line193); the d>1 full-quadratic branch (line211 onward) has no positive-definite projection, and forms G=inv(Transition.Ginv+2*UpdatedA) at line224. RunSMC passes this covariance to mvnrnd (line99), whose runtime rejects the observed indefinite matrix. A Gaussian precision Q must be positive definite and its covariance is Q inverse; unconstrained quadratic fitting does not itself impose that condition. The observed call violates the covariance domain. Ill-conditioning is observed, but the relative contribution of the fitted policy, roundoff and Octave solver semantics to this exact failure is not yet measured. No TensorFlow or XLA code is on this source call chain.

This is evidence of failure in this author-source/Octave run, not proof of a false paper or equivalence to the authors MATLAB runtime. The full Lorenz result and dependent APF comparison are incomplete. Leave numerical functions unchanged; finish unaffected neuroscience runs, then reproduce the failing case from its saved RNG and inspect the fitted precision/runtime compatibility before choosing any replication-preserving repair. Full C2 must reject missing cases; it cannot silently remove the failing cell or promote a partial matrix.

## Final review and executable next step

The skeptical review checked the original-source comparator, complete case
accounting, solver/domain uncertainties, source-to-consumer call chain, queue
and budget limits, CPU reference labeling, and separation of software success
from replication success. The final guard fixes a review finding: a known
driver failure must block C2 immediately even while unrelated work continues.
The corresponding regression and 23 other focused checks pass. The replay
source was syntax-checked and its precision/covariance indices traced through
the actual failing RunSMC call; numerical replay is pending its queue.

The active next phase is
[the frozen failure replay](controlled-smc-lorenz-source-failure-replay-20261004.md).
Its supervisor is live and waits for attempt145. No observed source failure is
discarded, no new seed is substituted, and no numerical protection is selected.
The memory repair has exact bounded verification; complete replication remains
unresolved. The replay may localize the Gaussian-domain violation, but a claim
about the upstream fit or MATLAB equivalence needs the corresponding evidence.

## Owner clarification: reproduce failures; do not repair the author method

The owner clarified that the first deliverable is a version no worse than the
author implementation. The earlier replay-plan statement that the source
covariance failure itself was a replication veto was wrong for this target and
has been removed. Intrinsic reference failures are accepted baseline outcomes.
Additional failures or material numerical regressions introduced by our port
block its acceptance. Report every evaluated outcome and leave unreached cases
unevaluated; preserve the distinction between the Octave reference and untested
MATLAB behavior.

C2 remains a separate full-paper-statistics analysis. Its missing Lorenz matrix
blocks that calculation, not matched translation on the audited source cases.
The queued replay captures a fixed failure case for the TensorFlow comparison.
Further diagnosis is limited to translation discrepancies or an invalid
reference setup; author-method numerical repair remains outside current scope.

All 24 software checks pass after this clarification (checks-05). The waiting
replay supervisor was replaced so it uses the new checkpoint/reporting policy.
The independent neuroscience numerical run was not interrupted. Exact process
and scope records: source-continuation-review-20261004-01/owner-scope-clarification.json.
