# HMC timeout supervision repair result

The execution repair is implemented. The reviewed plan is
[here](bayesfilter-hmc-timeout-aware-supervision-plan-2026-09-24.md), and the
[master](bayesfilter-hmc-repair-master-program-2026-09-16.md) and its A–F amendment
carry the same work. CPU checks and the bounded GPU execution-supervision
canary passed their engineering criteria. The GPU numerical fit exhausted its
allocation and remains unavailable. This repair does not close
the historical C2 confirmation requirement.

## What changed and what the audit corrected

The isolated supervisor now reads durable preparation, tuning and posterior
progress, samples workload, and publishes an atomic execution allowance for its
child. Numerical chunks and preparation boundaries consult that allowance.
Only recent progress plus fresh workload evidence can grant a declared
extension, within the cell's absolute remaining deadline. GPU evidence matches
an explicitly selected UUID and excludes the worker's own process group;
utilization by itself never grants time. CPU load is a saturation heuristic.
Extensions default to zero and stall detection defaults to disabled.

The initial design would have put remaining wall time inside the frozen search
configuration. The audit rejected that approach because it changes identity and
may change seeds, and cannot communicate a later extension to an already-running
child. Context-local execution scopes and the parent-owned allowance solve those
problems without changing numerical configurations. The source dependency
closure includes the new execution-budget module.

Further review repaired three related gaps: the cell now inherits its actual
parent deadline including startup; a reporting reserve lies inside that cap;
and a posterior resource stop writes a pause record without sealing partial
draws or a terminal member result. Resume reuses committed chunks and identical
streams. Legacy terminal summaries carrying a resource veto fail explicitly
instead of being silently accepted. A local timeout or cooperative budget stop
continues to later independent fits. Exhausted slots cannot restart. Stalls,
corrupt progress, missing assessments and unexpected failures stop the cell.

## Verification

The final isolated-source check passed **91 tests** in 298.57 seconds
(304.40 seconds enclosing wall time). It includes Gaussian, beta-binomial and
normal-conjugate complete isolated fits; frozen nonlinear transport on Gaussian,
banana and Dirichlet targets; conditional position-field cases including
LGSSM-location; deadline and failure classification; cumulative accounting;
and actual tuning and posterior pause/resume with exact draw/seed comparisons.
These are CPU reference/mechanics checks, with GPU intentionally hidden.

Earlier regression execution passed 206 checks with one failing test whose
synthetic probe did not actually cross the cooperative deadline. Correcting
that test made the intended late-probe scenario pass. A separate 42-check run
passed timeout, public-entry and documentation contracts; the final 91-check
run covers the later supervisor changes. After the GPU canary, a final
41-check run passed the reporting-only overhead change and documentation
contracts. Counts overlap and must not be summed
as independent evidence. The first telemetry mock failure used literal escaped
newlines and was also corrected. No sampler threshold was relaxed to pass.

All 12 changed/new runtime modules compile. The official `docs/main.tex` builds;
PDF pages 433–434 containing the changed tuning text were visually inspected.
The current book reports 15 unresolved citations elsewhere; this amendment
introduces no citations. The Markdown API reference describes the same policy.
No separate tuning guide was created.

Exact commands, source hashes, frozen source, JUnit results, logs and execution
receipts are under
`artifacts/hmc-timeout-supervision-2026-09-24/`. The snapshot uses the recorded
Git baseline plus this repair and excludes unrelated dirty runtime edits.
The final reporting delta is preserved separately from the canary snapshot.
Budget reconciliation charges 541.547 GPU seconds once and all saved CPU checks;
early debugging checks use pytest elapsed excluding interpreter startup. The
recorded balance is 69,588.313 CPU and 64,667.835 GPU worker-seconds. Of the latter,
60,022.255 remain earmarked for C2 and 4,645.580 are unreserved.

## GPU canary and decision

The canary used GPU UUID `GPU-3eb0894d-1bb7-c79f-73a7-ac5b5c1dc79c`, verified
memory growth before device initialization, TensorFlow/TFP GPU/XLA and TF32.
Nineteen workload samples consistently identified a foreign compute process;
the supervisor observed 55 numerical-progress updates. At 442.31 seconds it
granted the declared 90-second extension before the cooperative boundary.
The worker remained inside a numerical call when the final cap arrived and was
terminated with `timed_out / fit_budget_exhausted`. It preserved 33 tuning
observations with a valid checkpoint checksum and no verified member yet.
Trusted process inspection confirmed the worker had exited.

The process receipt records 540.324 seconds against a 540-second final hard
allocation. The 0.324-second scheduling/termination overhead is charged, not
treated as extra authorized numerical work. Review added explicit
`final_hard_allocation_seconds` and `deadline_overrun_seconds` reporting for
subsequent receipts. A process supervisor is not a real-time scheduler; zero
overhead cannot be guaranteed. The enclosing run cost 541.547 GPU worker-seconds,
inside the 585-second enclosing cap and 600-second diagnostic allocation. No
second GPU run was needed for this reporting-only correction.

The validation CLI exited 1 because its single planned fit was unavailable;
it still wrote the final assessment and preserved the denominator. This was an
allowed outcome under the supervision canary's predeclared criteria, not a
successful posterior fit or statistical confirmation. The observed contention
is real; its causal contribution to the earlier C2 slowdown remains unproved.
Automatic grace does not remove the need to price complete fits under the
intended sharing conditions.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Engineering repair passes | 91 isolated-source checks, resume equivalence, actual bounded GPU extension and termination | Valid checkpoint, growth/XLA provenance, no surviving process, total cap respected | Scheduling overhead and workload-dependent complete-fit price | Keep new supervision policy; price before confirmation | Statistical calibration or default timing thresholds |
| Historical C2 stays incomplete | 9/256 completed | Missing fits prevent existing confirmation closure | Full confirmation affordability on shared hardware | Preserve historical failures; require an explicit new-source continuation analysis | Null size or defect power |

| Inference status | Verdict |
| --- | --- |
| Hard veto screen | Final engineering tests pass; historical missing fits remain preserved |
| Statistically supported ranking | None; no stochastic ranking was tested |
| Descriptive-only differences | Runtime, acceptance, verified counts and GPU load |
| Default readiness | No default stall threshold or automatic extension pool is promoted |
| Next evidence | Prices on the intended sharing conditions and explicitly accounted complete confirmation |

The strongest alternative explanation for the historical slowdown is a changing
process, compilation or resource state unrelated to foreign GPU work. The new
instrumentation distinguishes observations but a single shared-device canary
cannot establish causality. A process leak, identity/stream change, unsupported
extension, or corrupted checkpoint would overturn engineering acceptance.
No such failure may be hidden by treating the run as merely slow.

## Next phase disposition

Historical C2 stays at 9/256 complete on its original frozen source. Its three
exhausted slots and all prior cost stay charged. The repaired source cannot be
loaded over those checkpoints under the existing identity contract. Continuing
on the obsolete supervisor would also fail to validate this repair. Therefore
there is no automatic confirmation launch at repair close.

The next discriminating work is complete-fit pricing under the intended GPU
sharing conditions, followed by one explicit continuation design that identifies
the remaining untouched slots, preserves missing fits, reports old/new source
strata and retains the original cumulative budget. If pooling source strata
cannot be justified, report the old confirmation as incomplete and price a
separate complete confirmation before reserving it. Do not reset the original
fits, silently pool development data, shorten the denominator, relax statistical
screens or interpret the canary's partial tuning as posterior evidence. C1's
coverage funding, difficult-map exploration, learned-map quality and the exact
MacroFinance inputs retain their separate open dispositions in the master.
