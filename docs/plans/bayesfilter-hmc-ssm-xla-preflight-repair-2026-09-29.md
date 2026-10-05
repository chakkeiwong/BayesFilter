# Repair the state-space full-chain XLA preflight

## Checked state and question

All ten C1 timeout recoveries are complete with their original source/data/seed
identities. The original confirmation remains unchanged; the post-hoc beta
summary has 256/256 complete fits. The nine new fits cost 2,468.172 seconds.
All eight K0--K7 GPU/XLA mechanics cells then passed independent value and
score checks in 62.828 enclosing seconds, with verified GPU placement and
memory growth. The four ordinary/prepared K0/K7 bridge cells failed immediately
with `target lacks full-chain XLA qualification`, costing 36.907 seconds.

`CampaignSSMTarget.value_score_capability` explicitly returns
`full_chain_xla_diagnostic_ready=False`. CPU public-pipeline regressions use
`use_xla=False` and therefore could not expose this mismatch. The campaign
declares GPU/XLA throughout, so its own adapter prevents its full-chain
preflight from running. This is an interface/qualification failure; it is
not evidence that an XLA HMC transition cannot compile.

Question: can these exact campaign targets execute the existing batched
value/score HMC transition inside one stable XLA chain graph, and then pass
the existing public tuning/reload/posterior-lifecycle preflight? The comparator
is an identical fixed-seed graph without XLA, using the same target, supplied
manual score, starts, HMC step size and leapfrog count. No alternative density,
autodiff score, pfor path or sampler is introduced.

## Repair, evidence contract and skeptical audit

First add a bounded diagnostic of the full TFP chain around the existing
`reviewed_independent_chain_target_fn`, which uses the supplied batch-native
score through a custom-gradient boundary. This helper does not issue tuning
artifacts or claim a target is qualified. Compare graph/XLA numerical results
and finite state/value/acceptance/status for all eight exact profile families.
Use four chains, four transitions, L=2 and epsilon=0.001. These are convenience
mechanics settings chosen to keep trajectories near finite prior starts, not
tuned kernels, posterior samples or convergence evidence. The predeclared
comparison tolerance is 2e-5 absolute/relative, inherited from the existing
campaign graph/XLA value-score parity test. The first diagnostic exposed a
test-design error: equal TensorFlow random seeds do not establish equal random
variates across graph and XLA execution (all 16 K0 positions differed, while
both complete chain graphs executed). Therefore compare the existing TFP
leapfrog operation with identical supplied momenta; full stochastic chains
are checked for compilation, finite outputs and target status separately.
Do not compare different random trajectories as a numerical parity test or
loosen the tolerance. This revises the comparator, not the target or sampler.

After that check passes, enable the existing full-chain *diagnostic* capability
for the campaign's XLA-compiled target configuration, scoped to
`inference_validation`. An explicit non-XLA target retains false readiness.
The evidence path points here and the tests. Public admission guards remain
unchanged. The actual GPU public preflight remains mandatory before pricing
or main admission; a static flag or CPU test is not GPU qualification.
Add a regression that calls the ordinary public preflight with XLA requested
and tests full-chain binding/transition behavior with GPUs intentionally hidden.
The public binding regression uses the existing acceptance policy's minimum
64 measurements per chain, epsilon=0.001 and L=2, without repairs. Its pass
criterion is execution and finite evidence, irrespective of candidate admission.
The raw all-profile full-chain tests passed (8 tests, 61.40 seconds).

Fresh GPU work re-executes the eight mechanics and four pipeline cells on the
repaired source, then the unchanged eight complete-fit prices and main cases.
Previous results are preserved; no failed record is rewritten as passing.
The C1 recovery is complete and must not run again. The queue gains a checked
completed-recovery input and retains the original SSM start, deadline, budget
and data. Audit cumulative SSM accounting across preparation roots: stage
receipts are outside the queue's runtime directory and must be counted too.

| Diagnostic | Role |
| --- | --- |
| Graph/XLA density, score, transition discrepancy; invalid status; nonfinite output | Repair trigger and continuation veto for affected route |
| Missing full-chain capability on an eligible campaign target | Interface regression failure |
| GPU mechanics and public lifecycle evidence with growth/XLA/source identity | Required engineering pass criteria |
| Candidate rejection, posterior readiness/precision/reference failure | Promotion veto only; preserve and continue independent declared work |
| Full-fit runtime | Pricing evidence only after complete declared member workload |
| Budget, original wall deadline, source corruption | Continuation veto; never reset or enlarge implicitly |

The skeptical audit rejects simply deleting the tuner guard or setting a flag
and declaring GPU readiness. It also rejects repeating C1, changing scientific
settings to make preflight pass, or reporting the earlier CPU graph tests as
full-chain GPU evidence. The targeted diagnostic precedes capability repair;
fresh GPU preflight precedes main admission. The plan passes with those bounds.

## Budget, commands and preserved result

The active additive ledger has 208,359.954 seconds (57.878 hours), including
the 1,200-second NeuTra reserve, and no active reservation. No new grant is
needed. GPU diagnostic/retry work consumes the existing SSM 36-hour cap and
six-hour repair allocation. CPU diagnostics, tests and reporting have a
1,800-worker-second ceiling from the existing balance (43,038.430 seconds).
This leaves the original 16-hour CPU campaign ceiling intact.

Use tfgpu Python, TF/TFP, float64, CPU-hiding for reference checks and trusted
GPU execution with memory growth. Existing per-stage GPU ceilings and the
22-hour main cap remain. New work stops September 29 at 23:03:52 Shanghai;
computation stops September 30 at 03:03:52; terminal reporting is due at
05:03:52. The 13-hour unattended interval after the failed preflight is elapsed
wall time, not a reason to reset this clock.

Artifacts and exact commands go under
`docs/plans/artifacts/hmc-ssm-xla-repair-2026-09-29/`; the active ledger remains
`artifacts/hmc-ssm-funded-2026-09-28/grant-ledger.json`. A fresh frozen package
and output tree preserve old evidence. Focused pytest receipts, full-chain
diagnostic output, launch manifest and stage receipts preserve the result.
Numerical failures may require an actual kernel repair and new parity evidence;
passing only the capability guard will not close that question.

This is an engineering repair. It cannot establish posterior calibration,
model superiority, sufficient warmup in every case, or a new numerical default.
The strongest alternative explanation for later failures is target geometry
or approximate filtering, rather than metadata. Complete-fit prices and
independent posterior/reference checks must distinguish those outcomes.

## Prelaunch repair review

The terminal focused regression set passed 69 tests in 73.80 seconds. It covers
all eight full-chain graphs and deterministic proposals, scoped capability
rejection for non-XLA targets, K0/K7 public prepared measurements, cumulative
stage accounting, completed-recovery evidence reuse and corruption detection,
queue interruptions, original deadlines, and preflight failure sequencing.
The earlier 8-test run overlaps this set; the failed RNG-comparator attempt
is retained as a test-design failure, not a numerical failure.

Trusted read-only reconciliation checked all ten actual recovery results,
unchanged original numerical source, fit identities, assessment checksums and
GPU/growth manifests. GPU ledger charges are unique and settled. Prior SSM
spending is 1,100.810 seconds, including 521.072 seconds of separately charged
capacity waiting. The continuation subtracts both earlier SSM roots from the
same ceiling, and uses the same ledger and clock. The package changes are
limited to scoped target capability metadata and snapshot dependency inclusion;
the external queue adds checked recovery reuse and corrected accounting.

CPU reference command (GPUs intentionally hidden):

```sh
CUDA_VISIBLE_DEVICES=-1 BAYESFILTER_TEST_DEVICE_SCOPE=cpu TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=2 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q tests/inference_validation/test_ssm_xla_full_chain.py tests/inference_validation/test_funded_ssm_queue.py tests/inference_validation/test_c1_recovery_sequence.py tests/inference_validation/test_ssm_campaign_plan.py --disable-warnings --maxfail=1 --junitxml=docs/plans/artifacts/hmc-ssm-xla-repair-2026-09-29/repair-regressions-r1.xml
```

The review passes for a fresh GPU preflight. CPU compilation does not establish
actual GPU public-pipeline qualification. The detached continuation must stop
on a failed preflight, preserve its reason, and never report main readiness
until that preflight and complete-fit pricing pass.
