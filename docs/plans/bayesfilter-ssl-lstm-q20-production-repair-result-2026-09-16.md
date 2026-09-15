# q20 production repair result

## Governing update: executable master and measured cost, 2026-09-16

Engineering status: `REPAIRED_TESTED_COST_STOP_READY`. The complete master is
implemented and tested through 25 real supervised Gaussian-fixture stages and
cached resume. The real q20 GPU/XLA run `real-q20-r2` passed both positive-beta
qualifications, then timed out during pricing. Its terminal result is
`MASTER_INCOMPLETE`; no full training cohort or posterior run was launched.

The measured batch-32 timing records imply 1617749.0855 seconds (449.37 hours)
under the current training reservation formula, before downstream reserves.
That formula reserves all heldout-bank expansions and all training rungs, with
a factor of two. It is a conservative planning requirement, not a statistical
lower bound on actual adaptive runtime. The optimizer floor alone is about
17.57 hours; the raw all-rung heldout reservation is about 207.11 hours.

A bounded accounting repair now stops pricing when measured positive
contributions already exceed the initial campaign allowance. Positive admission
still requires a complete forecast. Fourteen focused checks passed, including
an actual pricing worker, zero-worker resume and the complete pricing route.
The [plan](bayesfilter-ssl-lstm-q20-executable-master-repair-plan-2026-09-16.md)
records the review and explicit follow-up allocation. See the
[whole-program review and execution record](bayesfilter-ssl-lstm-q20-executable-master-review-result-2026-09-16.md).

The next committed-source run is `real-q20-r3`, from the isolated checkout
`/tmp/BayesFilter-q20-master-repair-20260916`, using
`cost-stop-protocol.json` and `cost-stop-allowance.json` under
`docs/plans/artifacts/ssl-lstm-q20-executable-master-2026-09-16/`.
The current CLI is `docs/benchmarks/run_ssl_lstm_q20_production_2026_09_15.py`.
The existing allowance, after r1/r2 and focused tests, is 124650.72741333082
campaign seconds, including 43674.27443840095 diagnostic seconds. Prior holds
remain deducted. The four initial diagnostics remain consumed; the plan
allocates three further capped attempts within these same balances.

GPU execution was admitted with verified memory growth. No current security
rejection occurred. Real qualification establishes graph execution and checked
numerics; it does not establish q20 learning, whitening, posterior accuracy,
method superiority or production readiness. A cost deficit rejects the funded
execution plan, not NeuTra as a research direction.

## Historical recovery audit — superseded by the governing update above

All current/next-step statements below belong to their recorded older snapshot.
They preserve the evidence that motivated this repair and must not select an
older runner or override the governing update.

Date: 2026-09-16  
Status: `PARTIALLY_IMPLEMENTED_MAIN_INTEGRATION_AND_LAUNCH_GAPS`

The training repair is merged into local `main` at
`965ba2949244ee2a1d911515b6865c268a4af8d2`. It implements the substantial
training/checkpoint/export part of the repair. **The complete production
program is not ready.** The former assertion that an idle GPU was the only
remaining condition for a development retry was unsupported and is withdrawn.
Continue under the [master's current section](bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md)
and the unfinished acceptance criteria in the
[repair plan](bayesfilter-ssl-lstm-q20-production-repair-plan-2026-09-15.md).

## Implemented and checked

The [entrypoint](../benchmarks/run_ssl_lstm_q20_production_2026_09_15.py)
implements `validate`, `price` and `train` over one explicit versioned protocol.
The new training route replaces the six-update canary with assessed cumulative
rungs and complete cohort state. A shortened smoke is marked ineligible for
promotion. Beta zero is analytic initialization; positive-beta levels carry
Adam state.

Updates record loss, target status, raw/clipped gradient norms, clipping,
optimizer iteration, beta, seed position and wall time. Checkpoints include
map tensors, Adam state, iteration, root/RNG position, beta and source/data
scope. The tiny Gaussian test restores the cohort through beta 0.5 and 1;
checkpoint tests check exact next-update continuation and reject corrupted or
changed scope. These tests do not establish q20 learning.

The frozen-map exporter preserves weighted dense IAF parameters, permutations,
scale transform and prior affine. Nonzero-map tests check forward, inverse,
log-Jacobian and score parity. Heldout paired loss assessment and map algebra
are distinct from posterior whitening and coverage.

## Unfinished or wrong relative to the declared program

- The CLI has no tuning, member-selection, posterior, ensemble, reference or
  confirmation dispatch. Callable adapters in `q20_production_hmc.py` have not
  passed the repair plan's connected trained-map/public-tuner/posterior test.
- `sample_member` passes `+1000` into
  `SequentialNeuTraHMCConfig.energy_error_log_accept_threshold`; the constructor
  requires a finite negative value. The ensemble converts a finite
  `abs(log_accept)>1000` to a nonfinite state. That is wrong relative to the
  documented explanatory-only finite-energy policy. Fixing the sign alone
  will not repair the ensemble's changed veto semantics.
- `q20_production_comparison.py` is an unavailable-reference reporting stub.
  It performs no independent reference calculation, posterior equivalence or
  start-group comparison. Existing ensemble primitive tests do not establish
  integration of the new adapter or the promised matched comparison ladder.
- `price_training` and `training_quote` cover training only. The CLI does not
  require their output, a complete-cost forecast or downstream reservation
  before training. It has no persistent whole-campaign settlement mechanism.
- `price` ignores `--max-seconds`; the training deadline is cooperative and
  starts after bridge initialization. Native compilation, validation and other
  calls can overrun it. The interrupted CPU smoke did not demonstrate a hard
  launch limit.
- Source identity hashes all Python files under `bayesfilter`, including
  concurrent unrelated untracked work. Serious execution needs a stable
  isolated source snapshot. The main checkout is not such a snapshot.

## Validation evidence and limits

| Check | Recorded result | What it establishes |
| --- | ---: | --- |
| Shared HMC/NeuTra prerequisites | 68 passed, 218.03 s | Component engineering regressions. |
| Final new q20 repair suite | 12 passed, 13.86 s in repair checkout; 12 passed, 14.72 s after main integration | Training, checkpoint, export, assessment and tiny Gaussian cohort recovery. No connected new HMC consumer test. |
| Existing NeuTra/tempered transport suite | 35 passed, 49.71 s | Existing transport numerical regressions. |
| Existing route/ensemble/replica suite | 28 passed, 14.76 s | Existing component/policy regressions, not the new complete ensemble run. |
| CLI help and metadata validation on main | Passed at September 16 master audit | Only three modes; protocol v1, development role, twelve candidates, `promotion_eligible: false`. No TensorFlow execution. |
| Sampling configuration reproduction | Expected `energy_error_log_accept_threshold must be finite and negative` exception reproduced | Concrete single-member constructor defect; GPU deliberately hidden, zero target evaluations and zero sampler transitions. |
| Real q20 CPU smoke | Manually interrupted after initialization/compilation failed to finish in the requested cooperative interval | Incomplete resource/compilation diagnostic; no successful training result or enforced hard timeout. |
| Trusted GPU probe | `no_idle_policy_permitted_gpu`, 2026-09-15 16:48:21 UTC | Historical availability only. No repaired q20 GPU training or HMC run. |

These suites overlap; their counts are not a total of unique tests. Logs,
source inventory and the owner budget amendment are preserved in the
[master audit inventory](artifacts/ssl-lstm-q20-master-recovery-2026-09-16/audit-r1/inventory.json),
which records their original private-checkout paths and ordinary checksums.
The new [constructor check](artifacts/ssl-lstm-q20-master-recovery-2026-09-16/audit-r1/sampling-constructor-check.json)
reproduces the configuration error only; ensemble behavior remains a static
finding pending a connected test.
The evidence is from the recorded source scopes; it does not certify unrelated
concurrent changes in main. The original repair allocation was 7,200 aggregate
worker seconds. Subsequent costs, including the interrupted smoke, still need
full settlement against the September 15 amendment before new numerical work.

## Decision

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Continue engineering repair | Training/checkpoint/export acceptance partly satisfied; end-to-end acceptance unmet | Known sampling argument error, changed ensemble veto, missing reference and external budget limits | Other current public-consumer incompatibilities have not been exercised | Repair and connect the actual consumers; complete supervision; run focused known-target checks | Complete production software readiness |
| Defer serious q20 cohort | No current complete-cost forecast or q20 GPU learning evidence | Downstream completion and reference conditions remain open | Full training/tuning/confirmation/reference cost | Reconcile spending, then bounded trusted q20 GPU pricing and full-stage forecast | Adequate training, whitening, posterior accuracy or affordable campaign completion |

## Inference status

| Item | Status |
| --- | --- |
| Hard veto screen | Concrete software/contract defects block the next sampling path. No repaired q20 candidate has undergone the full scientific screen. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Test runtimes and old diagnostics are engineering observations; no new method comparison was run. |
| Default-readiness | Not established. No q20 training, posterior precision, reference or confirmation qualification. |
| Next evidence needed | Connected consumer tests, bounded q20 GPU/XLA timing and learning, per-scope verified kernels, independent reference, matched posterior comparisons and fresh confirmation. |

The audit invalidates the earlier **software-completion claim**, not the q20
target or NeuTra research direction. The strongest alternative explanation for
prior poor exploration remains inadequate training/tuning or another consumer
defect; the current repair has not discriminated those explanations on q20.
Evidence that would overturn this assessment is a connected, source-bound run
meeting the original engineering and scientific criteria. The weakest current
evidence is the unexercised connection between new adapters and the actual
posterior/reference computation. Passing component tests cannot close it.
