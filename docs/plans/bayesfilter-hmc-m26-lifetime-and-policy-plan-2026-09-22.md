# M26: bounded fit processes and posterior-policy development

Execution is complete. See the
[audited result](bayesfilter-hmc-m26-lifetime-and-policy-result-2026-09-22.md)
and [next phase](bayesfilter-hmc-post-m26-next-phase-2026-09-22.md). The contracts
and original allocations below remain the record of what was executed.

This executes steps 1–3 of the [post-M25 design](bayesfilter-hmc-post-m25-next-phase-2026-09-22.md).
M25 and M21 are complete; their fits will not be rerun as confirmation.
The opening ledger is `m25-r1/reconciliation-terminal-r2.json`, with
84512.57208230694 CPU and 84803.5201060148 GPU worker-seconds remaining.
M26 reserves at most **12000 CPU worker-seconds**, including failed attempts,
tests and worker teardown. It uses at most two numerical workers concurrently.
GPU allowance is untouched. Outputs go to the new `m26-r1/` directory under
`docs/plans/artifacts/hmc-repair-master-2026-09-16/`.

## Research intent and evidence contract

The engineering question is whether recycling a process after each complete
ordinary fit bounds accumulation without changing numerical results, candidate
retention, independent assessment or restart behavior. The comparator is the
current persistent pipeline on identical new development designs, fit IDs,
seeds, data and source. A process must return normally; output files alone do
not establish successful shutdown. Preserve nonzero exits, timeouts and all
failed/missing fits in the declared denominator. Never use `os._exit` to bypass
framework cleanup. Resource measurements include current/peak RSS and live
TensorFlow graph/function and runner counts at fit boundaries. They can locate
accumulation, but a four-fit sequence cannot prove the cause of M21's 128-fit
shutdown exception or establish an asymptotic memory bound.

The statistical question is what posterior allocation deserves fresh
confirmation on Gaussian, beta-binomial and rotated Gaussian. Inspect M21's
saved final diagnostics and fixed arms first. Compare lugsail with the existing
autocorrelation estimator under unchanged numerical-health and posterior R-hat
requirements. Preserve **0.05** absolute MCSE for Gaussian/rotated Gaussian and
**0.005** for beta-binomial; the latter corrects the generic .05 wording in the
post-M25 design. Mean and median intervals are assessed separately in model
coordinates. All must eventually meet the existing pointwise lower-.90
coverage screen; development pilots do not establish that screen.

| Role | Criterion or diagnostic |
| --- | --- |
| Engineering pass | Same candidate/receipt numerical facts and tensor checksums between execution modes; complete archives and restart; ordinary zero process exit; bounded one-fit child lifetime. |
| Posterior promotion criterion | Unchanged per-model precision, posterior health and convergence; later fresh delivery and every declared quantity's coverage lower bound >=.90. |
| Promotion veto | Any posterior health failure, missing requested member, cap, invalid reference or insufficient confirmation. No tuning member is deleted by posterior failure. |
| Continuation veto | Corrupt source/checkpoint, invalid target/reference, exhausted phase/campaign budget, required input absent for the affected cell. |
| Repair trigger | Worker lifetime failure, inaccurate accounting, readiness-window failure, retained precision cap; repair the specific mechanism before larger batches. |
| Explanatory only | RSS, graph counts, runtime, acceptance, pilot coverage and scale/dependence estimates. No stochastic ranking from these. |

## Implementation and ordered execution

1. Extract a single pipeline replication without changing numerical operations.
   Add optional process isolation for numerical search/accuracy/stopping
   experiments, with an explicit per-fit timeout. The parent does not import
   TensorFlow; children establish device policy, execute a complete replication,
   save manifests/diagnostics and shut down normally. Keep the existing default.
   Unsupported engines fail validation rather than silently bypass isolation.
   Unit tests cover timeout/nonzero exit, partial-output retention, resume,
   source/design mismatch, absent output and full-denominator accounting.
2. Freeze the package and harness after focused checks. Use ordinary native broad
   search and the M21 preparation/count settings, new root seed 2026092271,
   `first_verified` member selection and two fits per model for Gaussian and
   beta-binomial. Run each identical design once persistently and once isolated.
   These eight fits are paired engineering replays, not eight independent
   confirmation observations. Limit each to 600 seconds, including shutdown;
   persistent pairs have a 1200-second total cap. Preserve all candidates and
   compare all tensor hashes and numerical receipt/assessment fields while
   excluding elapsed time and root-directory prefixes only. Test restart on
   completed outputs without new numerical work.
3. Inspect all saved M21 scale/dependence/cap evidence (read-only). Derive
   target-specific count hypotheses from final MCSE squared times count, analytic
   model scale and readiness failures. This plug-in estimate is a planning
   approximation, not a precision guarantee. Record the values before running.
   Run at most two fresh complete fits per target (six total), one lugsail and
   one autocorrelation arm with a predeclared identity-based member selection.
   Hold the proposed count policy fixed across these arms and keep independent
   fixed-count comparators. Seeds 2026092272/2026092273 are development-only.
   Allow at most 900 seconds per fit. Nominate a policy for confirmation only
   after these pilots; do not select a posterior winner or infer estimator
   superiority from two unpaired fits.
4. Audit numerical source, complete candidate inventory, process exits,
   warmup exclusion, posterior/fixed outputs, tests and budget. Reconcile time
   before specifying the next confirmation inventory. Update the master and
   progress record with measured results and unresolved gaps. Missing exact
   consumer inputs, full-fit power and supplied partial-map tail work remain
   separate work packages and are not closed by this phase.

## Assumptions, provenance and skeptical pre-execution review

The CPU/graph-only route is an explicit diagnostic/reference exception, matching
the M21 lifetime problem; it cannot certify GPU/XLA performance. Use
`/home/ubuntu/anaconda3/envs/tfgpu/bin/python`, `CUDA_VISIBLE_DEVICES=-1`,
`TF_FORCE_GPU_ALLOW_GROWTH=true`, `BAYESFILTER_PRELOAD_CUSTOM_OP=0` and one
intra/inter-op, OpenMP and OpenBLAS thread per worker before framework import.
Freeze source hashes and preserve the Git baseline plus dirty source snapshot.
Commands, environment, seeds, data, wall times and output paths are recorded
by the launcher. No external services or learned training are needed.

The 600-second per-fit cap is a conservative engineering ceiling over measured
M25 228–238-second profiled fits; 900 seconds permits longer posterior pilots.
These are resource limits, not success thresholds. Eight parity fits consume
at most 4800 seconds; six pilots at most 5400; focused tests, saved-data
diagnosis and final audit share the remaining 1800. Unused capacity transfers
within M26 without increasing its total. The seed integers are convenience
identifiers with no statistical significance. Native L grid, 128-draw evidence,
preparation settings and posterior thresholds are inherited baseline hypotheses.
`first_verified` removes the fixed-L missing-selection mechanism; it does not
assert that the selected member mixes better. The count/estimator defaults
remain unchanged in the public library.

Skeptical review passed with one correction: beta-binomial's original tolerance
is .005, not .05. A short memory test may fail to reproduce long-run accumulation,
so the repair claim is limited to process-lifetime containment and checked
parity. Merely writing a result before a hung exit is not a pass. The parent
must retain errors and count missing fits. Source identity must be identical
between paired modes, and imported source must come from the frozen snapshot.
Native search completion must be checked before interpreting parity; timeout
can change the number of candidates. Policy pilots cannot be confirmation,
AR(1) allocations cannot be transplanted, and changing selection defines a new
experiment rather than repairing M21's denominator retrospectively. The most
likely misleading success is excellent pilot coverage on too few fits; the
phase explicitly prohibits that conclusion.

## Saved-data diagnosis and frozen pilot hypotheses

`m26-r1/saved-allocation-diagnosis-r1/result.json` reads all 264 Gaussian,
beta-binomial and rotated-Gaussian fits without changing their archives. The
plug-in largest required retained counts are 13047, 4110 and 29715 per chain,
respectively. Seven Gaussian fits reach the warmup cap; neither other model
does. The largest final Gaussian recent-window R-hat is 1.90410 on 1000 draws.
Treating excess R-hat squared as inversely proportional to window size gives
`1000*(1.90410^2-1)/(1.05^2-1) = 25615` draws. This is an explicitly heuristic
planning calculation, not a burn-in theorem or proof of stationarity. It can
fail if the chain remains unequilibrated or has another slow timescale.

| Target | Warmup minimum / window / cap; chunk | Retained minimum / cap; chunk | Fixed discarded / retained |
| --- | --- | --- | --- |
| Gaussian | 30000 / 30000 / 60000; 5000 | 4000 / 20000; 2000 | 30000 / 20000 |
| Beta-binomial | 2000 / 1000 / 10000; 500 | 5000 / 10000; 1000 | 2000 / 10000 |
| Rotated Gaussian | 2000 / 1000 / 10000; 500 | 30000 / 60000; 5000 | 2000 / 60000 |

The Gaussian window rounds the heuristic 25615 upward; its retained cap rounds
13047 upward to 20000. The 4000 retained minimum is a development hypothesis
(four times the original 1000) to reduce very early stopping, not a reviewed
default. Beta and rotated minima round their observed retained requirements
upward. Warmup for those two models is unchanged because saved failures do not
implicate it. Doubled caps for the larger new allocations are convenience
margins, bounded by the explicit phase budget, with no claimed coverage
guarantee. All six pilots retain the target-specific MCSE requests above and
use `first_verified`; old L=3 estimates only motivate the hypotheses because
new member selection may change dependence substantially. Both estimator arms
share these counts. No pilot outcome will retrospectively change this table.

## Parity repair triggered during execution

The first paired Gaussian fit had identical mass, starts, search configuration
and root seed, but different `target_preparation_identity` and work seeds.
`initialize_bootstrap_step` stores `wall_seconds` in each probe round and first
failed proposal. `HMCGeometryInitializationResult.artifact_hash` binds that full
report; the candidate preparation identity in turn binds that hash. Consequently
clock variation changes fresh tuning streams. This is a reproducibility defect,
not evidence that process isolation changes the transition law. Preserve the
initial paired runs as diagnosis/cost evidence; they cannot pass parity.

Repair the numerical preparation identity separately from the full audit hash:
retain the original geometry artifact hash and timings for integrity, but derive
a numerical geometry hash excluding only the known bootstrap-probe clock fields.
Use that numerical hash for new candidate scopes. Old checkpoints retain their
saved scopes and are not upgraded. Test clock invariance, sensitivity to changed
seeds/epsilon/proposal outcomes, and unchanged geometry without probe timings.
Then freeze a new source and repeat paired fits. The source change necessarily
defines fresh candidate streams; it does not invalidate previously assessed
independent streams. This bounded implementation repair fits within M26's
unused 12000-second allocation. Count every initial run and failed test.

Skeptical repair review: do not ignore every hash mismatch in the parity audit
or delete timing evidence. Full artifact integrity and numerical seed identity
have different purposes. Exclude the two explicitly identified timing locations
only; new numerical fields must continue to affect the identity. No statistical
criterion, target, estimator or total budget changes.

## Terminal pilot repair trigger and confirmation pricing

Five pilots passed their posterior checks. The rotated-Gaussian autocorrelation
pilot selected a verified L=25 member and exhausted the 10000-transition warmup
cap without a numerical-health veto. Its 1000-transition windows have low bulk
ESS. This is a member/allocation repair trigger, not evidence against the
estimator or permission to drop a tuning candidate. Before phase close, inspect
the saved warmup and independent 60000-transition fixed arm at larger diagnostic
windows. Compare both existing mean-MCSE estimators on the same saved draws;
their differences are explanatory only. Preserve the failed actual stop.
Record hashes of the inspected tensors. This read-only diagnosis is capped at
120 CPU seconds within the unchanged M26 allocation.

Also calculate the exact binomial operating characteristics and measured cost
of possible confirmation inventories. For the inherited pointwise two-sided
95% lower-bound screen at .90, find the smallest passing success count, then
the probability of reaching it at explicitly hypothetical true probabilities
.90, .95 and .975. These calculations price evidence, not estimate actual
coverage from six pilots. Use the maximum observed per-model fit cost and
report its uncertainty; two observations cannot establish a runtime bound.
No underpowered inventory is silently promoted to a closure experiment.

Skeptical review: larger-window diagnostics on the same failed fit cannot
replace its recorded readiness outcome, and a covered fixed interval need not
meet the precision request. Estimator arms have different seeds/members, so
they cannot support a causal comparison. Confirmation needs fresh complete
fits and all requested quantities in the denominator. No extra MCMC is
authorized by this read-only diagnosis, and no public default changes.
