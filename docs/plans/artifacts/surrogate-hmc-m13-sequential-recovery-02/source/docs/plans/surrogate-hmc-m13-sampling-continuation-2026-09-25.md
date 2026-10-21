# M13 controls, public tuning and retained posterior continuation

**Authorized continuation. Owner: “I approve. continue”, 2026-09-25.** The
master's existing goal remains testing surrogate HMC. This proposal covers an
M13 posterior pilot after the successful native four-chain connection. LG1 stays
closed, Claude review remains deferred, and other-model gaps remain separate
items of the same master. It does not promise all-model completion.

## Question and evidence contract

For the fixed data1704/T20 two-parameter M13 problem, does canonical analytical
DCTR-force surrogate HMC produce useful posterior samples agreeing materially
with the independent same-data reference? The state dimension is4, observation
dimension2; parameters are lambdaQ/lambdaR, with independent N(0,log(2)^2)
priors. The FP32/TF32/GPU/XLA force remains the default. The independent frozen
full-support4096x8 trajectory-importance endpoint and public FP64 coordinate
interface remain explicit reference/compatibility precision roles.

Current evidence: finite-grid refinement and domain sensitivity passed, endpoint
differences were materially small, and the public four-chain kernel worked.
Controls are not selected. The40-transition mechanics run proves neither
mixing nor posterior correctness. Data1704 has already informed diagnostics;
this is an exploratory same-data pilot, not an untouched confirmatory claim.
The reserved data3701 confirmation is excluded from this budget.

Primary posterior evidence uses mean and SD agreement in posterior-SD units,
marginal CDF agreement at reference quartiles in probability units, and paired
reference uncertainty where available. Freeze margins before new chains:
means .1referenceSD, SDs .1referenceSD, CDFs .05absolute probability. Preserve
point differences and simultaneous MCSE-aware intervals separately. Rank/folded
R-hat<=1.01, bulk and tail ESS>=400, finite endpoint/state/energy and movement
checks are required for a qualified pilot. High acceptance alone cannot pass.
Practical agreement with an interval too wide to establish equivalence is
reported as practically favorable but formally inconclusive; it is not a
candidate rejection or permission to relax margins after looking.

Numerical invalidity, wrong endpoint/data/prior, corrupted provenance or missing
required reference diagnostics are continuation vetoes. Failed control candidates
or tuning/convergence checks trigger their bounded repair; they are not rejection
of surrogate HMC. No step may relabel the DCTR force as the gradient of the
independent endpoint. No result here establishes all-model validity, superiority,
or a rigorous posterior tail bound outside the checked region.

## Work in dependency order

1. Finish a scope-specific control protocol. The old.001 relative change of
   forward filter statistics is an engineering diagnostic and does not prove
   posterior harm. Evaluate the numerical-protection family using non-harm on
   healthy trajectories and finite, flagged behavior on stress trajectories.
   Include matched-momentum proposal/MH materiality checks, with a prospective
   .01 acceptance-probability allowance, before freezing a candidate. This
   protocol needs implementation and focused tests before execution; the
   mechanics candidate .01 is only a warm start. Use fresh independent
   calibration/validation streams1705/2705; preserve earlier failures. Bind the
   issued selection to the actual particle count, controls, data regime,
   dtype/device/chunk policy and source dependencies. Claim3701 remains unused.
   If no safe candidate can be selected, use the bounded repair budget and
   report that result instead of launching unqualified sampling.
2. Use the repository-issued position-only-force binding with the public
   `tune_hmc_kernel` and `VerifiedTensorFlowHMCConfig`. Nomination: two mass
   windows128/256, up to256 step-adaptation and128 screening transitions per
   candidate, candidate L in{1,2,4}; no unbounded ladder. Prior-scale geometry
   is a warm start, followed by target-specific window adaptation with its
   explicit regularization diagnostic. Initial normalized step .2 is a
   hypothesis, with the actual public adaptation tracing its failures and
   changes. These inherited tuning settings are not promoted defaults.
   Initialize fresh verification from four distinct prior-scale positions,
   archive/discard500 warmup and at least1000 verification draws per chain.
   Freeze selected controls/geometry/step/L before retained sampling. R-hat
   admission and all health checks must pass. L<=4 is a bounded initial search,
   not a claim that longer trajectories are inferior or unnecessary; failure
   can nominate the declared longer-trajectory repair if budget permits.
3. Grow retained sampling in250-transition chunks to prospective looks1000 and
   2000 per chain. Archive warmup and verification but exclude them from these
   estimates. Preserve each chain, endpoints, acceptance, force-fallback and
   energy diagnostics. Account for the two looks when forming intervals.
   Stop early only under the predeclared materiality/precision rule; otherwise
   finish2000 or report the stated budget/validity stop. Do not silently sample
   longer to obtain a favorable interval.
4. Construct cheap diagnostic adversaries: direct draws from the finite-grid
   reference, a tuned random-walk Metropolis sampler of the same frozen
   endpoint, and an endpoint-gradient HMC comparator where its differentiation
   contract has been checked. Evaluate center and marginal-tail summaries
   separately. IID grid draws validate uncertainty reporting for the discrete
   diagnostic law; they are not exact continuous-posterior samples. At minimum
   the random-walk baseline and independent reference must be present before
   interpreting the complex method as useful. Any efficiency claim additionally
   requires matched posterior precision and uncertainty on cost; this initial
   budget does not authorize a superiority claim.

The next executable modules/configurations must implement these dependencies
and fail closed on missing control selection. Their public-tuner connection is
already tested by the cost stage. The master currently contains the completed
connection/recovery stages; a qualified M13 retained-sampling stage is still to
be implemented. The owner approval authorizes this bounded continuation,
not claim that unfinished code or scientific prerequisites are already satisfied.

## Measured budget and stop conditions

Remaining authorized CPU time is1374.5054seconds, about23minutes. Measured
four-chain cost was about9.04seconds/transition atL2 and15.44seconds atL4. The
canonical four-row force alone cost3.0736seconds, versus.0780seconds for the
endpoint. First-call compilation is extra. The completed cost worker charged
about1.165 CPU worker seconds per GPU-wall second including compilation; use
1.2 for planning. Runtime measurements are descriptive and workload-dependent.

With50% time headroom, L4 costs23.1566seconds/four-chain transition. Even1000
fresh verification transitions alone cost about6.43 GPU hours or7.72 CPU
worker hours at that rate. A conservative nominal upper accounting is384 mass
transitions, up to3*(256+128) candidate transitions,500 discarded warmup,1000
fresh verification, and2000 retained transitions:5036 total. Charging all atL4
is intentionally conservative; it gives32.39 GPU hours/38.87 CPU worker hours
before other allowances, while the shorter L1/L2 stages reduce that forecast.
The staged hard reservations below, not the forecast, bound execution; no
stage starts if its remaining reservation is unavailable.

Approved **an additional40 CPU-worker hours (144000seconds)**. Retain the existing
226800-second GPU ceiling;185147.30 GPU seconds remain. The CPU campaign ceiling
is now198000seconds (55hours), raised from54000 by the owner approval.
Reserve up to126000 of the existing remaining GPU seconds for this continuation.

| Stage | CPU worker seconds | GPU wall seconds | Stop/repair |
|---|---:|---:|---|
| Controls, reference diagnostics, implementation checks and cheap baselines | 6000 | 6000 | No selected safe control or invalid reference prevents inference |
| Public nomination plus fresh verification | 60000 | 50000 | Archive rejection; repair only within reserve |
| Retained posterior pilot through at most2000/chain | 56000 | 48000 | Health, precision or cap; no automatic extension |
| Local repairs and bounded L/initialization retry | 22000 | 22000 | Total continuation cap is final |
| Total | 144000 | 126000 | No extra paid or different-hardware compute |

On 2026-09-27, 12000 CPU and 12000 GPU seconds were transferred from unused
public tuning allowance into the repair allowance. The total remains unchanged.
The evidence, one-attempt limit and exact command are in
`docs/plans/surrogate-hmc-m13-initialization-repair-2026-09-27.md`.

These are maximum allocations, not predicted consumption or guaranteed closure.
The unmeasured L8 forecast is42.35seconds/four-chain transition with headroom;
it may enter only a small repair that fits the remaining allocation, not silently
double the full campaign. No unbounded retry, external service, package change,
Claude review, or other-model campaign is included in this expansion.

## Skeptical audit and decision readiness

The largest risk is spending a day on high-acceptance chains that explore poorly.
Address it with the tuned random-walk comparator, prospective movement/ESS/R-hat
checks and posterior reference agreement. The force is expensive despite FP32;
there is currently no basis for a speed benefit. A failed long run can result
from control selection, metric/step tuning or insufficient mixing, so preserve
those distinctions. Selection uses fresh calibration/validation streams; data1704
results remain exploratory. A finite-grid oracle cannot prove global tail
accuracy, and repeating prior diagnostics does not remove that limitation.

This plan survives the scope/baseline/stop-condition audit as a bounded next
phase, subject to implementing and testing the stated control and comparison
protocols. The resource increase is approved; this is not a ready-to-promote method. Exact launch commands, manifests, seeds, artifact roots and tests must
be recorded before each concrete stage runs under the master. A budget approval
does not waive any mathematical or numerical prerequisites.

## Execution checkpoint

2026-09-25: owner approved the stated 144000 CPU-worker-second expansion.
Global limits now198000 CPU/226800 GPU seconds; prior charges52625.494613292
CPU/41652.696728097 GPU. This continuation is additionally bounded by its
144000 CPU/126000 GPU allocation, including failed attempts. Skeptical audit:
current .01 controls are unselected; materiality selection must be prospective
and use independent streams before tuning. The actual public position-only
force binding is required; tiny-step acceptance cannot establish convergence.
Implement and test the selection/tuning consumer before GPU launch.
