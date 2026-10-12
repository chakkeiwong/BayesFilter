# D4R4c adaptive comparison consumer repair

Date: 12 October 2026. This is the executable engineering dependency between
the [window ladder](bayesfilter-mams-d4r4b-preconfirmation-2026-10-12.md) and
the full-model comparison in the [master](bayesfilter-modern-hmc-implementation-plan-2026-10-07.md).
The structural 1024/256 diagnostic may continue independently from its frozen
source. No numerical sampler or tuning-admission default changes here.

## Question, baseline and scope

Can the comparison consumer faithfully execute a frozen proposal cohort with
steps above one and duration-preserving repair descendants while retaining
every member that the public tuner verifies? The baseline is the already
tested `run_adapted_mams_search` public-tuner/export/reload path. The older
`run_modern_hmc_method_fit.py` is historical D4b execution: its exact-count
assertion, domain `[.001,1]`, disabled repairs and old proposal/reference
loaders cannot support the new cohort. Do not reinterpret its old artifacts.

Factor the existing public dispatch, export and reload into one reusable
frozen-cohort consumer. MAMS continues to obtain its proposals from the shared
discarded pilot, then calls this consumer. Fixed HMC, randomized HMC and
SNAPER proposals can use the same frozen-cohort consumer with their own
explicit binding and search configuration. This is not a third tuner.

The consumer must validate the binding's actual epsilon domain, acceptance
policy and candidate allowance before creating an output directory. Reserve
the declared repair opportunities when computing the candidate/error cap.
Do not impose the old epsilon upper bound or assume that proposal and terminal
candidate counts agree. Do not use posterior diagnostics or short-path prices
to change membership. A rejected parent and its verified child both remain
in the result; a verified child must have a valid recorded parent chain.

Record requested primary pairs, uncreated primary pairs, every created
candidate's identity, parent, family, numerical parameters and disposition,
all repair actions, and every verified export. Primary membership is checked
by pair only for roots. Descendants keep identity even if a different family
has the same numerical pair. If the public tuner pauses or exhausts its
budget, preserve its actual inventory instead of fabricating passes or
requiring a completed cohort. Fresh public verification and checked replay
remain mandatory for every export.

## Evidence contract and implementation order

| Item | Role and decision |
| --- | --- |
| Real public tuning, export/reload and continuation on ordinary and fixed-transport bindings | Engineering pass criterion |
| Wrong epsilon domain, policy mismatch, unfunded candidate cap, unrequested roots, missing/invalid repair ancestry, export-set mismatch | Fail before dependent use; a shared defect stops affected execution |
| Numerical candidate rejection or incomplete search | Recorded candidate outcome; not a direction veto |
| Posterior reference, R-hat, ESS, MCSE, time and acceptance means | No consumer-membership authority; this engineering phase makes no method ranking |
| Unique source and outputs, unchanged shared tuning authority | Reproducibility requirement |

1. Implement the common frozen consumer and use it from the existing MAMS
   consumer, preserving the existing pilot/search directory layout and fields.
2. Test epsilon above one on actual HMC and MAMS, both coordinate routes,
   actual repair descendants, paused/uncreated inventories, deliberate
   missing exports and wrong ancestry. Reuse existing LGSSM and nonlinear
   accumulation integration checks to guard MAMS wiring.
3. Run the focused CPU reference/mechanics suite with GPUs deliberately
   hidden. Use XLA where the test requests it; graph-only fixture checks are
   explicit CPU mechanics exceptions. No GPU performance claim follows.
4. Commit the source and refresh the master. Only then build the full-model
   consumer on this tested foundation. Its fresh reference bridge must bind
   the final committed target closure; the old report must not be asserted
   current after source changes.
5. Before a posterior comparison, freeze and price a complete design for
   adequately searched fixed/randomized HMC, SNAPER and adaptive MAMS, plus
   the old MAMS grid as a causal diagnostic where funded. Account for every
   pilot, candidate, repair, failed trial and posterior member. Use fresh
   paired seed replications, model/tail quantities and uncertainty, and keep
   unfunded cells visible if the full design exceeds the remaining grant.

The pass criterion for this phase is correct routing and complete inventories,
not sampler convergence, a robustness gain, nominal MCSE coverage or release.
There is no new numerical threshold. Test epsilons, trial counts and seeds are
mechanics fixture inputs, inherited where possible; they are not scientific
defaults. Local tests use the existing 3600-second diagnostic reserve, whose
known use before this phase is 792.834369 seconds. Keep wall time and output
under `N5/generic-adaptation/preconfirmation-01/consumer-checks-01/` and settle
it once. Stop or reallocate within the grant before exhausting the reserve.

## Skeptical pre-execution review

The source of the problem is the consumer's old fixed cohort assumptions,
not evidence that the shared controller loses members. The chosen factorization
reuses that controller rather than reproducing its acceptance or repair logic.
The audit checks unfair comparisons: a successful adapted MAMS cohort alone
cannot establish improvement against unsearched classical controls. It checks
proxy metrics: short prices and pilot availability have no qualification
authority. It checks missing outcomes: repairs, paused roots and failures
remain explicit. It checks environment mismatch: the active GPU run retains
its immutable source; CPU mechanics do not become GPU evidence. Existing
reference compatibility is not silently transferred to new source.

Review verdict: proceed with steps 1–3 under the existing local reserve;
steps 4–5 remain necessary before launching claim-bearing comparisons.
The smallest misleading success would be exporting only the old proposal
count after a child verifies. The deliberate descendant and missing-export
regressions directly test that failure. A consumer test cannot establish
statistical calibration; the antithetic MCSE and stopped-chain reference
questions remain separate open work.

## Execution result

The common frozen consumer is implemented in
`bayesfilter/testing/inference_validation/modern_tuning_consumer.py` and the
existing MAMS consumer calls it. It preserves the existing pilot, tuning and
member directory locations; moving completed checkpoints was rejected in
review because those checkpoints contain absolute paths. Existing outputs
cannot be overwritten. All actual candidates, their parent/root identities,
repair actions, uncreated primary pairs and verified exports are recorded.

Twenty-four distinct regressions passed: four above-one HMC/MAMS tests across
both public coordinate routes; a real acceptance rejection followed by a
duration-preserving child, export and replay; fixed HMC; budget stop;
incomplete inventories; five ancestry/identity mutations; four configuration
failures before numerical work; overwrite prevention; missing-export detection;
and the existing ordinary/transport, LGSSM and nonlinear accumulation consumer
checks. The initial 22-test suite took 172.55 pytest seconds; conservatively
charge 180 seconds including startup. A follow-up had one incorrect test
expectation (`budget_bound` instead of the public `partial_budget` completion
status); after correcting only the assertion, the resource-stop test passed.
The two follow-up commands took 15.224024 and 5.304087 seconds. Total charge is
200.528111 seconds and known local usage is 993.362480 of the reserved 3600
seconds. The [test record](artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/preconfirmation-01/consumer-checks-01/result.json)
preserves commands, source hashes, environment and outcomes.

This closes steps 1–3 at the working-source level. The full-model executable
and fair, priced comparison design remain steps 4–5. The old fit driver has
not been silently repurposed. No posterior or statistical calibration claim
follows from these tests. The strongest remaining alternative is that a
correctly preserved adaptive cohort still explores only a local region;
independent posterior references and fresh cross-model comparisons must
answer that question.

## Full-model consumer implementation review

Step 4 will use an explicit frozen design file, with a typed transition,
controller and numerical execution configuration, geometry path, epsilon
domain, repair allowance, seed and optional posterior settings. Reconstruct
the existing prepared target and mass with those declared settings; use the
common frozen consumer for all methods. The design labels fixed HMC,
randomized HMC, SNAPER-proposed randomized HMC and MAMS, and verifies that
the label agrees with the executed transition. A label alone cannot establish
that a SNAPER/MAMS adaptation was performed; preserve its proposal provenance
and make no adaptation claim from an arbitrary frozen table.

The executable must validate the exact reference bridge before dependent
posterior work and assess every verified identity, including descendants.
Posterior settings and the finite-target agreement screen remain explicit
inputs. The inherited D4b screen, `(absolute error + 3 MCSE + reference
sensitivity)/reference SD <= .10`, may be used only when explicitly chosen
and labeled an operational screen without stopped-chain coverage guarantees.
Missing/nonpositive uncertainty cannot pass. Qualification-only operation
has no posterior authority and does not need a posterior reference. Resource
stops preserve public checkpoints and the complete result inventory.

Review before implementation: pass. This fixes the concrete consumer gap,
keeps the verified set intact, makes every numerical setting inspectable and
does not require changing target mathematics or the old archived loaders.
The final committed source still needs its reference audit and a priced
comparison allocation before serious posterior execution. Validate shared
binding construction, method labels, all-member posterior dispatch and
unavailable-reference outcomes with focused CPU mechanics tests.

The implementation audit found three issues before serious execution: imports
from `scripts` are not recursively included by the existing library-source
scanner; supervisor interruption can leave a stale running manifest; and
reference controls could otherwise be validated only after qualification.
Include the actual script helpers explicitly, validate posterior and reference
settings before target preparation, and preserve completed dispositions while
finalizing interrupted work. The reference screen must use the design's
declared precision estimator settings, rather than independently inheriting a
second batch-size or lugsail configuration. Exercise these failure paths and
one real full-horizon CPU CLI qualification smoke. This smoke tests routing and
serialization only; its small cohort, broad floor and short trials cannot
support a scientific admission, performance or posterior claim.

CPU reference execution may use XLA or the explicit graph-debug exception;
device choice and compilation choice are distinct. GPU campaign execution
still requires XLA. The first working-tree CPU CLI attempt was stopped after a
driver edit invalidated its source identity; preserve its 200.760842-second
charge and incomplete checkpoint. Repeat from a committed immutable source,
using CPU/XLA for the bounded full-horizon check. This is a local execution
repair, not evidence against a sampler or reason to relax qualification.

## Full-model consumer checks, 12 October

The explicit-design driver is implemented in
`scripts/run_modern_hmc_frozen_full_model.py`, with preparation and all-member
assessment in `modern_full_model_consumer.py`. Eighteen distinct regressions
pass: four actual full-data target preparations (PP-SGQF, PP-UKF, SIR and
structural), method-label rejection, shifted/constant reference controls,
explicit precision settings, peer continuation after a posterior veto,
resource stops, reference/member mismatch, interruption preservation of
completed members and repair descendants, source-helper inventory, seed and
policy preservation, and CPU/GPU compilation-policy separation. Posterior
dispatch tests are mocked engineering controls, not full-model posterior
results. The earlier common consumer's real public tuning and replay tests
remain the numerical routing authority for these checks.

The driver records the compiled operator, source helpers, exact design and
geometry inputs. It validates reference settings before qualification, keeps
missing uncertainty inconclusive, and finalizes supervisor stops without
discarding completed outcomes or uncreated roots. No numerical tuning default
or posterior threshold changed. The historical D4b driver remains unchanged.

Known local use is 1537.897999 seconds of the 3600-second reserve, including
the interrupted CLI attempt, all follow-ups and the 58.109641-second earlier
committed-source consumer check. Detailed commands and measured times are in
`preconfirmation-01/consumer-checks-01/`. The next action is a scoped commit,
immutable-source CLI check and fresh reference compatibility audit. The final
full comparison still needs its priced classical search and paired design;
the new driver alone does not establish adequate tuning, posterior delivery,
MCSE coverage, a method ranking or release readiness.
