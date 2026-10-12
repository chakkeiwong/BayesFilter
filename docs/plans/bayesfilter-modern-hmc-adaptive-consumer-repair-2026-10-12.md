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
