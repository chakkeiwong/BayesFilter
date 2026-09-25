# Supplied-whitening funnel execution and next repairs

The supplied-map engineering checks passed. The fixed-transport public tuner
produced verified candidates for the exactly whitened funnel in both seeds.
The two partial maps needed the predeclared intermediate-epsilon follow-up;
both then produced verified members in both seeds. This establishes the tested
search, verification, retention and replay behavior given those frozen maps.
It does not establish satisfactory posterior sampling: every selected posterior
failed its separate assessment.

This executes the [reviewed plan](bayesfilter-hmc-supplied-whitening-plan-2026-09-22.md).
The untransformed centered funnel remains a bounded-failure stress test.
Learning or improving a transport is upstream work, not a requirement that
ordinary epsilon/L tuning must solve arbitrary funnel geometry.

## Repair and validation

The validation harness previously passed base-adapter starts to a public API
that expects latent starts. That was wrong for a comparison claiming identical
physical starts across maps. It did not change the evaluated density. The
harness now applies the reconstructed map's inverse, checks its forward
roundtrip, and saves model, adapter and latent starts. Tests also inspect the
actual saved numerical binding to ensure the corrected bank reaches the tuner.
Old experiments retain their original records and actual starts.

The exact control composes the existing noncentered chart and a frozen affine
scale. The partial fixtures use the supported dense-IAF codec to represent
`s(v)=6*tanh(a*v/12)`, with a=1 or 0.5. Neither fixture is trained. Independent
density, score, Jacobian, inverse, zero/tail and matched-start checks passed,
as did real public tuning, multiple-member retention, export/reload and warmup
exclusion. Existing Gaussian, banana and Dirichlet transport integrations and
position-field checks also passed.

There are **80 distinct passing checks** across the three recorded commands.
The first command passed 28 checks. The second passed 62 and exposed one
ordinary positive-handoff test using an inadequate 12-transition smoke budget:
one repeated MH state among its four final-window rows made the four-start
bank unavailable. The rejection was correct. The positive test now uses the
existing standard preparation preset, with the same seed and assertions.
The changed test and two existing negative start-bank tests all passed. No
test is skipped, no dispersion condition was relaxed, and the failed smoke
record is preserved in `tests-r2/failed-smoke-preparation.json`.

The official book source under `docs/main.tex` and its tuning chapter, plus the
agent/API reference, now explain supplied geometry, exact transformed targets,
latent starts and the separation from posterior assessment. The book rebuilt
successfully with no undefined citations or references; the changed PDF page
418 was visually inspected. `docs/main.pdf` contains the rebuilt local copy.

## Numerical results

Seeds are 2026092251 and 2026092252. Every fit uses the same four physical
starts and L=(3,5,9,13,18,25), the unchanged acceptance policy, fresh measurement
and verification, and unchanged posterior requirements. The selected member
is the first verified identity, fixed before posterior assessment. Every other
verified sibling remains retained and is explicitly unassessed.

| Map and search | Candidates, seeds 1 / 2 | Verified, seeds 1 / 2 | Selected posterior outcome |
| --- | ---: | ---: | --- |
| Exact map, native search | 44 / 51 | 6 / 8 | Both reached 10,000 retained draws per chain; requested precision unmet, no chunk health veto |
| Partial map a=1, native search | 12 / 13 | 0 / 0 | No posterior started |
| Partial map a=0.5, native search | 7 / 10 | 0 / 0 | No posterior started |
| Partial map a=1, intermediate grid | 100 / 100 | 3 / 3 | Seed 1 failed warmup health at 1,500; seed 2 failed retained health at 10,000 |
| Partial map a=0.5, intermediate grid | 100 / 100 | 2 / 2 | Seed 1 failed warmup health at 500; seed 2 failed retained health at 8,500 |

Native partial searches encountered high finite acceptance at small steps and
nonfinite proposals at larger steps; some measurement survivors lost fresh
verification. The finite/nonfinite boundary did not provide valid opposing
acceptance evidence. One explicit intermediate grid with the existing optional
failed-interval exploration found survivors. The follow-up reached its
100-candidate limit, so it does not establish exhaustive coverage. Grid density
and refinement changed together; this experiment cannot attribute the outcome
to one feature or establish a statistically supported improvement.

The four selected partial-map posteriors encountered nonfinite log acceptance,
proposed target/score, momentum or acceptance correction. Retained states could
remain finite while these proposal-health checks failed. Those failures veto
these posterior runs; they do not silently delete a previously verified member
from its tuning inventory or prove that every sibling would fail.

Even the exact latent Gaussian control has heavy-tailed model quantities.
For each child, `Var(x)=E[exp(v)]=exp(9/2)`, since its conditional mean is zero
and v is N(0,9). Thus 40,000 independent model draws give mean MCSE about
0.0474, close to the requested 0.05. This is a planning comparator, not a lower
bound on HMC uncertainty. The selected runs' lugsail child-mean MCSEs were
0.06777/0.06094 and 0.07035/0.08688; the scale mean and all three medians passed.
Dependence and tail variability need separate investigation. The existing
MCSE calibration gap prevents interpreting the estimates as validated coverage.
R-hat, ESS and MCSE did not qualify or rank any tuning candidate.

## Provenance, audit and budget

The compact [terminal audit](artifacts/hmc-repair-master-2026-09-16/m20-r2/terminal-audit.json)
checks all ten complete fits, 537 candidate records, 754 distinct receipt seed
pairs, all 24 verified members, physical starts, source closures, checkpoint
agreement and selected/unassessed inventories. The public pipeline independently
checked each candidate inventory during execution. Exact commands, maps,
designs, code hashes and raw numerical archives remain under `m20-r2/`.

The environment is `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`, with GPUs
deliberately hidden and one numerical thread. These are CPU reference and
development runs with graph execution and an explicit non-XLA exception; they
do not establish GPU/XLA readiness. The trusted capacity probe found no idle
policy-permitted GPU. The package baseline is `01d67ec41` with the recorded
supplied-map edits; the saved execution closures match the inspected source.
The unrelated live M21/M22 queue still uses frozen `f9c86f41a` and cannot certify
these later changes.

The ten fits plus the initial pre-import launcher path failure cost
1199.752751 CPU worker-seconds. Test commands cost 367.837813 seconds. The
terminal audit, including a repaired reader assumption about the baseline's
missing optional `search` field, cost 0.743899 seconds. Total tranche charge is
**1568.334463 CPU seconds (0.436 hours)**, below its 3600-second allowance;
GPU charge is zero. No numerical run or result was discarded after failure.
The temporary additional-worker exception has ended.

At the reconciled snapshot, the continuation has charged 20464.122420 CPU
seconds. Its uncharged allowance is 152335.877580 seconds, including 69000
seconds still reserved for uncompleted M21 jobs; running work will be charged
when its receipts close. GPU allowance remains 86400 seconds. The live ledger
is `m21-r2/reconciliation-progress.json`; these numbers are a dated snapshot.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close supplied-map/start-coordinate engineering cell | Algebra, actual public route, retention and checkpoint/replay tests pass | No invalid map, start, inventory or source finding | CPU fixtures and finite tested region | Preserve regressions; repeat on permitted GPU/XLA | Learned transport quality or arbitrary-map robustness |
| Keep residual geometry and posterior usability open | Partial-grid members verified, selected posteriors failed | All four selected partial posteriors vetoed by proposal health | Unassessed siblings and remote tail geometry | Inspect saved failure locations before choosing a new map or posterior study | Every verified member is posterior-ready |
| Keep precision and stopping calibration open | Exact-map means missed the unchanged tolerance | No chunk health veto in exact runs | Dependence, model tails, estimator and stopping calibration | Finish M21; then plan member-specific precision using model units | Whitened coordinates guarantee model-moment precision |
| Continue independent campaign | Valid fixed inventories remain funded | No continuation veto | M21 results pending; GPU capacity and exact consumer inputs absent | Complete/audit queue and refresh its next phase | Completion of the whole repair program |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Four selected partial-map posteriors failed; exact-map runs had no chunk health veto but failed precision |
| Statistically supported ranking | None; two seeds per map and combined search changes cannot support ranking |
| Descriptive-only differences | Candidate counts, acceptance behavior, MCSE estimates and times |
| Default readiness | No acceptance, search, posterior or learned-map default promoted |
| Next evidence needed | Posterior safety under residual curvature, calibrated stopping/precision, GPU repetitions, complete-fit defect power and exact consumer replay |

## Refreshed continuation

1. Finish and audit the unchanged M21 inventory: 128 Gaussian and 128
   beta-binomial fits, plus eight rotated-Gaussian and eight LGSSM controls.
   Retain caps, missing intervals and failed fits in the denominator. Diagnose
   fixed-count estimator behavior separately from optional stopping before any
   default change. The completed M22 frozen-Gaussian null confirmation passed
   its predeclared screen; broader calibration and subtle full-fit power stay
   open, as recorded in its separate result.
2. Use the saved partial-map failures to locate residual curvature and assess
   proposal health. There is no third epsilon search in this development
   tranche. A future study must predeclare either unchanged-map sibling
   assessment or an improved supplied map with a new tuning scope. For exact
   maps, plan posterior length in model-quantity units and preserve the 0.05
   request unless the scientific requirement itself changes.
3. Complete current-source GPU integrations when permitted capacity is
   available. Keep global mixture occupancy and target-specific learned-map
   quality separate from this supplied-funnel test. Full-fit defect power needs
   a measured-cost redesign with an adequate fixed replication inventory;
   the earlier approximately 20-CPU-hour cost per arm remains a real constraint.
4. Obtain the exact MacroFinance target/data/prior/coordinate bundle and a
   matched reference for its integration. Preserve bounded exception recovery
   and the limits of XLA failure attribution. Profile actual remaining costs
   before refactoring; compile or smoke success cannot substitute for these
   numerical checks.

Post-run skeptical review: the strongest alternative explanation for the
positive exact-map result is that its analytic chart makes the latent problem
Gaussian; this is precisely its control role. Partial-map survivors could be
short-run successes that miss unstable tails, and the later posterior failures
show that limitation directly. Two seeds cannot locate a universal safe map
quality or rank search settings. A failure of source, map, score, Jacobian,
start or inventory identities would invalidate the engineering result; none
was found. The current candidates failed posterior requirements, not the
research direction. The next work remains separate calibrated posterior and
geometry assessment, with no relaxed qualification rule.
