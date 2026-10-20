# NeuTra warm-start repair campaign: terminal review

The declared 30-case matrix finished at 11:26:11 Asia/Shanghai on September 30.
Twelve cases passed the final sequential-HMC checks. Reliable training across
all five benchmark targets remains unestablished. No campaign worker remains
active; the declared repair ladders are exhausted, while compute remains.

| Target | SMC final passes | Gabrié final passes | Remaining failure |
|---|---:|---:|---|
| Gaussian | 3/3 | 3/3 | None in these finite cases |
| Unwarped mixture | 0/3 | 0/3 | No warm map passed all fit/shape checks |
| Warped mixture | 2/3 | 2/3 | One SMC HMC-search failure; one Gabrié fit failure |
| Wiggle | 1/3 | 1/3 | Four HMC-search failures |
| Funnel | 0/3 | 0/3 | Three SMC teacher failures; three Gabrié HMC-search failures |

These are descriptive counts from three seeds, not evidence ranking the two
generators. The eight `fresh_final_check_failed` cases specifically failed to
find a verified HMC step-size/leapfrog pair within the bounded search. They did
not reach a retained-posterior comparison. The twelve final passes have saved
sequential warm-up and retained checks, empty hard-veto lists, and recorded
exclusion of warm-up from estimates. They retained 1,000–6,000 transitions per
chain after 2,000 warm-up transitions in the inspected successful cases.

## Findings and interpretation

The ordinary mixture still exposes a training problem, but it is too strong to
say no map learned nonlinear structure. The repaired SMC seed-11 width-8/.001
fit reached forward KL 0.174 at 8,192 updates against the development Gaussian
baseline 0.947. It passed the nonlinear-learning screen and the shape-precision
screen, but failed shape agreement. Other inspected configurations remained
near the Gaussian plateau. The specific evidence is
`attempts/closure-fit-mixture-smc-s11-v1-r1/calibration-progress.json`; the
Gabrié seed-11 repaired fits in the parallel directory stayed near that
plateau. These diagnostics identify separate residual shape and optimization
questions; they do not support one explanation for all failed maps.

The funnel's repaired calibration selected a curvature-scaled time step rather
than silently using a rejected candidate. The controller then ran the planned
teacher and fitting stages. All three SMC teachers still failed the independent
accuracy/precision assessment at the 8,192-particle rung. Gabrié maps reached
the final qualification phase in all three seeds, but no bounded HMC search
produced a verified pair. Successful short mutation acceptance therefore did
not establish teacher accuracy or useful HMC geometry.

The earlier stop was a controller defect: a legitimate calibration rejection
escaped as an exception. Its typed failure handling and curvature-based pilot
extension passed the focused 67-test suite. The resumed workers recorded GPU
memory growth and completed the pending funnel matrix. This repairs that
execution defect without upgrading the failed scientific candidates.

## Evidence and accounting

The terminal audit is
`artifacts/neutra-warm-start-master-2026-09-29/campaign-r1/terminal-review-20260930-r1.json`.
It checked all 30 terminal outcomes against phase reports, all 600 archived
source-file checksums associated with those outcomes, GPU memory-growth records,
and the twelve successful posterior reports. No inconsistencies were found in
those checks. This was artifact verification, not an independent statistical
reanalysis or a proof of the entire implementation. Earlier targets retain
their original source-version evidence; only the funnel was rerun after the
calibration repair.

All 176 phase attempts, including the failed calibration attempt, consumed
9,833.48 worker wall-seconds, 9,672.16 GPU process-seconds (2.69 hours), and
15,198.62 CPU core-seconds (4.22 hours). The roughly nine-hour calendar span
includes the stopped interval; it is not nine hours of computation. These
amounts cover the repair controller's recorded workers, not earlier campaigns
or interactive editing and focused tests. Historical costs remain in the
shared ledger, including the conservative charge for an earlier uncertain CPU
counter. Remaining launch allowance is 94,677.84 GPU process-seconds (26.30
hours) and 189,626.18 CPU core-seconds (52.67 hours).

Per-attempt `manifest.json` files record exact commands, source hashes,
git commit, interpreter, seeds, inputs, device and allocation policy. The
checkout commit is `de80aaff5812ebfbed551977476c0868551a2c88`; dirty source
snapshots, not that commit alone, identify the executed code. The campaign
uses `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`, host GPU 1, batched
TensorFlow/XLA kernels and separately labeled CPU preparation/reference work.
The final recovery command was
`scripts/run_neutra_warm_start_repair_master.py run --targets funnel` under the
recorded trusted systemd service. The active plan and detailed repair review
are `bayesfilter-neutra-warm-start-repair-master-2026-09-30.md`.

## Decision

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Preserve 12 passing cases | Declared sequential posterior checks passed | No recorded terminal hard veto | Finite seeds and benchmark scope | Independent confirmation before stronger claims | Reliable all-target training or q20 readiness |
| Reject current unwarped-mixture fits | Warm fit/shape requirement failed | Candidate promotion blocked | Plateau versus remaining nonlinear shape error | Inspect saved shape residuals and update histories before a targeted training repair | Failure of canonical IAF or all forward initialization |
| Reject current funnel SMC teachers | Accuracy/precision failed at capped particle ladder | Dependent SMC training blocked | Proposal overlap and mutation mobility | Diagnose weighted overlap and movement separately | More particles alone or high acceptance solves the funnel |
| Preserve eight HMC-search failures | No verified pair | Posterior sampling not admitted | Learned geometry versus bounded tuning range | Inspect candidate verification evidence before changing map or search | Incorrect posterior estimates from chains that were never admitted |

| Inference status | Finding |
|---|---|
| Hard veto screen | Current candidate failures block promotion; no newly detected artifact inconsistency |
| Statistically supported ranking | None |
| Descriptive-only differences | Pass counts, loss values, lengths and runtimes |
| Default-readiness | Not established |
| Next evidence needed | Targeted, predeclared repairs followed by fresh downstream checks and independent replication |

The strongest alternative explanation for a rejected map is inadequate
calibration or an overly coarse bounded search, rather than a failure of the
research idea. A map passing the unchanged shape criteria and fresh HMC
verification would overturn that candidate-level failure. The weakest evidence
is inference from three training seeds and development screens to a general
training procedure. Failure to pass all targets prevents admitting the present
recipe as a reliable general procedure; it does not invalidate the targets,
mathematics or warm-start research direction.
The next work should distinguish the failure mechanisms above before spending
the remaining allowance on another broad repetition. Full upstream Gabrié
controller equivalence and AFT/CRAFT repairs remain outside this completed matrix.
