# SQMC fixed-data accuracy check at 1,008 particles — 2026-09-26

## Question and scope

Do the conspicuous first-score errors in the corrected P44 diagnostic persist
with at least 1,000 particles, and how variable are they across independent
filter randomizations on the same observations? The user requested this check.
Use /home/chakwong/BayesFilter-SQMC, branch sqmc-development. This is a
diagnostic of the current analytical recursive implementation, not admission.

N=1008 matches the earlier campaign count and satisfies the current Contract-E
residual design N divisible by 2D, with D=3. N=1002 would also satisfy it.
Transport chunks obey the repository rule K=N=1008.

## Evidence contract and research intent

- Target: the four derivatives of the exact P44 log likelihood at the saved
  theta, on each of the two saved T=2 datasets (93001 and 93002).
- Computed quantity: analytical derivative of the finite-particle likelihood
  program, compared with the exact Kalman derivative on identical observations.
- Comparators: exact Kalman oracle; saved corrected N=12 results; all four
  current routes including the IID baseline.
- Primary completion criterion: finite values and all four score components
  for every requested N=1008 cell; raw scores, signed/absolute errors and
  absolute vector error. There is no relative-score accuracy threshold or
  method-promotion criterion in this diagnostic.
- Validity/continuation vetoes: changed data/oracle, nonfinite output, broken
  canonical validity flag, inconsistent directional values, failed GPU/XLA
  versus graph parity, source change during execution, budget exhaustion.
  Preserve failures; do not rank invalid cells.
- Explanatory diagnostics: signed first-score error, full vector L2 error,
  log-likelihood error, independent-scramble mean/SD/MCSE and observed range,
  device, timing, allocator peak. High error is a repair trigger, not evidence
  of a derivative bug by itself or a veto on the research direction.
- Do not conclude: global derivative correctness, unbiasedness, particle-number
  convergence, method superiority, long-horizon accuracy, production/default
  readiness or HMC readiness. No controls are selected on these results.
- Heuristic adversary set for this restricted question: exact Kalman (available
  conditional Gaussian oracle); existing IID particle route (plain stochastic
  counterpart); zero score (stationary-point sanity comparator, diagnostic
  only). Construct their absolute errors separately on each dataset. No
  promotion is being attempted; no route can be promoted by beating these.
- Conditional situations: each dataset separately, especially the first-score
  sign discrepancy at data seed 93001. Do not pool the two datasets to obscure
  the discrepancy.
- Artifacts: unique attempt directories under
  docs/plans/artifacts/sqmc-n1008-20260926; result note
  docs/benchmarks/sqmc-n1008-results-20260926.md.

## Defaults and assumptions audited before execution

| Choice | Provenance and reason | Risk and earliest check | Status |
| --- | --- | --- | --- |
| P44 D=3, T=2, theta and data | Exact troublesome saved cases | Narrow horizon; replay exact Kalman values/scores against saved artifacts | Diagnostic baseline |
| N=1008 | User minimum 1000; prior campaign count; reset requires multiple of 6 | Not a particle-count convergence study; report that limit | Requested test scope |
| Four routes and controls | Saved corrected N=12 workflow, identical controls except route ancestry/cap | Controls are UNTUNED at N=1008; in particular only two flow substeps. Report approximation error without promoting controls | Frozen comparison baseline |
| GPU float64 XLA | Accelerates larger clouds while retaining prior reference precision | Backend change; compare graph and XLA on identical frozen CPU-generated inputs for every route on 93001, tolerance 1e-7*(1+abs(reference)) | Explicit FP64 reference exception |
| CPU data/input generation | Reproduce prior stateless random generation and freeze input tensors | CPU/GPU RNG mismatch; preserve tensors and hashes, check saved oracle before GPU execution | Reproducibility choice |
| Original seed plus eight new scrambles | Original seed preserves direct cell replay; seeds 94001..94008 are fixed in advance and independent of the data seeds | Eight gives limited uncertainty; SD/MCSE are descriptive, no ranking or tail inference | Bounded diagnostic design |
| Ridge, damping, trust and cap settings | Frozen from saved controls; no default or safety-policy change | Approximation may remain biased after more particles; report as possible explanation | Untuned at this scope |
| GPU0 memory growth | Available hardware, verified before initialization | Other workloads; record device and allocator peak; stop on allocation failure | Resource policy |

Per-scope tuning is required before a claim-bearing run. This deliberately
changes only particle count (plus verified backend) to diagnose the existing
result. It does not transfer the N=12 tuning artifact to N=1008 and does not
describe these controls as tuned.

## Execution, budget and skeptical review

One runner freezes the two CPU-generated observations, checks their exact
Kalman oracle against the saved results, writes all input arrays as JSON,
and executes GPU XLA values/scores through the existing public campaign
entrypoint. It first checks XLA/non-XLA GPU parity on the original 93001
inputs for each route. Non-XLA is an explicitly reference-only exception.
It then runs all four routes on both datasets with their original input seed,
followed by eight independent filter seeds per dataset and route: 72 XLA
cells in total. Input generation is shared for the three Halton routes.

Command: TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=0
/home/chakwong/anaconda3/envs/tftwogpu/bin/python
docs/benchmarks/run_sqmc_n1008_diagnostic.py --output <fresh-attempt-directory>.
The launcher enforces a process timeout and records the full log. At most
three attempts and 1,800 seconds total process time (including imports and
compilation) are allocated. The first attempt has a 1,500-second timeout;
the runner stops starting new cells after 1,350 numerical seconds. A repair
uses only the remaining budget and a new attempt directory. No package
changes, remote writes or unrelated workloads are authorized by this plan.

Skeptical pre-run review: PASS for this limited question. The baseline is
the exact same model/data/parameterization, not a relative error near zero.
Keeping numerical controls fixed isolates the particle-count intervention;
it does not establish their adequacy at N=1008. Backend parity addresses the
other changed variable. Additional scrambles expose a lucky original draw.
T=2, eight randomizations, and untuned controls cannot certify a production
method. Successful execution with large remaining bias must be reported as
such; no threshold is relaxed and no flow/grid retuning is hidden in this
comparison. The earliest misleading-pass check is exact data/oracle replay,
followed by cross-backend parity. Observed score error alone does not show a
derivative implementation error; prior finite-difference evidence remains
local and separate from exact-likelihood accuracy.
