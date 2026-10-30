# Lorenz Figure6 execution-cost pilot — 30 September 2026

## Research intent

Measure whether the validated Lorenz repeated-refinement/APF route can execute
the paper's40-forcing-by100-repeat experiment within the inherited GPU budget.
This pilot tests fresh-seed validity, stable graph reuse and cost only. It must
not select forcing values, policy controls or empirical winners.

The main matrix remains the published section6.2/Figure6 comparison of log
normalizer variance. Pinned source cSMC_VaryF.m:72–112 specifies512 particles,
maximum4 refinements,40 equally spaced forcing values2.5–8.5 and100 repeats.
APF_VaryF.m:81–112 selects1382,2867,4505,6144 APF particles according to each
cSMC run's refinement count. These match author CPU costs, not local GPU costs.
Retain this distinction; observed GPU times will accompany any later result.

## Evidence contract and assumptions

Run4 fresh paired repetitions at source-grid indices0,15,39 (forcing2.5,
4.8076923076923075,8.5), using the same source101-observation dataset, d8,
initial covariance.01I,Q.001I,R1e-4 and the validated FP64 centered-QR route.
Candidate seeds use base202609370 and distinct deterministic offsets by grid,
repeat and method; these are convenience reproducibility choices, not tuned.
The central pilot value is on the actual grid, unlike the data-generating4.8801
used in earlier identity checks. Extremes cover different refinement counts.

Comparator: validated source APF with the count corresponding to the completed
cSMC run. Also preserve the initial APF512 estimate from every cSMC history.
Both final and initial likelihoods restore the known observation constant.
No change to solver, damping, rank/precision guard or stopping is permitted.

Primary pilot pass: all12 controlled/APF pairs valid, all claimed stable
functions traced once, complete synchronized timing and memory/provenance.
Numerical validity and unexpected retracing are promotion vetoes and repair
triggers. Wrong source data, corrupted records or budget exhaustion are
continuation vetoes. A candidate failure triggers localized diagnosis; it does
not invalidate the research direction. Save every partial row before stopping.

Compilation and steady execution times are separate. Forecast4000 pairs using
the largest measured warm controlled and APF cost, add40 times the largest
first controlled-call cost and160 times the largest first APF-call cost, then
double for uncertainty. A missing warm timing blocks a numerical forecast.
This conservative planning estimate is explanatory, not a performance claim.
A full matrix needs a subsequent explicit evidence contract and budget allocation.

Heuristic adversaries for the later empirical comparison, constructed from
this model: BPF512 (no learned policy), APF512 (same initial particle budget),
and the source count-matched APF (local observation conditioning with the
author's extra particles). Conditional situations are every forcing grid point,
with central versus extreme forcing also reported; do not pool away failures.
A full quality claim requires these comparisons and uncertainty, not this pilot.
No exact normalizer oracle has yet been established for this nonlinear model.

Default/assumption audit: source dataset/model/counts/stopping are replication
baselines; centered QR is the checked equivalent-objective numerical repair;
FP64/TF32-off is a source-reference exception; seed and pilot subset are
convenience choices restricted to feasibility. Stable one-system-per-call
graphs reuse compiled kernels across repeats; Python orchestrates independent
runs and writes artifacts, never particle/time/refinement computations.
No NumPy runtime, pfor, package changes, HMC or new default.

Pre-mortem: compilation can dominate short timing, and untested forcing points
can take more refinements or fail. Four repeats per point separate cold/warm
calls; the forecast uses maxima and a factor2 but cannot certify whole-grid
validity. New failures are retained, never excluded to improve a forecast.

## Execution and review

Phase c2_lorenz_figure6_pilot:450 GPU seconds total, at most2 attempts, within
CPU500/GPU18000 allocation. Raise the attempt ceiling72->74 solely for this
bounded pilot; no new overall grant. Execute with trusted GPU access and
verified memory growth. Save structured rows, summary, full logs, source/input
hashes and manifest under unique attempt directories in the existing campaign.

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c2_lorenz_figure6_pilot

Skeptical audit PASS: the pilot answers feasibility, not variance reduction;
source APF counts are not misrepresented as GPU-cost matching; no weak baseline
is used to promote quality; stable graphs and synchronized timing are checked;
stop conditions and total budget are explicit. The earlier70 failure remains
visible and72's revised contract is the prerequisite.

## Terminal result73

All12 pairs valid; stable traces1. Charged174.375847 GPU seconds. Worst warm
controlled15.866031s, APF0.477008s; cold24.274194s/4.198998s. Conservative
full forecast134029.929s exceeds remaining116809.511s. No full matrix launch.
Next is the scoped independent-repetition batching repair in
controlled-smc-lorenz-batch-repair-20260930.md. Pilot figures are feasibility
evidence only; the favorable runtime outcome required for full execution did
not pass. Peak TensorFlow allocation509352704 bytes.
