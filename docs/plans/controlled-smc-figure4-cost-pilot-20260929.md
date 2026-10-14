# Controlled SMC Figure 4 cost pilot

This is the next bounded execution phase of the authorized controlled-SMC/KDM
master. It measures feasibility of the actual Figure 4 right experiment after
the first validation tranche passed. It does not estimate performance rankings
or complete empirical replication.

## Evidence contract and research intent

Question: what would the source's 20-variance, 100-repetition experiment cost
on the validated scalar TensorFlow/GPU/XLA consumer? The mechanisms are the
source's three backward refinements with 128 particles and its bootstrap
comparator with 5529 particles. The source neuroscience data, alpha=.99,
initial N(0,1), binomial trial count 50 and horizon 3000 are fixed.

Run two independent repetitions at variance .01, .10 and .20. These cover the
endpoints and middle of the source grid; they are cost probes, not a reduced
claim experiment. Use seed 20260929 and streams starting at 100000, disjoint
from prior checks and future claim streams. The first repetition measures cold
compilation plus execution; the second measures execution on the same graphs.
Each repetition resets the policy. Every refinement uses fresh random arrays.

Primary engineering pass: all prescribed calls finish with finite values,
valid fitted policies and proposals, one trace per invoked signature, complete
timing records and recorded GPU memory growth. Numerical invalidity, missing
rows, changed source/data identity or corrupt outputs veto this pass and stop
the affected run. A failed candidate triggers diagnosis; it does not reject
controlled SMC. ESS, regression margins, projection counts and log likelihoods
are explanatory only. No stochastic ranking follows from two repetitions.

The heuristic adversary set for this pilot is the source BPF N5529 and the
uncontrolled N128 first pass, the latter already produced before refinement.
Their purpose here is timing and validity. The empirical phase must additionally
construct the observation-guided classical comparators required by the master
before superiority/default claims. Noise settings are reported separately;
informative/extreme-observation performance is not evaluated by this cost probe.

Report a conservative planning estimate, not a runtime guarantee:
20 times the largest observed combined cold cost plus 20*99 times the largest
observed combined warm cost, multiplied by 1.5, plus 60 seconds. Also report the
available balance without authorizing an automatic full launch. A full launch
requires its own implemented comparator/statistical contract under the existing
owner authorization. Passing this pilot proves neither unbiasedness nor
reproduction of Figure 4's variance curve.

## Defaults, failure risks and skeptical review

| Choice | Provenance and status | Risk and early check |
|---|---|---|
| N128/I3, BPF N5529, alpha, data, variance grid | Published Figure 4 and author NormConst_cSMC/BPF.m; source settings | A demo had N1024/I2; assert the actual experiment sizes in every row |
| Scalar TF consumer, FP64, TF32 off, XLA | Passed source replay and exact Kalman controls; scoped implementation | N128 may expose poor design rank; retain minimum Gram ratio and projection count at every fit |
| Source precision projection and normal equations | Explicit source comparison, not a universal default | Projection can change unstable fits; count and report it, stop on invalid accepted results |
| Two repetitions, three variances | Convenience cost design only | Cannot rank methods or estimate tail runtime; use maximum observed costs and 1.5 planning factor |
| BPF implemented by zero-policy forward call | Same source cSMC_Resample endpoint with I=0 | Check BPF uses no fitting and exactly one forward call |

Pre-mortem: compilation could dominate one short probe, thermal/load variation
could make extrapolation optimistic, and the BPF implementation retains full
clouds that a production streaming implementation could omit. Record cold/warm
times, allocator peaks and source snapshots. Do not claim equal GPU cost merely
because the paper chose N5529 to match its original CPU cost. Do not optimize
or select a variant using these likelihood outcomes.

Skeptical audit PASS for cost/validity only: the baseline, source sizes and
measured quantity now match the intended next question. The unresolved
statistical/heuristic requirements are explicit and prohibit promotion from this
pilot. Existing source and Kalman checks support reusing the unchanged core;
this harness changes configuration and orchestration only.

## Execution and budget

Maximum two attempts and 300 GPU-charged seconds in total, drawn from the
existing campaign and its unchanged 7200-second execution allocation. No
additional numerical CPU allocation. Charge all child/reference/import/compile
wall time as GPU time. Preserve failures in unique attempt directories.

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c3_cost_pilot

Use trusted GPU permissions. The runner freezes this plan, the driver, kernel,
source settings and input hashes in the output directory under
docs/plans/artifacts/controlled-smc-kdm-replication-20260928/. Required outputs:
manifest.json, rows.jsonl, observations.json, summary.json and command.log.
Stop at completion or the budget/validity veto. Record the measured feasibility
and next implementation requirement in the master and checkpoint.
