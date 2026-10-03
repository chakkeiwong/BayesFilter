# Rectangular SRUKF SVD cost result and review

All 48 fresh-process workers completed: CPU 03561--03584 and GPU 03585--03608,
three observations for each of four arms and two horizons. The candidate graph
and XLA arms and the prior graph arm pass the independent Kalman comparison.
Every prior-XLA arm fails it. Preserve those failures; no speed ratio or ranking
against prior XLA is eligible. This is a correctness repair with descriptive
resource evidence, not terminal filter or analytical-gradient qualification.

The baseline is f4ea46dde, with original rectangular filter/factor modules and
an exact hash check on the unchanged cubature dependency. All 48 workers share
the same candidate source hashes. These hashes predate the later reporting
regression registration and repair. The full commands, environment, source
hashes, device settings and durations are in each run.json. The single physical
GPU is GPU2, UUID GPU-541e1e19-2df4-9064-4db9-9d0d2abc3eba; growth and clean
sharing observations pass for the complete GPU cohort. CPU intentionally hides
CUDA. Inputs are fixed deterministic linear 3-state fixtures, so no random
seed or tuning is applicable. The measured boundary owns one reusable outer
filter and materializes its outputs; it does not rebuild a public API per call.

The evidence root is `artifacts/filter-gradient-repair-20260917/` in the live
BayesFilter checkout. `svd-cost-complete-03608.json` checks the raw numerical
outputs independently of their saved pass flags, repeat/source/environment
identity, physical UUID, growth and sharing. Its analyzer SHA-256 is
5faf609f85338b91d6ab6dd88b631a5e4b109880e47b4827c2958ce651f23117.
Eight analyzer adverse checks pass in 03560. Prior-XLA covariance error reaches
1.884e-9 at T=1 and likelihood error reaches 1.663e-7 at T=3, above the fixed
1e-10 absolute/relative screen. No tolerance has changed.

Cold time is construction/trace plus first synchronized call. Warm time is the
median of each process's 20 calls, then the median across three processes. RSS
is the largest observed snapshot, not an exact native peak.

| Device | T | Arm | Numerical screen | Cold seconds | Warm ms | RSS MiB |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| CPU | 1 | prior graph | pass | 0.187 | 0.922 | 595.73 |
| CPU | 1 | prior XLA | fail | 0.754 | 0.729 | 790.38 |
| CPU | 1 | candidate graph | pass | 0.178 | 0.931 | 595.73 |
| CPU | 1 | candidate XLA | pass | 0.834 | 0.741 | 796.23 |
| CPU | 3 | prior graph | pass | 0.182 | 1.454 | 595.88 |
| CPU | 3 | prior XLA | fail | 0.801 | 1.249 | 794.75 |
| CPU | 3 | candidate graph | pass | 0.190 | 1.383 | 596.00 |
| CPU | 3 | candidate XLA | pass | 0.844 | 0.948 | 800.50 |
| GPU | 1 | prior graph | pass | 1.959 | 4.750 | 1073.43 |
| GPU | 1 | prior XLA | fail | 2.559 | 2.136 | 1034.79 |
| GPU | 1 | candidate graph | pass | 1.942 | 3.622 | 1074.82 |
| GPU | 1 | candidate XLA | pass | 2.575 | 2.302 | 1038.07 |
| GPU | 3 | prior graph | pass | 1.937 | 7.092 | 1076.64 |
| GPU | 3 | prior XLA | fail | 2.553 | 4.818 | 1033.93 |
| GPU | 3 | candidate graph | pass | 1.982 | 8.844 | 1075.25 |
| GPU | 3 | candidate XLA | pass | 2.650 | 5.605 | 1039.34 |

Candidate-XLA GPU allocator peaks are 19,456 / 19,968 bytes at T=1/3;
graph peaks are approximately 8--16 MiB. These are allocator statistics, not
nvidia-smi reservation or total CUDA-context residency. Maximum observed warm
RSS growth is 0.148 MiB over 20 calls. This short observation cannot prove native
executable eviction, asymptotic stability or capacity with many signatures.

CPU graph-to-XLA cold ratios of 4.70 / 4.43 trigger investigation. Trace times
are close (about 0.11--0.15 seconds in the final repeats); most of the difference
occurs in the first call (graph 0.07--0.08 seconds versus XLA 0.70--0.72).
Observed host differences are 200--205 MiB and lie mainly after first execution.
This localizes the cost to the compilation/first-execution lifecycle; it does
not distinguish all native compiler/cache allocations. Retain a reusable owner
for the tested signature; do not reconstruct it per call. Full consumer capacity
and process-containment integration remain required before terminal disposition.

The GPU T=3 prior-graph-to-candidate-graph warm ratio is 1.247 and also triggers
investigation. Individual process medians are 7.092/6.947/8.678 ms before and
8.844/11.738/8.665 ms after. The source graph branch delegates to the same
TensorFlow SVD; both recorded graphs have 314 nodes and 36,361 bytes. Counts and
source inspection alone do not prove graph identity or establish the cause of
the timing difference. Do not ascribe it to device load without evidence. The
next discriminating diagnostic must compare the actual traced operations and
interleave original/candidate calls with synchronized outputs and sharing
observations. Preserve the trigger until that result is assessed.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Nonclaim |
| --- | --- | --- | --- | --- | --- |
| Retain repaired candidate for qualification | Independent Kalman checks pass in both modes/backends | No candidate numerical veto here | Broader endpoints and gradients | Continue original-record and consumer renewal | No terminal default readiness |
| Exclude prior-XLA ranking | Raw values fail the unchanged screen | Numerical veto present | Other fixtures may fail differently | Preserve failures and repaired value evidence | No performance benefit over an inaccurate method |
| Keep CPU lifecycle disposition open | Cold ratio exceeds 2x | Investigation trigger, not a numerical failure | Native allocation attribution and full consumer capacity | Reuse compiled owner; finish containment/capacity integration | No native eviction claim |
| Keep GPU graph timing disposition open | Warm ratio exceeds 1.2 at T=3 | Investigation trigger present | Same graph versus scheduling/runtime cause | Actual graph comparison and bounded interleaving | No unexplained regression waiver |

| Inference status | Result |
| --- | --- |
| Hard veto screen | Prior XLA fails independent numerics; candidate and graph references pass |
| Statistically supported ranking | None; three descriptive process observations do not establish superiority |
| Descriptive differences | Candidate graph/XLA cold, warm, RSS and allocator values above |
| Default readiness | Open; public/actual consumers and remaining cost/source gates remain |
| Next evidence | GPU graph attribution, full consumer lifecycle/capacity, terminal source renewal |

Review: the strongest misleading interpretation would credit the candidate's
speed against an inaccurate baseline, or read a tiny allocator peak as total GPU
memory use. Both are explicitly excluded. The weakest evidence is the
unattributed graph timing difference and short residency window. A mismatched
numerical, source or sharing record would invalidate the matched comparison;
the analyzer checks those conditions. This primary-agent review does not supply
independent certification or close the other 28-call SVD audit obligations.
