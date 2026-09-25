# M12 preparation trajectory diagnosis and sequential metric probes

Status: complete as a bounded stage; preparation robustness remains open. The active
[master program](bayesfilter-hmc-repair-master-program-2026-09-16.md) contains the
pre-run contract, amendments, skeptical audits and fixed compute allowance.
Outputs are under `artifacts/hmc-repair-master-2026-09-16/m12-r1/` (R below).

## Diagnosis

M11 fresh fit 2 failed because two rejected leapfrog trajectories drove the
regression scale outside the range representable in float64. It did not fail
because R-hat was large or because retained states were nonfinite.

The diagnostic replay uses M11's executed frozen source, original seed, data,
initial position, L=25 and configuration. The three completed window payloads
match exactly except runtime. It reproduces the two nonfinite log acceptance
values at indices 31 and 153 of the third slow window. Every retained position,
target and score is finite. The two invalid proposals are rejected.

A separate TensorFlow/XLA reconstruction follows the installed TFP leapfrog
equations and stores every substep. It exactly reproduces the preceding finite
transition's final position, value, score and momentum, and both invalid
endpoints including their nonfinite masks. The first nonfinite target occurs
at zero-based leapfrog indices 18 and 20. At these evaluations, log sigma is
approximately -4.94e53 and -5.61e30. The target computes
`exp(-2*log_sigma)`; its exponent exceeds the FP64 exponential limit, about
709.78, by many orders of magnitude. This is observed proposal instability.
The target's retained values are not implicated by this evidence. Installed
TFP `safe_sum` replaces the nonfinite Metropolis energy sum with negative
infinity, which rejects the proposal. BayesFilter's finite-window requirement
then rejects the preparation.

`R/trajectory-assessment.json`, the full window tensors and the substep JSON
records preserve this result. The first attempted replay instead hit a
diagnostic-writer dtype error after the first window; it supplies no trajectory
diagnosis and remains charged. The corrected typed writer has a focused test.

## Implemented hypothesis and limits

The existing metric-boundary qualification can now run each independent
momentum probe as a short fixed-step, fixed-L chain. The public option is
`HMCKernelTuningConfig(metric_probe_num_results=16)`; the default remains one.
The value sixteen is a convenience allocation under this experiment, not a
calibrated safety guarantee or promoted numerical default.

Longer probes check every proposed and retained position, endpoint target and
score, initial/final momentum, energy correction and acceptance calculation.
Retained-state inconsistency stops preparation. Only a normally returned
inconclusive bracket can reject a proposed metric and retain the incumbent;
an exception from an optional sequential probe is fatal. Target-status checks
honor the existing declared policy. Shared target and execution failures
cannot be turned into a smaller-step success.

Probe length is carried through ordinary preparation configuration and saved
with the attempts. Readers accept known historical one-transition payloads,
validate positive lengths and reject changed probe designs within a search.
The one-transition numerical branch keeps its original operations and seeds.
The optional path is not qualified for the separate G2 engineering route and
fails before that route executes. No new tuner, R-hat admission condition,
posterior policy or automatic invalid-window retry is introduced.

The matched repaired preparation completes with three qualified updates and
final epsilon 1.0172105007254313. Its first two boundary searches each reject
one nonfinite attempt before qualifying a step. This is a matched engineering
result; its changed early trajectory cannot establish why the original later
trajectory would have behaved differently at a particular fixed epsilon.
Two of three fresh fits failed later windows. Therefore longer local
probes do not close the preparation robustness gap.

The reserved replay of fresh fit 0 reproduces both completed windows exactly
except timing and the failure at index 19 of the following slow window.
The preceding finite endpoint and the failed endpoint again match the
independent reconstruction exactly. At leapfrog index 18, log sigma reaches
-633.2078455180368, so `exp(-2*log_sigma)` overflows. Retained states, values and
scores remain finite. This is another counterexample to the same local-probe
stability assumption, not evidence of a new target or retained-state defect.

## Execution evidence

Baseline Git commit: `d86dadf68ea57772642c6802990f46a3c6a04c30` plus source
snapshots; the dirty tree also contains unrelated q20 work, preserved unchanged.
M12 source `source-gpu-r1` has identity
`53dab3c1c903ecd8a0ddd3c47fc655c69798638adf09e281cfe3f9419c7c963b` and executes
the matched repair and fresh fits. Final source `source-final-r2` has identity
`1df5152260fd54197830d54cfce9d271cb37eb50c089e78c8cf0e53c5a67b3fc`.
The only package difference is a wrapper that makes all exceptions from a
longer probe fatal at a metric boundary. This strengthens error handling and
changes no finite-input arithmetic or seeds. The earlier GPU fits are not
evidence that this final exceptional branch executed; live injection tests are.

The installed environment is `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`,
Python 3.13.13, TensorFlow 2.20.0 and TFP 0.25.0. Numerical runs use trusted GPU
2 with XLA, TF32 recorded and verified memory growth before initialization.
CPU tests intentionally hide all GPUs. Exact commands, seeds, source hashes,
input versions, wall times and device settings are recorded per attempt.
The target remains pinned posteriordb `sblrc-blr`, using the checked observed-data
initialization and no supplied geometry hint. Fresh streams use root 2026092060
and domain `m12-sequential-probe`, replications 0, 1 and 2.

The whole-fit criteria are unchanged: retain every verified candidate, select
the first sorted verified identity before posterior sampling, apply separate
posterior equilibration and lugsail precision checks, and compare all six
quantities against the uncertainty-bearing ten-chain Stan reference. A
preparation failure stays a failed fit; it supplies no posterior comparison.

| Fresh fit | Qualified metric updates | Candidate records / verified members | Warmup / retained per chain | Result |
| --- | ---: | ---: | ---: | --- |
| 0 | 1 | Not reached | Not reached | Preparation veto: one nonfinite log acceptance, first index 19 of the next slow window |
| 1 | 3 | Not reached | Not reached | Preparation veto: two nonfinite log acceptances, first index 4 of the final window |
| 2 | 3 | 100 / 24 | 2000 / 1500 | Posterior health, equilibration, lugsail precision and all six reference comparisons pass |

The passing fit's maximum retained modern R-hat is 1.0072976714061435.
This belongs to the separate posterior screen and never changes candidate
admission or the two failed preparation outcomes.

All 24 verified candidates remain recorded. The first sorted identity was
selected before posterior sampling; the other 23 members were not assessed
for posterior accuracy. The three fits are a development check, not a
powered comparison against M11's different seeds. Their one-pass/two-failure
count is descriptive and cannot establish a reliability ranking.

| Decision | Primary criterion | Veto evidence | Main uncertainty | Next action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept the M11 failure diagnosis | Same completed windows and invalid indices; reconstructed endpoints match | Failed trajectories remain vetoed; retained values finite | Full substep parity on other targets is unchecked | Check whether a fresh failure has the same mechanism | Target invalidity or posterior failure |
| Keep sequential probes experimental | Matched preparation and one fresh full fit complete | Two of three fresh preparations fail | Reliability across seeds and geometry | Design bounded recovery from a validated checkpoint | Default readiness or guaranteed numerical health |
| Preserve strict invalid-window rejection | No failed warmup window reused | Nonfinite log acceptance vetoes the attempt | A bounded recovery rule is not yet qualified | Design a separately reviewed discarded-window recovery after diagnosis | Permission to accept invalid tuning evidence |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Matched repair passes; fresh failures remain explicit in the denominator |
| Statistically supported ranking | None; no powered comparison |
| Descriptive-only differences | Update counts, acceptance, runtime and observed success/failure counts |
| Default readiness | Not established; default probe length remains one |
| Next evidence needed | Reproduced fresh-failure diagnosis, bounded recovery invariants and fresh complete-fit validation |

## Terminal review and continuation

All eight allocated GPU attempts are terminal. The inactive-option check on
the final source reproduces all five saved latent tensors, final affine
transform and epsilon from M11 exactly. The longer-probe implementation has
not changed this default-path trajectory. Configuration propagation, finite
controls, intermediate numerical corruption, retained-state inconsistency,
malformed target-status telemetry, fatal exception propagation, historical
readers and complete artifact round trips have focused coverage. Together
with the affected preparation/public-interface tests there are 383 distinct
passing identities and no unresolved failures. This is not a sum of overlapping
pytest totals. The original faulty test expectation remains in the log and
is superseded by its corrected test and an explicit accepted-state corruption
test; all failed test runs remain charged.

`R/reconciliation-terminal.json` verifies three snapshots, one complete
candidate inventory, 175 numerical receipts and 240 tensor checksums. There
are no corrupt artifacts, outstanding work items or active M12 workers.
The prior diagnostic dtype error is classified as a harness failure, not
numerical evidence. The original M11 and M12 numerical failures remain
failures in their respective fit denominators.

| Worker-wall seconds | M12 charged | Cumulative charged | Remaining campaign allowance |
| --- | ---: | ---: | ---: |
| CPU reference/tests/build/overhead | 973.5593341009226 | 109702.67083916246 | 149497.32916083754 |
| GPU | 1333.359057117952 | 115821.77311925084 | 56978.226880749164 |

The CPU charge includes 600 seconds of predeclared inspection/accounting
overhead. Approximately 41.53 CPU and 15.83 GPU worker hours remain. The
stage stays within its 5000/9000-second allocation and eight-attempt cap.

The agent guide and official LaTeX chapter agree on the optional preparation
control and its limits. The rebuilt 566-page `docs/main.pdf` was inspected
at physical page 412 and installed, SHA-256
`916f4d25c16d422fae5e31fb960fd862bdab5e42141c0d893bb06edfcdafadf9`.
Three preexisting unresolved citations remain Afshar2015, Gorinova2020 and
Pakman2014. Build inputs, logs and the rendered page are in `R/guide-r1/`.

The next repair is a bounded discarded-window or preparation restart from a
validated checkpoint, with all failed evidence preserved. Its design must
specify a genuinely contracted step search that cannot expand back to the
rejected ceiling, fresh streams, cleared adaptation statistics, explicit
attempt/work limits and continuous metric/state lineage. It must explain how
a high-acceptance preparation step can nominate later work without relaxing
final candidate acceptance. This recovery is not implemented or authorized
by a probe passing; it requires its own reviewed contract within the remaining
campaign allocation. The master records this as the first continuation task.

Post-run red-team interpretation: a successful matched replay may depend on
changed early momenta and trajectory, while finite short probes can miss
regions encountered later. The fresh failures support that alternative. They
reject sufficiency of the tested probe allocation, not the regression target,
HMC, or metric adaptation as a research direction. A bounded restart must
preserve the failed draw/trace evidence and validated state, discard failed
adaptation statistics, use fresh streams and retain all later numerical and
posterior checks. Replacing a failed seed or relaxing a trace requirement
would not answer the robustness question.
