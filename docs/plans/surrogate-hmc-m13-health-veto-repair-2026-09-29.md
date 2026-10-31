# M13 energy-veto diagnosis and bounded repair — 2026-09-29

The first48-hour tuning attempt failed its declared energy health check after
250 warmup transitions per chain. No posterior samples were retained. The
failure was one rejected proposal with deltaH1211.0069, above1000. All accepted
proposals had |deltaH|<=4.1384; fresh endpoint, finite-state, movement, divergence
and force-fallback checks passed. Per-chain acceptance was46.0%,64.8%,66.4%,54.8%.
These short-run values describe the candidate; they establish no convergence.

Nomination selected epsilon0.949153975512196,L1 for the equal(1/8,1) step mixture.
Its maximum absolute log acceptance ratio was158.979, below1000. The nomination
code had no corresponding prospective energy guard, but adding one alone would
not have detected this later event. The bounded repair therefore halves the
allowed base step and retains all fresh health checks. The force, approximate
MH endpoint, data and posterior comparison remain unchanged.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Reject this candidate | Warmup readiness not reached | One energy-limit breach; other fresh checks pass | Rare proposal tail and smaller-step mixing | Two discarded energy-repair probes, then public retuning only if healthy | No rejection of surrogate HMC or M13 target |
| Preserve existing success criteria | No retained assessment yet | No threshold relaxation | Final convergence may remain slow | Independent fresh warmup and cumulative verification/retention | No posterior correctness claim |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Current candidate fails declared energy bound |
| Statistically supported ranking | None |
| Descriptive-only differences | Short-chain acceptance and energy distribution |
| Default-readiness | Not established; scoped repair only |
| Next evidence needed | Healthy repair probe, public fresh verification, retained ESS/Rhat/reference assessment |

Archive analysis: artifacts/surrogate-hmc-m13-sequential-48h-development-01/health-veto-analysis-02/analysis.json.
All tensor checksums passed. Analysis01 failed a diagnostic integer-dtype reader;
02 corrected it. Energy-probe01 stopped before inference on a tuple/list JSON
identity comparison;02 normalizes serialization without weakening identity.
Every attempt has a worker receipt and remains charged.

Post-run red team: Metropolis correctly rejected the extreme proposal, so this
does not prove target bias. Conversely, acceptance and finite output do not
override the predeclared energy veto. A shorter step may satisfy that veto while
mixing inadequately; the unchanged downstream criteria must decide.

## Repair implementation and verification

The original-source screen02 reached its CPU budget after one healthy
64-transition replication; its second replication was unfinished. It is not a
qualified repair screen. Its complete receipt charges1780.205717 CPU seconds
and1628.804052 GPU seconds. Preserve its diagnostic archives.

The final implementation adds optional nomination step and energy limits to the
public TensorFlow tuner, caps the M13 second-attempt step at0.474576987756098,
and retains the fixed mixture(0.125,1). The energy limit remains1000.
Finite energy excursions can request the existing bounded factor-two step
repair; non-finite checks remain vetoes. Integration, affine transformations,
force and MH endpoint are unchanged. The master requires a complete healthy
final-source GPU screen before second-attempt sampling, checks attempt2 on
resumed preflight, and keeps cumulative attempt/compute accounting.

Verification: the full master suite recorded171 passes and one compatibility
failure. That failure correctly detected a changed native-tuner source hash in
the saved LEDH control selection. The explicit compatibility proof now accepts
only the reviewed hash pair and verifies that every module statement outside
six nomination/configuration definitions is unchanged. It preserves the saved
selection and evidence, recomputes its decision, and tests both actual control
consumers against the original transition loop. All17 focused compatibility
checks passed, including rejection of changed force, scope, source or evidence.
Two earlier test-fixture defects were corrected and their two reruns passed;
46 other focused repair checks had already passed. The master will rerun its
whole test phase after the GPU screen. Existing source hashes are never
rewritten to impersonate new sources.

Skeptical review: the screen retains attempt1 geometry to isolate the step
change; attempt2 adapts new geometry, so a healthy screen only permits the
planned attempt. It cannot establish energy safety, convergence, posterior
agreement or performance. The full fresh checks and unchanged downstream
criteria still decide qualification. This repair remains within the original
scientific question, method, target, data, hardware and48-hour authorization.

At pre-launch, aggregate charges are CPU24429.854637/GPU23015.114224 seconds.
Probe03 reserves3600 CPU/3000 GPU seconds inside the six-hour development
envelope. The hard deadline remains2026-09-30T16:27:39Z. One full attempt
remains; the ten-hour retained-sampling reserve is protected.
The coordinator will start the existing master automatically only after a
healthy completed screen. This is a phase of the current master program.

The detached coordinator launched under trusted GPU permissions at 2026-09-29T06:00:52.281363+00:00 (PID2147222). Probe03 reached RUNNING with verified GPU memory growth, XLA/TF32 enabled, unchanged target/force identity and unchanged integration code. Its state and automatic continuation are recorded in development/energy-repair-continuation.json. The full master has not yet restarted; the coordinator will launch it only after this screen passes.

## Automatic continuation observed at 2026-09-29 09:32 UTC

The final-source GPU screen completed both64-transition replications with all
declared health checks passing. The largest absolute energy errors were12.2242
and5.08443. Its complete receipt records2293.482517 CPU seconds and2115.630000
GPU/wall seconds. These discarded probes permit the planned repair attempt;
they establish no posterior or convergence claim.

The coordinator automatically started48h-03. Its164 master tests, GPU
qualification and attempt2 preflight passed. The second and last full tuning
attempt remains live, with500 warmup transitions per chain and0
verification transitions archived. Every archived chunk passes the declared
health checks. No retained posterior draws exist; readiness is not yet due.
The owner requested continuation, and the existing master continues without
a duplicate launch or any change to its scientific criteria or deadline.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next action | Unsupported conclusion |
|---|---|---|---|---|---|
| Continue attempt2 | Warmup readiness pending | All archived chunk checks pass | Whether mixing and later energy checks pass | Continue to minimum2000 warmup and modern R-hat check; then independent verification | M13 posterior qualification or all-model completion |

## First warmup-readiness result — 2026-09-29 13:58 UTC

Attempt2 remains live with2250 warmup transitions archived per chain.
At2000, the maximum rank/folded R-hat values on the latest1000 were1.3511476
and1.2512312; both exceed the declared1.05 warmup-readiness threshold.
Every archived chunk still passes the numerical, endpoint, movement, fallback
and energy checks. No verification or posterior samples have been retained.

This result means the chains are not ready for verification. It triggers the
master's already planned warmup extension; it is not a numerical failure or a
continuation veto. No inference about posterior correctness or comparative
performance follows. The live master continues within the existing maximum
and budget, without changing thresholds or starting another attempt.

## Runtime estimate — 2026-09-29 15:48 UTC

The live run has3250 warmup transitions per chain. Its3000-transition
readiness check gave maximum R-hat1.2063962 and1.1617967, still above1.05;
all archived health checks pass. From archive completion timestamps, the12
intervals between250-transition chunks averaged2.23 hours per1000 transitions
per chain; the latest four intervals imply2.34 hours. This measures all four
chains together, not a time to multiply by four.

The next4000-transition readiness check is roughly1.5–2 hours away.
Conditional on passing that check, independent verification at its minimum1000
and the first retained assessment at1000 require roughly another4–6 hours,
with possible compilation and assessment overhead. About6–8 more hours is
therefore an optimistic M13 completion scenario, not a convergence prediction.
Every additional1000 transitions per chain costs roughly2–3 hours at current
rates. The all-model campaign has no completion estimate from these timings.

The budget deadline remains2026-09-30T16:27:39Z, or2026-10-01 00:27:39 Hong Kong.
It limits execution, not the time by which qualification is guaranteed.
The master continues automatically under unchanged criteria.

## Warmup update — 2026-09-29 18:37 UTC

The live master has4500 warmup transitions archived per chain. At4000,
the latest1000-draw readiness window gave maximum rank/folded R-hat values
1.0748322 and1.0167887. The first parameter remains above1.05, so readiness
has not passed. All archived health checks pass. Verification and retained
sampling remain at zero. The master automatically continues to the5000
readiness check, with no change of thresholds or scientific attempt.

The earlier optimistic completion estimate was conditional on readiness at4000.
That condition was not met; it was not a promised deadline. The next check is
roughly an hour away at recent rates, while successful qualification remains
uncertain.

## Terminal review — 2026-09-30

The final permitted attempt ended at06:37:46UTC (14:37:46Hong Kong) with
M13_PUBLIC_TUNING_BUDGET_EXHAUSTED. Its worker exited normally and the master
stopped. Warmup passed at9000 transitions per chain: maximum rank/folded
R-hat1.029793 and1.036218 on the latest1000, below1.05.
Independent verification reached2250 transitions per chain. Its last scheduled
check, at2000, gave1.057747 and1.025917, still above1.01. No posterior samples
were retained.

All45 archived chunks passed every declared health check. The terminal audit
verified450 tensor checksums, sequential state boundaries and45 unique seeds.
The guard receipt is complete with exit0; the master recorded unchanged
sources. The second tuning worker charged85431.117285 CPU seconds and
84785.202148 GPU/wall seconds. Overall extension charges are112433.525712 CPU
and109994.262141 GPU seconds. Against172800-second caps, unused compute is
60366.474288 CPU and62805.737859 GPU seconds, subject to the original deadline.

The tuning phase exhausted its time allowance, which had been reduced to
protect the planned ten-hour retained-sampling reserve. The total allocation
was not fully spent. Retained sampling cannot start without qualified
verification. Both allowed fresh attempts have been used, and the current
master has no further automatic launch. This is a budget-limited inconclusive
candidate, not evidence of a broken harness, invalid target, numerical failure
or rejection of surrogate HMC.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Unsupported conclusion |
|---|---|---|---|---|---|
| Current master stops; M13 remains unqualified | Warmup passed; verification R-hat has not passed; posterior comparison not run | Numerical checks pass; phase budget veto fired | Whether longer verification and retained sampling would meet their criteria | Preserve terminal evidence; evaluate a bounded continuation of the same frozen chain before any revised launch | Posterior qualification, default readiness, all-model success or rejection of surrogate HMC |

| Inference status | Finding |
|---|---|
| Hard veto screen | No numerical or artifact veto in the45 chunks; phase time limit stopped continuation |
| Statistically supported ranking | None; no comparative superiority established |
| Descriptive-only differences | Successive R-hat values describe mixing; they do not predict when a later window will pass |
| Default readiness | Not established; verification and retained posterior criteria remain unmet |
| Next evidence needed | Complete admissible verification, then independently retained draws with the declared R-hat, ESS and reference checks |

Post-run skeptical review: passing one recent warmup window did not imply
independent verification would converge quickly. The failed optimistic runtime
scenario depended on earlier readiness. Slow mixing, the fixed force/endpoint
mismatch and conservative step size remain possible explanations; this run
does not isolate them. Passing continued verification and the independent
posterior comparison would overturn the present nonqualification. The weakest
evidence is posterior accuracy, because no retained assessment was reached.

Terminal machine-readable review:
artifacts/surrogate-hmc-m13-sequential-48h-03/terminal-review-20260930-01.json.
