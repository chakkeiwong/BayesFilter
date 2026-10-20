# Direct author-code Lorenz audit — 30 September 2026

The owner's question exposes an unfinished baseline: the unchanged author
consumer was exercised for one refinement at F=4.8801 on two seeds. Our later
four-refinement checks use a new centered QR/SVD implementation. Neither
establishes that the author's full adaptive source fails or succeeds in the
regime where our implementation fails. Check the original source before
designing another safeguard.

## Evidence contract and intent

Question: does the pinned, unchanged Lorenz consumer complete valid adaptive
runs at F=2.5,4.8801,8.5 with the published settings? The comparator is the
actual author source and its own generated clouds. Execute the setup prefix
of cSMC_VaryF.m, then call its unchanged cSMC_Resample.m, RungeKutta4.m and
Systematic_Resampling.m on the bundled SimulatedData.mat. Preserve the source
normal equations, all cross terms, unnormalized observation density, N512,
d8, T100, maximum four refinements and minimum-ESS>90% stopping rule.

Primary pass: every declared case returns real finite policies, clouds,
weights, ESS and logZ, with positive definite Gaussian precisions at every
retained stage and time. A source exception, invalid precision, nonfinite
value or complex output is a source-run failure and blocks interpreting that
case as a successful replication. Wrong callable paths or settings, a missing
artifact, or timeout invalidate the diagnostic itself. Continue the other
predeclared cases after a source exception; do not repair the source.

Counts, final logZ, ESS, covariance margins and warnings are explanatory only.
No stochastic ranking, GPU parity, complete Figure6 replication, default
promotion or correctness of the author's archived experiments follows.
Fresh Octave streams are not identical to TensorFlow streams: a source success
does not establish success on attempt76's exact random inputs. An Octave
failure does not establish the same failure in MATLAB.

## Defaults, audit and budget

F=2.5 and8.5 are the original grid endpoints;4.8801 is the data-generating
parameter, retained as an interior control. Seeds20260929/30 are reproducible
diagnostic conveniences, not reported author seeds or tuned choices. Early
checks freeze the actual source setup and resolved call paths. Octave with the
already installed/pinned statistics package is an explicit independent CPU
reference exception; CUDA_VISIBLE_DEVICES=-1, one BLAS thread. Record versions,
source/data hashes, full command logs, every successful source return and
each exception. No dependency install or source modification.

Skeptical audit PASS for this narrow source-execution question: this directly
addresses the missing comparator, uses the original method/settings and data,
and keeps RNG/runtime limitations explicit. The test neither compares efficacy
nor ranks an optimized method, so heuristic-dominance and uncertainty-based
ranking criteria are not applicable. The central pre-mortem is mistaking six
working runs for a replicated paper or a proof that the TF failure is absent
in the source; both inferences are forbidden.

One new campaign attempt85, maximum90 CPU wall seconds, zero GPU seconds,
inside the existing CPU500/GPU18000 allocation and inherited grant. The
attempt ceiling changes from84 to85; compute allocations stay unchanged.
Command: /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B
scripts/run_controlled_smc_kdm.py run --phase c1_lorenz_author_native
Output: next unique campaign attempt directory, manifest, summary.json,
progress.json, command.log and original author result files.
Afterward separate source success/failure from port divergence and reconsider
the next master step. Do not automatically resume projection tuning.

Attempt85 infrastructure failure: Octave completed the first author case but
does not implement jsonencode. Source results are preserved; no source
algorithm failure is inferred. Repair only the output boundary to MAT plus
SciPy decoding (independent diagnostic use), as in the existing source harness.
Charge3.473717547 CPU seconds; retry once as86 under the same90-second aggregate
cap, data, seeds, methods and criteria. Syntax check passed; no source edits.

Terminal:86 passes all six unchanged-source cases in13.042128388 seconds.
Total attempt85–86 CPU charge16.515845935 seconds; no GPU use. All original
algorithm/data hashes match; the pinned author checkout has no tracked changes.
See controlled-smc-lorenz-author-native-result-20260930.md. The next priority is
matched-input source/port comparison, not another ungrounded safeguard.
